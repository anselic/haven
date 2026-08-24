use std::collections::HashMap;
use haven_common::ast::*;
use haven_common::defs::{deinstance, DefId, Defs, Instances, Member, MemberTable, ModId,
                         TyHead};
use haven_common::layout::TypeTable;

/// The bare names of a generic parameter list, which is the form [`unify`] wants
/// its free-variable set in.
pub fn param_names<'a>(generics: &[GenericParam<'a>]) -> Vec<&'a str> {
    generics.iter().map(|g| match g {
        GenericParam::Type { name, .. } => *name,
        GenericParam::Const(name, _) => *name,
    }).collect()
}

/// Field index of the discriminant tag in a data-enum aggregate's synthetic
/// struct, and of the payload byte-blob. Referenced by name in the struct table
/// and by index in MIL lowering (`FieldPtr`).
pub const ENUM_TAG_FIELD: &str = "$tag";
pub const ENUM_PAYLOAD_FIELD: &str = "$payload";



/// A generic function's signature, in terms of its own type params. param/return
/// types hold `Type::Param`. Used to typecheck calls before mono materializes the
/// concrete instances
#[derive(Clone, Debug)]
pub struct GenericFnSig<'a> {
    /// Generic parameters in declaration order (type and const params
    /// interleaved). Turbofish arguments are matched against this positionally.
    pub generics: Vec<GenericParam<'a>>,
    pub params: Vec<Type<'a>>,
    pub return_type: Type<'a>,
}

/// How a receiver method call adjusts its base expression to form the `self`
/// argument. `AddrOf` takes the address (a `*self` method called on a value `T`);
/// `AsIs` passes the base's value straight through (a `*self` method called on a
/// `*T`, or a value-`self` method - where the aggregate is already handled by
/// pointer, and a scalar value is passed directly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecvAdjust { AddrOf, AsIs }

/// Resolution of a receiver method call `recv.method(args)`, keyed by the `Call`
/// expression's node id. Produced by the typechecker (which alone knows the
/// receiver's type) and consumed by MIL lowering, which emits a direct call to
/// `target` with the adjusted receiver prepended to `args`. Associated calls
/// (`Type::method`) don't appear here - name resolution already rewrote them to a
/// plain function name.
#[derive(Clone, Debug)]
pub struct MethodCall<'a> {
    /// Final function name to call, e.g. `Point$as_array_ref`.
    pub target: &'a str,
    /// How to turn the receiver base into the `self` argument.
    pub adjust: RecvAdjust,
    /// The method's full parameter types (including `self`), for arg coercion.
    pub param_tys: Vec<Type<'a>>,
    /// The method's return type.
    pub return_type: Type<'a>,
}

/// One required method of a trait, in resolved form (enum names rewritten to
/// `Type::Enum`; a `Self` in a type is left symbolic for per-impl substitution).
/// `params` excludes the receiver, which is captured by `receiver`.
#[derive(Clone, Debug)]
pub struct TraitMethodSig<'a> {
    pub receiver: Receiver,
    pub params: Vec<Type<'a>>,
    pub return_type: Type<'a>,
}

/// A declared `trait`: its required method signatures, keyed by method name.
/// Used to check `extend T: Trait` conformance and to resolve a bounded type
/// param's method call (`x.m()` where `x: T` and `T: Trait`).
#[derive(Clone, Debug)]
pub struct TraitDef<'a> {
    pub methods: HashMap<&'a str, TraitMethodSig<'a>>,
    /// Names of the trait's associated types (`type Item;`). Every conforming
    /// impl must bind all of them; a method signature's `Self::Item` was
    /// resolved to `Param("Item")`, substituted per impl during conformance.
    pub assoc_types: Vec<&'a str>,
}

#[derive(Clone, Debug)]
pub struct Context<'a> {
    /// Lexical scope stack. Each entry maps a name to its binding identity and
    /// type. The binding is `None` for module-level globals/functions (resolved
    /// via the global namespace in codegen, not a local slot) and `Some` for
    /// params and locals.
    pub scopes: Vec<HashMap<&'a str, (Option<Binding<'a>>, Type<'a>)>>,
    /// Map from Expr/Stmt/TopLevel IDs to their inferred types, for use in later codegen
    pub node_types: HashMap<usize, Type<'a>>,
    /// Turbofish arguments recovered by inference for a generic call written
    /// without one (`printf(x)` instead of `printf::<T>(x)`), keyed by the call
    /// node's id and in the callee's declared param order. The AST node still
    /// carries an empty `type_args`, so monomorphization reads the inferred args
    /// from here - exactly as it reads receiver types from `node_types` - to know
    /// which instance a bare generic call asks for.
    pub inferred_type_args: HashMap<usize, Vec<GenericArg<'a>>>,
    /// Name resolution: maps each `Var` use's node id to the specific param/local
    /// binding it refers to. Globals are absent (they fall back to the global
    /// namespace). Consumed by MIL lowering to key variable storage, which makes
    /// shadowing correct - two same-named locals get distinct `Binding::Local`s.
    pub resolved: HashMap<usize, Binding<'a>>,
    /// Layout of every named type, by identity: real structs, data-enum
    /// `{ $tag, $payload }` aggregates, and the synthetic per-variant payload
    /// structs alike, plus each enum's discriminant repr.
    ///
    /// This is the same table `layout`/`abi`/`llvm` take, which is why it holds
    /// `TypeInfo` rather than a bare field list: the repr and `has_payload` flag
    /// used to be copied into every `Type::Enum` value, and now live here once.
    pub types: TypeTable<'a>,
    /// Structs declared with generic parameters (`struct Option<T>`,
    /// `struct Buf<T, const N: u32>`), mapped to their declared params in order
    /// (type and const interleaved). Field types are stored with `Type::Param`s
    /// and `ConstVal::Param`s; a construction/turbofish binds those to concrete
    /// args positionally (the `len()` gives the declared arity). Monomorphization
    /// rewrites a concrete use to a flat instance before codegen.
    pub generic_structs: std::collections::HashMap<DefId, Vec<GenericParam<'a>>>,
    /// Type-param names in scope for the function being checked, e.g. `["T"]`
    /// inside `proc id<T>(...)`. used to resolve a bare `Type::Struct(name)` into
    /// a `Type::Param(name)`. empty outside generics.
    pub generics: Vec<&'a str>,
    /// Const-param names in scope for the function being checked, e.g. `["N"]`
    /// inside `proc f<const N: u32>(...)`. Used to validate that a `ConstVal::Param`
    /// in a type position names a declared const param. Empty outside generics.
    pub const_generics: Vec<&'a str>,
    /// Generic function signatures, by name. NOT callable via the ordinary
    /// `Type::Function` path; calls go through `check_generic_call`.
    pub generic_fns: HashMap<&'a str, GenericFnSig<'a>>,
    /// Names of module-level constants. Used to reject direct assignment to a
    /// `const` global (they are read-only).
    pub global_consts: std::collections::HashSet<&'a str>,
    /// Declared field-less enums, by name. Each carries the discriminant's
    /// integer repr type (from `@repr(<int>)`, default `i32`) and its variants
    /// mapped to their discriminant values. Used to resolve a `Struct(name)` type
    /// to `Type::Enum` and an `E::V` variant reference to its constant.
    pub enums: HashMap<DefId, EnumDef<'a>>,
    /// Enums declared with generic parameters (`enum Option<T>`), mapped to their
    /// declared params in order - mirrors `generic_structs`. A variant constructor
    /// or destructuring pattern binds these positionally via turbofish; monomorphization
    /// rewrites a concrete use to a flat instance before codegen.
    pub generic_enums: std::collections::HashMap<DefId, Vec<GenericParam<'a>>>,
    /// Receiver method calls (`recv.method(...)`), keyed by the `Call` node id.
    /// Populated by `infer` and consumed by MIL lowering. Rebuilt on each typecheck
    /// pass, so it always matches the AST that lowering will see.
    pub method_calls: HashMap<usize, MethodCall<'a>>,
    /// Declared traits, by name. Populated in the forward-declaration pass;
    /// consumed by conformance checking and bounded method-call resolution.
    pub traits: HashMap<DefId, TraitDef<'a>>,
    /// The prelude's `Delete` trait, copied from `Defs::lang` at the start of
    /// each pass. Implementing it is what makes a type own a resource: it stops
    /// being `Copy` (so it moves rather than aliases) and acquires a destructor
    /// the ownership pass calls automatically.
    ///
    /// `None` means the program has no prelude (`--no-prelude`) and therefore
    /// nothing that owns anything - never that the lookup failed, which the
    /// module loader rejects outright.
    pub delete_trait: Option<DefId>,
    /// Which conformances hold, from `extend T: Trait` blocks (verified during
    /// the forward pass). A `T: Trait` bound at a generic call site is satisfied
    /// iff some impl here covers the concrete argument type — see
    /// [`Context::implements`].
    ///
    /// A list rather than a `HashSet<(DefId, DefId)>`: an impl's subject can be a
    /// structural type (`[T]`) with no identity to hash, and deciding whether it
    /// covers a given type takes unification, not a lookup.
    pub impls: Vec<ImplDecl<'a>>,
    /// Trait bounds on the type params of the function currently being checked,
    /// e.g. `{"T": ["Display"]}` inside `proc show<T: Display>(...)`. Lets a
    /// method call on a `T`-typed receiver resolve through the bound trait. Empty
    /// outside a bounded generic.
    pub generic_bounds: HashMap<&'a str, Vec<DefId>>,
    /// Every method and associated function in the program, keyed by
    /// `(type name, method name)`. Built by the module resolver, which knows each
    /// method's emitted name directly - so neither receiver-call resolution nor
    /// conformance checking has to rebuild `Type$method` and hope it exists.
    pub members: MemberTable<'a>,
    /// The module owning the function currently being checked. A receiver call
    /// `recv.method()` reaching a private method (see [`Member::is_pub`]) declared
    /// in a different module is an error; this is the module the comparison is
    /// against. Set before walking each top-level item's body.
    pub current_module: ModId,
    /// Every monomorphized instance, by its own identity. Recorded by `mono`;
    /// empty on the pre-mono pass, where no instance exists yet.
    ///
    /// Two things need it. A match pattern always names the template, so
    /// matching it against a scrutinee whose type is the instance goes through
    /// `template`. And every table keyed on a type - members, conformances - is
    /// keyed on the template too, so a post-mono lookup has to put the instance
    /// back into template form first, which needs `args`.
    pub instances: Instances<'a>,
    /// `(enum, variant) -> the synthetic struct holding that variant's payload`.
    /// Minted at name resolution and by `mono`, so the payload struct is a real
    /// identity rather than a name this stage rebuilds - which matters because
    /// this pass runs twice, and the second run must land on the same one.
    pub payloads: HashMap<(DefId, &'a str), DefId>,
    /// How each definition reads in a diagnostic: `std/string::String`, or
    /// `alloc::<Vec2>` for something `mono` minted. A read-only projection of
    /// `Defs`, rebuilt at the start of each pass, so a message can name a type
    /// without the whole typechecker having to borrow `Defs`.
    pub names: HashMap<DefId, String>,
}

/// A declared enum's definition: the discriminant repr, variant discriminant
/// values, and (for data-carrying variants) their payload field types.
#[derive(Clone, Debug)]
pub struct EnumDef<'a> {
    pub repr: Type<'a>,
    pub variants: HashMap<&'a str, i64>,
    /// Payload fields per variant, keyed by variant name: `(field_name, type)` in
    /// declaration order. Tuple variants get synthesized names `"0"`, `"1"`, ...;
    /// struct-style variants keep their written names. A unit variant has an empty
    /// vec (or no entry). Types are resolved (enum names rewritten to `Type::Enum`).
    pub payloads: HashMap<&'a str, Vec<(&'a str, Type<'a>)>>,
    /// `true` if any variant carries a payload - the enum is then an aggregate
    /// (`Type::Enum { has_payload: true }`) rather than a bare scalar discriminant.
    pub has_payload: bool,
    /// Whether an `@repr` attribute was actually written on this enum (vs. the
    /// implicit default `i32` tag). A data-carrying enum must have this set to
    /// cross an `@export`/`extern` boundary (see `check_export_type`) - mirrors
    /// Rust's requirement that `#[repr(C)]` be written explicitly before an enum
    /// is treated as a committed FFI layout, even though the layout itself
    /// (`{ tag, payload }`) is identical either way.
    pub has_explicit_repr: bool,
}

impl<'a> Context<'a> {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()], // global scope
            node_types: HashMap::new(),
            inferred_type_args: HashMap::new(),
            resolved: HashMap::new(),
            types: TypeTable::new(),
            generic_structs: std::collections::HashMap::new(),
            generics: Vec::new(),
            const_generics: Vec::new(),
            generic_fns: HashMap::new(),
            global_consts: std::collections::HashSet::new(),
            enums: HashMap::new(),
            generic_enums: std::collections::HashMap::new(),
            method_calls: HashMap::new(),
            traits: HashMap::new(),
            delete_trait: None,
            impls: Vec::new(),
            generic_bounds: HashMap::new(),
            members: MemberTable::new(),
            // overwritten before any body is walked; a placeholder until then.
            current_module: ModId(u32::MAX),
            instances: Instances::new(),
            payloads: HashMap::new(),
            names: HashMap::new(),
        }
    }

    /// The method `name` reachable on a receiver of type `ty`, together with the
    /// bindings that specialize it to this receiver.
    ///
    /// Two steps, because the member table's key is deliberately coarse: the
    /// head narrows every impl in the program down to at most one candidate, and
    /// unification then both *confirms* the candidate applies (an `extend [i32]`
    /// must not answer for a `[f32]`) and recovers the type arguments the head
    /// discarded. For a non-generic target unification degenerates to equality,
    /// which is exactly the old `DefId` comparison.
    pub fn member_for<'s>(&'s self, ty: &Type<'a>, name: &'s str)
        -> Option<(&'s Member<'a>, Unified<'a>)>
    {
        // after mono the receiver names an instance (`Scale$f32`) while the
        // impl is registered against the template it came from (`Scale<f32>`),
        // so both the head lookup and the unification want template form. Before
        // mono, and for anything that is not an instance, this is the identity.
        let ty = deinstance(&self.instances, ty);
        let m = self.members.get(&(TyHead::of(&ty)?, name))?;
        let params = param_names(&m.generics);
        let mut u = Unified::default();
        unify(&m.self_ty, &ty, &params, &mut u).then_some((m, u))
    }

    /// Whether `ty` implements `trait_`, by some `extend ...: Trait` block.
    ///
    /// A generic impl covers every type it unifies with *and* whose `where`
    /// clause it satisfies, so one `extend Vec<T>: Delete where T: Delete`
    /// answers yes for `Vec<Res>` and no for `Vec<u8>`. See
    /// [`ast::implements`](haven_common::ast::implements), which mono shares.
    /// A bound on a type parameter of the function being checked counts: it is
    /// what every instantiation is separately checked against, so `A: Mono`
    /// makes `Serial<A, Gain>` a `Mono` under `extend Serial<A, B>: Mono where
    /// A: Mono, B: Mono` while `A` is still symbolic.
    pub fn implements(&self, ty: &Type<'a>, trait_: DefId) -> bool {
        // conformance is recorded against the template, for the same reason
        // membership is - see `member_for`.
        haven_common::ast::implements(
            &self.impls, &self.generic_bounds, &deinstance(&self.instances, ty), trait_)
    }

    /// Load the diagnostic name of every definition. Called once per pass.
    pub fn load_names(&mut self, defs: &Defs<'a>) {
        self.names = (0..defs.len() as u32)
            .map(|i| (DefId(i), defs.show(DefId(i))))
            .collect();
    }

    /// How a definition reads in a diagnostic.
    pub fn name_of(&self, def: DefId) -> String {
        self.names.get(&def).cloned().unwrap_or_else(|| format!("#{}", def.0))
    }

    /// How a type reads in a diagnostic.
    ///
    /// `Type`'s own `Display` can't do this: a resolved named type is just an
    /// identity, and turning that back into `std/string::String` needs the
    /// definition table. Every user-facing message that mentions a type goes
    /// through here.
    pub fn show(&self, ty: &Type<'a>) -> String {
        match ty {
            Type::Named { def, args } if args.is_empty() => self.name_of(*def),
            Type::Named { def, args } => {
                let args = args.iter().map(|a| self.show_arg(a)).collect::<Vec<_>>().join(", ");
                format!("{}<{}>", self.name_of(*def), args)
            }
            Type::Pointer(i) => format!("*{}", self.show(i)),
            Type::Slice(i) => format!("[{}]", self.show(i)),
            Type::Array(i, n) => format!("[{}; {}]", self.show(i), n),
            Type::Simd(i, n) => format!("simd[{}, {}]", self.show(i), n),
            Type::Function { params, return_type } => {
                let ps = params.iter().map(|p| self.show(p)).collect::<Vec<_>>().join(", ");
                format!("proc({}) {}", ps, self.show(return_type))
            }
            // primitives, type params, and the backend's symbol-named forms all
            // print themselves.
            other => other.to_string(),
        }
    }

    fn show_arg(&self, arg: &GenericArg<'a>) -> String {
        match arg {
            GenericArg::Type(t) => self.show(t),
            GenericArg::Const(c) => c.to_string(),
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn insert(&mut self, name: &'a str, binding: Option<Binding<'a>>, ty: Type<'a>) {
        self.scopes.last_mut().unwrap().insert(name, (binding, ty));
    }

    /// Walk scopes from innermost to outermost, returning the first match for `name`
    pub fn lookup(&self, name: &str) -> Option<&(Option<Binding<'a>>, Type<'a>)> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    /// Like [`lookup`], but also returns the matched entry's interned `&'a str`
    /// key. Used to recover a function's arena-lifetime name (e.g. to record a
    /// method call's target) from a lookup keyed by a temporary `String`.
    pub fn lookup_kv(&self, name: &str) -> Option<(&'a str, &(Option<Binding<'a>>, Type<'a>))> {
        self.scopes.iter().rev().find_map(|scope| scope.get_key_value(name).map(|(k, v)| (*k, v)))
    }
}
