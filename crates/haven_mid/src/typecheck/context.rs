use std::collections::HashMap;
use haven_common::ast::*;

/// Field index of the discriminant tag in a data-enum aggregate's synthetic
/// struct, and of the payload byte-blob. Referenced by name in the struct table
/// and by index in MIL lowering (`FieldPtr`).
pub const ENUM_TAG_FIELD: &str = "$tag";
pub const ENUM_PAYLOAD_FIELD: &str = "$payload";

/// The synthetic payload-struct name for a data variant, e.g. `Msg::Note` ->
/// `Msg$Note`. `$` can't appear in a source identifier, so this never collides
/// with a user type. Leaked to `'static` (coerces to any `'a`), as elsewhere in
/// the pipeline.
pub fn enum_payload_struct_name(enum_name: &str, variant: &str) -> &'static str {
    Box::leak(format!("{}${}", enum_name, variant).into_boxed_str())
}

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
    /// Name resolution: maps each `Var` use's node id to the specific param/local
    /// binding it refers to. Globals are absent (they fall back to the global
    /// namespace). Consumed by MIL lowering to key variable storage, which makes
    /// shadowing correct - two same-named locals get distinct `Binding::Local`s.
    pub resolved: HashMap<usize, Binding<'a>>,
    /// Struct definitions from name to ordered list of (field name, field type)
    pub structs: HashMap<&'a str, Vec<(&'a str, Type<'a>)>>,
    /// Structs declared with generic parameters (`struct Option<T>`,
    /// `struct Buf<T, const N: u32>`), mapped to their declared params in order
    /// (type and const interleaved). Field types are stored with `Type::Param`s
    /// and `ConstVal::Param`s; a construction/turbofish binds those to concrete
    /// args positionally (the `len()` gives the declared arity). Monomorphization
    /// rewrites a concrete use to a flat instance before codegen.
    pub generic_structs: std::collections::HashMap<&'a str, Vec<GenericParam<'a>>>,
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
    pub enums: HashMap<&'a str, EnumDef<'a>>,
    /// Enums declared with generic parameters (`enum Option<T>`), mapped to their
    /// declared params in order - mirrors `generic_structs`. A variant constructor
    /// or destructuring pattern binds these positionally via turbofish; monomorphization
    /// rewrites a concrete use to a flat instance before codegen.
    pub generic_enums: std::collections::HashMap<&'a str, Vec<GenericParam<'a>>>,
    /// Receiver method calls (`recv.method(...)`), keyed by the `Call` node id.
    /// Populated by `infer` and consumed by MIL lowering. Rebuilt on each typecheck
    /// pass, so it always matches the AST that lowering will see.
    pub method_calls: HashMap<usize, MethodCall<'a>>,
    /// Declared traits, by name. Populated in the forward-declaration pass;
    /// consumed by conformance checking and bounded method-call resolution.
    pub traits: HashMap<&'a str, TraitDef<'a>>,
    /// Which `(type, trait)` conformances hold, from `extend T: Trait` blocks
    /// (verified during the forward pass). A `T: Trait` bound at a generic call
    /// site is satisfied iff the concrete argument type is present here.
    pub impls: std::collections::HashSet<(&'a str, &'a str)>,
    /// Trait bounds on the type params of the function currently being checked,
    /// e.g. `{"T": ["Display"]}` inside `proc show<T: Display>(...)`. Lets a
    /// method call on a `T`-typed receiver resolve through the bound trait. Empty
    /// outside a bounded generic.
    pub generic_bounds: HashMap<&'a str, Vec<&'a str>>,
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
            resolved: HashMap::new(),
            structs: HashMap::new(),
            generic_structs: std::collections::HashMap::new(),
            generics: Vec::new(),
            const_generics: Vec::new(),
            generic_fns: HashMap::new(),
            global_consts: std::collections::HashSet::new(),
            enums: HashMap::new(),
            generic_enums: std::collections::HashMap::new(),
            method_calls: HashMap::new(),
            traits: HashMap::new(),
            impls: std::collections::HashSet::new(),
            generic_bounds: HashMap::new(),
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
