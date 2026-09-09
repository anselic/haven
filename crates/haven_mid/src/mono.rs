//! Rewrite generic functions and types into concrete instances.
//!
//! This runs after the first typecheck and before MIL. Each reachable set of
//! arguments gets a mangled instance, and call sites are rewritten to use it.
//! The result is typechecked again; MIL never sees generics.

use std::collections::{HashMap, VecDeque};

use bumpalo::Bump;

use haven_common::ast::*;
use haven_common::defs::{deinstance, Def, DefId, Defs, Instances, Linkage, Member,
                         MemberTable, TyHead};

/// The one method a `Delete` impl provides. Mirrors `own::DELETE_METHOD`, which
/// is where the lang item is actually interpreted; mono only needs to recognize
/// the name to know which member to instantiate eagerly.
const DELETE_METHOD: &str = "delete";

/// one requested instantiation: `base` specialized to `args`, emitted as
/// `mangled`. `span` is the call site that first asked for it (for errors).
struct Instantiation<'a> {
    base: DefId,
    args: Vec<ConcreteArg<'a>>,
    /// The identity minted for this instance, and the symbol it is emitted as.
    def: DefId,
    mangled: &'a str,
    span: Span,
}

/// one requested generic-struct instantiation: `struct Buf<T, const N>` at `args`
/// (`[i32, 8]`), emitted as the concrete `struct Buf$i32$8`. args are types and/or
/// const values, matching the struct's declared params. `span` points at the
/// context that first named the type, for the depth-limit diagnostic.
struct StructInstantiation<'a> {
    base: DefId,
    args: Vec<ConcreteArg<'a>>,
    /// The identity minted for this instance, and the symbol it is emitted as.
    def: DefId,
    mangled: &'a str,
    span: Span,
}

/// one requested generic-enum instantiation: `enum Option<T>` at `args` (`[i32]`),
/// emitted as the concrete `enum Option$i32`. Mirrors `StructInstantiation` - a
/// data-carrying enum is lowered as a struct-shaped aggregate, so once
/// monomorphized it goes through exactly the same downstream machinery.
struct EnumInstantiation<'a> {
    base: DefId,
    args: Vec<ConcreteArg<'a>>,
    /// The identity minted for this instance, and the symbol it is emitted as.
    def: DefId,
    mangled: &'a str,
    span: Span,
}

/// cap on distinct instantiations. backstop for combinatorial blowup (not
/// depth-growing); real programs are nowhere near this.
const INSTANTIATION_LIMIT: usize = 10_000;

/// cap on how deep a type argument can nest. polymorphic recursion like
/// `f<T>(...) { f::<*T>(...) }` grows the type forever (T, *T, **T, ...); this
/// catches it early, before the growing types make each instance expensive.
/// nothing hand-written nests this deep.
const TYPE_DEPTH_LIMIT: usize = 128;

/// how deep a type nests: i32 = 1, *i32 = 2, [*i32; 4] = 3, ...
fn type_depth(ty: &Type) -> usize {
    match ty {
        Type::Pointer(inner)
        | Type::Array(inner, _)
        | Type::Slice(inner)
        | Type::Simd(inner, _) => 1 + type_depth(inner),
        Type::Function { params, return_type } => {
            1 + params.iter().chain(std::iter::once(&**return_type))
                .map(type_depth).max().unwrap_or(0)
        }
        _ => 1,
    }
}

struct Mono<'p, 'a> {
    /// arena the mangled names live in. outlives the AST we produce, so the
    /// `&'a str` names it hands out are valid for `'a`.
    arena: &'a Bump,
    /// generic function templates, by emitted name - which is how a call site
    /// still names its callee. The identity comes along so an instance can be
    /// registered against it.
    templates: HashMap<&'a str, (DefId, &'p TopLevel<'a>)>,
    /// instantiations still to build.
    queue: VecDeque<Instantiation<'a>>,
    /// mangled name per instantiation we've already asked for, keyed by the
    /// mangled string, so repeated call sites reuse one instance (one alloc).
    seen: HashMap<InstanceKey<'a>, (DefId, &'a str)>,
    /// generic struct templates, by identity.
    struct_templates: HashMap<DefId, &'p TopLevel<'a>>,
    /// generic-struct instantiations still to build.
    struct_queue: VecDeque<StructInstantiation<'a>>,
    /// mangled name per struct instance already requested, keyed by the mangled
    /// string, so every concrete `Option$i32` use shares one emitted struct.
    struct_seen: HashMap<InstanceKey<'a>, (DefId, &'a str)>,
    /// generic enum templates, by identity.
    enum_templates: HashMap<DefId, &'p TopLevel<'a>>,
    /// generic-enum instantiations still to build.
    enum_queue: VecDeque<EnumInstantiation<'a>>,
    /// mangled name per enum instance already requested, keyed by the mangled
    /// string, so every concrete `Option$i32` use shares one emitted enum.
    enum_seen: HashMap<InstanceKey<'a>, (DefId, &'a str)>,
    /// best-effort span for the context currently being rebuilt, so a struct or
    /// enum instance requested deep inside `subst_ty` (which has no span of its
    /// own) still gets a source location for the depth-limit error.
    cur_span: Span,
    /// definition arena, extended as we go: every instance minted below is
    /// registered here with the template it came from, so post-mono passes can
    /// map an instance back to its template without taking its name apart.
    defs: &'p mut Defs<'a>,
    /// Every method in the program, snapshotted from `defs` before minting
    /// starts. A copy rather than a borrow because `defs` is also written to
    /// here; nothing added during mono needs to be *dispatched* through, so the
    /// snapshot cannot go stale in a way that matters.
    members: MemberTable<'a>,
    /// What the pre-monomorphization typecheck inferred for every expression.
    ///
    /// This is the one piece of type information mono has, and it exists for a
    /// single job: a call `xs.show()` names no callee that a syntactic pass could
    /// resolve, so instantiating `extend [T]`'s methods needs to know what `xs`
    /// is. Keyed by the *template* AST's node ids, which is what `rebuild_expr`
    /// reads from, and the types are in terms of the template's own parameters -
    /// so `subst_ty` with the current bindings turns one into the concrete
    /// receiver at this instantiation.
    node_types: &'p HashMap<usize, Type<'a>>,
    /// Turbofish args the typechecker inferred for bare generic calls
    /// (`printf(x)`), keyed by the call node's id. The AST node carries an empty
    /// `type_args`, so `rebuild_expr` falls back to this to learn which instance
    /// the call asks for. Keyed by template node ids, same as `node_types`.
    inferred_type_args: &'p HashMap<usize, Vec<GenericArg<'a>>>,
    /// Every `extend T: Trait` in the program, for deciding whether a
    /// *conditional* impl covers the instance being minted — see
    /// [`Self::impl_applies`]. Only destructors consult this: they are the one
    /// thing mono creates without a call site asking for it.
    impls: &'p [ImplDecl<'a>],
}

/// A bound const generic parameter: its concrete value plus declared type, so a
/// value-position use (`return N;`) can be re-emitted as a typed literal.
#[derive(Clone)]
struct ConstBind<'a> {
    val: usize,
    ty: Type<'a>,
}

/// Substitutions applied when specializing a template: type params -> concrete
/// types, const params -> concrete values.
struct Bindings<'a> {
    types: HashMap<&'a str, Type<'a>>,
    consts: HashMap<&'a str, ConstBind<'a>>,
}

impl<'a> Bindings<'a> {
    fn empty() -> Self {
        Bindings { types: HashMap::new(), consts: HashMap::new() }
    }
}

/// A fully concrete generic argument at an instantiation site: the value a type
/// or const param is specialized to. Keys the instance cache and mangled name.
#[derive(Clone, PartialEq, Eq, Hash)]
enum ConcreteArg<'a> {
    Type(Type<'a>),
    Const(usize),
}

/// What identifies one instantiation: the template plus the concrete arguments
/// it was specialized with. The instance caches key on *this* rather than on the
/// mangled string, so `mangle_ty` no longer has to be injective for the compiler
/// to be correct - see its doc comment.
type InstanceKey<'a> = (DefId, Vec<ConcreteArg<'a>>);

/// Substitute a bound const param with its literal value; leave literals and
/// unbound params untouched.
fn subst_cv<'a>(cv: &ConstVal<'a>, b: &Bindings<'a>) -> ConstVal<'a> {
    match cv {
        ConstVal::Param(n) => match b.consts.get(n) {
            Some(cb) => ConstVal::Lit(cb.val),
            None => cv.clone(),
        },
        ConstVal::Lit(_) => cv.clone(),
    }
}

/// Materialize a bound const param used in value position as a typed integer
/// literal (`const N: u32` bound to 4 -> `4u32`).
fn const_literal<'a>(cb: &ConstBind<'a>) -> ExprNode<'a> {
    match &cb.ty {
        Type::Int8   => ExprNode::Int8(cb.val as i8),
        Type::Int16  => ExprNode::Int16(cb.val as i16),
        Type::Int32  => ExprNode::Int32(cb.val as i32),
        Type::Int64  => ExprNode::Int64(cb.val as i64),
        Type::Uint8  => ExprNode::Uint8(cb.val as u8),
        Type::Uint16 => ExprNode::Uint16(cb.val as u16),
        Type::Uint32 => ExprNode::Uint32(cb.val as u32),
        Type::Uint64 => ExprNode::Uint64(cb.val as u64),
        other => panic!("const generic parameter has non-integer type {}", other),
    }
}

/// encode a concrete type into an identifier-safe fragment for a mangled name.
/// `$`, `.`, `_` and alphanumerics are all fine in unquoted LLVM identifiers.
///
/// Injectivity: every type *constructor* (pointer/array/slice/simd/param/fn/
/// generic-struct) is encoded with a leading `.tag`. A user struct name is a bare
/// identifier and can never contain `.`, and the scalar keywords are a fixed set
/// with no `.` - so a constructor fragment can never collide with a struct name
/// or a scalar. Struct arguments are wrapped in balanced `.lt`/`.gt` so their
/// boundaries stay unambiguous under nesting.
///
/// This is no longer load-bearing for *correctness*: `request*` keys its instance
/// caches on `InstanceKey` (the structural args), not on this string. A collision
/// would now mean two distinct instances emitted under one symbol - a duplicate
/// symbol at link time, which is loud - rather than two instances silently
/// sharing one instantiation.
fn mangle_ty<'a>(defs: &Defs<'a>, arena: &'a Bump, ty: &Type<'a>) -> String {
    let mangle_ty = |t: &Type<'a>| mangle_ty(defs, arena, t);
    match ty {
        Type::Void => "void".into(),
        Type::Bool => "bool".into(),
        Type::Int8 => "i8".into(),
        Type::Int16 => "i16".into(),
        Type::Int32 => "i32".into(),
        Type::Int64 => "i64".into(),
        Type::Uint8 => "u8".into(),
        Type::Uint16 => "u16".into(),
        Type::Uint32 => "u32".into(),
        Type::Uint64 => "u64".into(),
        Type::Float32 => "f32".into(),
        Type::Float64 => "f64".into(),
        Type::Str => "str".into(),
        Type::Path { path, .. } => Type::unresolved(path),
        Type::Pointer(inner) => format!(".ptr{}", mangle_ty(inner)),
        Type::Array(inner, n) => format!(".arr{}.{}", n.expect_lit(), mangle_ty(inner)),
        Type::Slice(inner) => format!(".slice{}", mangle_ty(inner)),
        Type::Simd(inner, n) => format!(".simd{}.{}", n.expect_lit(), mangle_ty(inner)),
        // a named type mangles as its emitted symbol, which is already unique
        // across the program.
        Type::Named { def, args } if args.is_empty() => defs.symbol(*def, arena).into(),
        // a generic instance is normally flattened to a bare identity before it
        // reaches here (subst_ty requests it); encode defensively with balanced
        // brackets so nested args stay unambiguous.
        Type::Named { def, args } => {
            let inner = args.iter()
                .map(|a| mangle_generic_arg(defs, arena, a))
                .collect::<Vec<_>>().join(".");
            format!(".struct.{}.{}", defs.symbol(*def, arena), inner)
        }
        // Neither should appear in a fully-concrete instantiation; encode them
        // defensively rather than panicking so a bug surfaces as a bad symbol.
        Type::Param(name) => format!(".param.{}", name),
        // `!` is only ever an expression's inferred type, never part of a
        // monomorphized signature - encode defensively rather than panic.
        Type::Never => ".never".into(),
        Type::Function { params, return_type } => {
            let ps = params.iter().map(mangle_ty).collect::<Vec<_>>().join(".");
            format!(".fn{}.{}.ret{}", params.len(), ps, mangle_ty(return_type))
        }
    }
}

/// Mangle a generic argument in a struct type's arg list (a type or a const
/// value). Only reached on the defensive not-yet-flattened path in `mangle_ty`.
fn mangle_generic_arg<'a>(defs: &Defs<'a>, arena: &'a Bump, ga: &GenericArg<'a>) -> String {
    match ga {
        GenericArg::Type(t) => mangle_ty(defs, arena, t),
        GenericArg::Const(cv) => cv.to_string(),
    }
}

/// Identifier-safe fragment for one generic arg: a mangled type, or a const's
/// decimal value (`4`). Const values are pure digits and no type mangles to bare
/// digits, so the two never collide within a single arg list.
fn mangle_arg<'a>(defs: &Defs<'a>, arena: &'a Bump, arg: &ConcreteArg<'a>) -> String {
    match arg {
        ConcreteArg::Type(t) => mangle_ty(defs, arena, t),
        ConcreteArg::Const(n) => n.to_string(),
    }
}

/// `id`, `[i32]` -> `id$slice_i32`; `splat`, `f32`, `4` -> `splat$f32$4`.
fn mangle_name<'a>(defs: &Defs<'a>, arena: &'a Bump, base: &str, args: &[ConcreteArg<'a>]) -> String {
    let parts = args.iter().map(|a| mangle_arg(defs, arena, a)).collect::<Vec<_>>().join("$");
    format!("{}${}", base, parts)
}

/// Human-readable spelling of an instance, for diagnostics only: `alloc`,
/// `[Vec2]` -> `alloc::<Vec2>`.
///
/// The template's *source* name is used, not its symbol, so the module slug
/// never appears - it is a linkage detail, not something the user wrote. That
/// used to be recovered by stripping everything before the last `$`, which only
/// worked because the slug happened to contain no `$` of its own.
fn display_name<'a>(defs: &Defs<'a>, base: DefId, args: &[ConcreteArg<'a>]) -> String {
    let targs = args.iter().map(|a| match a {
        ConcreteArg::Type(Type::Named { def, .. }) => defs.show(*def),
        ConcreteArg::Type(t) => t.to_string(),
        ConcreteArg::Const(n) => n.to_string(),
    }).collect::<Vec<_>>().join(", ");
    format!("{}::<{}>", defs.get(base).source_name, targs)
}

/// The method `name` a receiver of type `ty` dispatches to, plus the bindings
/// that specialize the impl to it.
///
/// The same two-step as `Context::member_for`/`receiver_member`, duplicated here
/// rather than shared because mono holds a plain [`MemberTable`] and no
/// typecheck context. Both must agree on which impl a receiver picks — mono
/// mints the instance the typechecker will later expect to exist.
/// Returns the resolved member, its bindings, and whether it was reached through
/// the one-level pointer fallback (`via_deref`) rather than on the receiver's own
/// head. The receiver adjustment depends on the distinction: a `*self` method
/// found directly on a `*T` receiver (an `extend *T` blanket) has `self: **T`, so
/// the receiver must be address-taken, whereas the same method found by derefing
/// (an `extend T` reached through a `*T`) has `self: *T` and passes as-is.
fn member_of<'a>(members: &MemberTable<'a>, instances: &Instances<'a>, ty: &Type<'a>, name: &str)
    -> Option<(Member<'a>, Unified<'a>, bool)>
{
    fn direct<'a>(members: &MemberTable<'a>, instances: &Instances<'a>, ty: &Type<'a>, name: &str)
        -> Option<(Member<'a>, Unified<'a>)>
    {
        // both the head and the unification want template form: the head
        // because `Vec$i32` was never registered against anything, and the
        // unification because that is the form the impl was written in.
        let ty = deinstance(instances, ty);
        let m = members.get(&(TyHead::of(&ty)?, name))?;
        let params: Vec<&'a str> = m.generics.iter().map(|g| match g {
            GenericParam::Type { name, .. } => *name,
            GenericParam::Const(name, _) => *name,
        }).collect();
        let mut u = Unified::default();
        unify(&m.self_ty, &ty, &params, &mut u).then(|| (m.clone(), u))
    }
    match (direct(members, instances, ty, name), ty) {
        (None, Type::Pointer(inner)) =>
            direct(members, instances, inner, name).map(|(m, u)| (m, u, true)),
        (found, _) => found.map(|(m, u)| (m, u, false)),
    }
}

/// The concrete function a method call `recv.field(...)` dispatches to, for a
/// receiver of type `ty`. Returns `None` when the call is not a resolvable
/// concrete method: an associated function (spelled as a path, never reached
/// here), a generic-`extend` template (which mono has already rewritten into a
/// direct call on the instance, so its `Access` form no longer survives), or a
/// receiver whose method can't be found. Callers that can't resolve a method
/// this way should treat the call conservatively.
///
/// Used by the `@alloc(false)` check, which runs on the post-mono AST where a
/// concrete method call is still `Call { func: Access { .. } }` (only generic
/// ones were lowered to named calls). Resolving it to `m.name` - the top-level
/// function `lower_methods` desugared the method into - turns the call into a
/// real call-graph edge instead of an opaque indirect call.
pub(crate) fn concrete_method_name<'a>(
    members: &MemberTable<'a>,
    instances: &Instances<'a>,
    ty: &Type<'a>,
    field: &'a str,
) -> Option<&'a str> {
    let (m, _, _) = member_of(members, instances, ty, field)?;
    // an associated fn has no receiver to dispatch on, and a template's name is
    // a base that has no clean-status of its own; neither is a concrete edge.
    if m.receiver == Receiver::Associated || !m.generics.is_empty() {
        return None;
    }
    Some(m.name)
}

impl<'p, 'a> Mono<'p, 'a> {
    /// Record an instantiation request, return its (stable) mangled name.
    /// de-dupes so each distinct instance is built exactly once.
    fn request(&mut self, base_name: &'a str, args: Vec<ConcreteArg<'a>>, span: Span) -> &'a str {
        let base = self.templates[base_name].0;
        let key = (base, args.clone());
        if let Some(&(_, m)) = self.seen.get(&key) {
            return m;
        }
        let (def, mangled) = self.mint(base, &args);
        self.seen.insert(key, (def, mangled));
        self.queue.push_back(Instantiation { base, args, def, mangled, span });
        mangled
    }

    /// The template `base` names, for the instantiation loops.
    fn template_of(&self, base: DefId) -> &'p TopLevel<'a> {
        self.templates.values().find(|(d, _)| *d == base).expect("function template").1
    }

    /// Mint the identity for one instance of `base`: a fresh `DefId` emitted
    /// under the mangled symbol, recorded against the template it specializes.
    ///
    /// The instance is `Fixed`, not `Mangled`: its symbol is derived from the
    /// template's already-final symbol plus the arguments, not from a source name
    /// plus a module slug.
    fn mint(&mut self, base: DefId, args: &[ConcreteArg<'a>]) -> (DefId, &'a str) {
        let base_sym = self.defs.symbol(base, self.arena);
        let mangled: &'a str =
            self.arena.alloc_str(&mangle_name(self.defs, self.arena, base_sym, args));
        let display = display_name(self.defs, base, args);
        let t = self.defs.get(base);
        let (module, kind, source_name, is_pub, span) =
            (t.module, t.kind, t.source_name, t.is_pub, t.span.clone());
        let def = self.defs.alloc(Def {
            module, kind, source_name, is_pub, linkage: Linkage::Fixed(mangled), span,
        });
        self.defs.add_instance(def, base, display, args.iter().map(|a| match a {
            ConcreteArg::Type(t) => GenericArg::Type(t.clone()),
            ConcreteArg::Const(n) => GenericArg::Const(ConstVal::Lit(*n)),
        }).collect());
        (def, mangled)
    }

    /// Put a monomorphized instance type back into template form, so head
    /// dispatch can find the impl it was declared against. See
    /// [`haven_common::defs::deinstance`], which every post-mono lookup shares:
    /// mono needs it because a generic function's type parameter can be bound to
    /// a generic *instance* (`foo::<Vec<i32>>` reaching `x.display()` on `x: *T`),
    /// and the passes after it because by then every generic type is an instance.
    fn deinstance(&self, ty: &Type<'a>) -> Type<'a> {
        haven_common::defs::deinstance(self.defs.instances(), ty)
    }

    /// Record a generic-struct instantiation request, return its mangled name
    /// (`Buf` + `[i32, 8]` -> `Buf$i32$8`). de-dupes so each concrete instance is
    /// built once. Enqueues onto the struct queue, drained after all functions.
    fn request_struct(&mut self, base: DefId, args: Vec<ConcreteArg<'a>>) -> DefId {
        let key = (base, args.clone());
        if let Some(&(d, _)) = self.struct_seen.get(&key) {
            return d;
        }
        let (def, mangled) = self.mint(base, &args);
        self.struct_seen.insert(key, (def, mangled));
        self.instantiate_destructor(base, def, &args);
        self.struct_queue.push_back(StructInstantiation {
            base, args, def, mangled, span: self.cur_span.clone(),
        });
        def
    }

    /// Whether the `extend` block `m` came from actually covers the instance
    /// `base<args>` — that is, whether its `where` clause holds there.
    ///
    /// Unifies rather than zipping `m.generics` against `args` positionally,
    /// because the two need not line up: `extend Vec<Box<T>>`'s single parameter
    /// is `T`, while the instance's single argument is `Box<i32>`.
    fn impl_applies(&self, m: &Member<'a>, base: DefId, args: &[ConcreteArg<'a>]) -> bool {
        if !m.generics.iter().any(|g| matches!(g,
            GenericParam::Type { bounds, .. } if !bounds.is_empty())) {
            return true;
        }
        // the arguments are in instance form (`Vec$i32`), and the `where` clause
        // this is about to check asks whether one of them implements a trait -
        // a question the conformance list answers in template form only. Without
        // the round trip a conditional impl silently never applies as soon as its
        // argument is itself generic, and the instance goes undestroyed.
        let concrete = Type::Named {
            def: base,
            args: args.iter().map(|a| match a {
                ConcreteArg::Type(t) => GenericArg::Type(self.deinstance(t)),
                ConcreteArg::Const(n) => GenericArg::Const(ConstVal::Lit(*n)),
            }).collect(),
        };
        let params: Vec<&'a str> = m.generics.iter().map(|g| match g {
            GenericParam::Type { name, .. } => *name,
            GenericParam::Const(name, _) => *name,
        }).collect();
        let mut u = Unified::default();
        // no `scope`: this runs after substitution, so every argument is a real
        // type and there is no parameter left whose bound could answer for it.
        unify(&m.self_ty, &concrete, &params, &mut u)
            && bounds_hold(self.impls, &ParamBounds::new(), &m.generics, &u)
    }

    /// Mint the destructor for a freshly created generic-type instance.
    ///
    /// Every other method of a generic `extend` is instantiated on demand, from
    /// the call that needs it. A destructor has no such call: the ownership pass
    /// *synthesizes* the calls to it, and runs after monomorphization has
    /// finished, so by the time anything wants `Vec$delete$i32` there is nobody
    /// left to mint it. Creating the type is therefore what creates its
    /// destructor, and the instance's `delete` is registered as a member of the
    /// instance so the ownership pass can find it by the concrete type in hand.
    ///
    /// Requesting it here rather than lazily costs one specialized function per
    /// owning instance, which is exactly the set that can need destroying.
    fn instantiate_destructor(&mut self, base: DefId, inst: DefId, args: &[ConcreteArg<'a>]) {
        let Some(m) = self.members.get(&(TyHead::Def(base), DELETE_METHOD)) else { return };
        // a non-generic `delete` on a generic type cannot happen (the impl's
        // parameters are the type's), but a template that never made it into the
        // function table can - a conformance error, already reported.
        if m.generics.is_empty() || !self.templates.contains_key(m.name) { return; }
        // a conditional impl - `extend Vec<T>: Delete where T: Delete` - does not
        // cover every instance. `Vec<u8>` gets no destructor, and so stays `Copy`;
        // minting one anyway would specialize a body that calls `u8::delete`.
        if !self.impl_applies(m, base, args) { return; }
        let (name, receiver, is_pub, module) = (m.name, m.receiver, m.is_pub, m.module);
        let mangled = self.request(name, args.to_vec(), self.cur_span.clone());
        self.defs.add_member(TyHead::Def(inst), DELETE_METHOD, Member {
            name: mangled,
            receiver,
            self_ty: Type::named(inst),
            generics: Vec::new(),
            // inherited from the template's `Delete` impl (a trait method, so
            // always public); the instance's destructor is as reachable as it.
            is_pub,
            module,
        });
    }

    /// Record a generic-enum instantiation request, return its mangled name
    /// (`Option` + `[i32]` -> `Option$i32`). de-dupes so each concrete instance is
    /// built once. Enqueues onto the enum queue, drained (interleaved with the
    /// struct queue, since either can request the other) after all functions.
    fn request_enum(&mut self, base: DefId, args: Vec<ConcreteArg<'a>>) -> DefId {
        let key = (base, args.clone());
        if let Some(&(d, _)) = self.enum_seen.get(&key) {
            return d;
        }
        let (def, mangled) = self.mint(base, &args);
        self.enum_seen.insert(key, (def, mangled));
        self.instantiate_destructor(base, def, &args);
        // a data variant's payload struct is an identity of its own, and the
        // instance needs its own set - the template's payloads are typed in terms
        // of the template's parameters.
        let TopLevelNode::Enum { variants, .. } = &self.enum_templates[&base].value
        else { unreachable!("enum template is not an enum") };
        let data_variants: Vec<&'a str> = variants.iter()
            .filter(|(_, _, payload)| !payload.is_empty())
            .map(|(v, _, _)| *v).collect();
        for v in data_variants {
            let sym = self.arena.alloc_str(&format!("{}${}", mangled, v));
            self.defs.add_payload(def, v, sym);
        }
        self.enum_queue.push_back(EnumInstantiation {
            base, args, def, mangled, span: self.cur_span.clone(),
        });
        def
    }

    /// Substitute bound type and const params in `ty` with their concrete
    /// bindings.
    ///
    /// A concrete generic type (`Option<i32>`, args non-empty) is monomorphized
    /// here: its args are substituted, the instance is requested, and the flat
    /// instance identity (no args) is returned. Every downstream pass therefore
    /// only ever sees ordinary no-arg named types.
    ///
    /// A type parameter is a `Type::Param` and a real type is a `Type::Named`,
    /// so the two can no longer be confused. This used to carry a FIXME: both
    /// arrived as `Type::Struct { name }`, and a real `struct T` used inside an
    /// `f<T>` was substituted as if it were the parameter. A `DefId` is not a
    /// name a parameter can collide with, so that is now impossible to write.
    fn subst_ty(&mut self, ty: &Type<'a>, b: &Bindings<'a>) -> Type<'a> {
        match ty {
            Type::Param(n) if b.types.contains_key(n) => b.types[n].clone(),
            // a concrete generic instance: substitute inside its args (types and
            // const values), then request it, collapsing to the flat instance.
            // Whether it is a struct or an enum is decided by template-table
            // membership; either way the result is a no-arg `Named`.
            Type::Named { def, args } if !args.is_empty() => {
                let cargs: Vec<ConcreteArg<'a>> = args.iter()
                    .map(|a| self.concrete_targ(a, b)).collect();
                let inst = if self.enum_templates.contains_key(def) {
                    self.request_enum(*def, cargs)
                } else {
                    self.request_struct(*def, cargs)
                };
                Type::named(inst)
            }
            Type::Pointer(inner)  => Type::Pointer(Box::new(self.subst_ty(inner, b))),
            Type::Array(inner, n) => Type::Array(Box::new(self.subst_ty(inner, b)), subst_cv(n, b)),
            Type::Slice(inner)    => Type::Slice(Box::new(self.subst_ty(inner, b))),
            Type::Simd(inner, n)  => Type::Simd(Box::new(self.subst_ty(inner, b)), subst_cv(n, b)),
            Type::Function { params, return_type } => Type::Function {
                params: params.iter().map(|p| self.subst_ty(p, b)).collect(),
                return_type: Box::new(self.subst_ty(return_type, b)),
            },
            other => other.clone(),
        }
    }

    /// Substitute bound params in `ty` *without* collapsing generic types to
    /// their instances.
    ///
    /// [`Self::subst_ty`] does both jobs at once, which is right everywhere a
    /// type is being rewritten for emission - but wrong for deciding which
    /// `extend` block a receiver dispatches to. Dispatch happens on the head, and
    /// collapsing turns `Buf<i32>` into the fresh instance `Buf$i32`, whose head
    /// is an identity no impl was ever registered against. The generic form is
    /// what has to be matched against the impl's `Buf<T>`, so this keeps it.
    fn subst_params(&self, ty: &Type<'a>, b: &Bindings<'a>) -> Type<'a> {
        match ty {
            Type::Param(n) if b.types.contains_key(n) => b.types[n].clone(),
            Type::Named { def, args } => Type::Named {
                def: *def,
                args: args.iter().map(|a| match a {
                    GenericArg::Type(t) => GenericArg::Type(self.subst_params(t, b)),
                    GenericArg::Const(cv) => GenericArg::Const(subst_cv(cv, b)),
                }).collect(),
            },
            Type::Pointer(inner)  => Type::Pointer(Box::new(self.subst_params(inner, b))),
            Type::Array(inner, n) => Type::Array(Box::new(self.subst_params(inner, b)), subst_cv(n, b)),
            Type::Slice(inner)    => Type::Slice(Box::new(self.subst_params(inner, b))),
            Type::Simd(inner, n)  => Type::Simd(Box::new(self.subst_params(inner, b)), subst_cv(n, b)),
            Type::Function { params, return_type } => Type::Function {
                params: params.iter().map(|p| self.subst_params(p, b)).collect(),
                return_type: Box::new(self.subst_params(return_type, b)),
            },
            other => other.clone(),
        }
    }

    /// Substitute bound params in a turbofish argument. A const generic forwarded
    /// by name (`simd_load::<f32, N>`) reaches here as a bare-ident `Type` - but
    /// the name binds in `consts`, not `types` - so resolve it to a literal
    /// `Const` argument; the specialized call then re-typechecks against a
    /// concrete value.
    fn subst_targ(&mut self, ga: &GenericArg<'a>, b: &Bindings<'a>) -> GenericArg<'a> {
        match ga {
            GenericArg::Type(Type::Param(n)) if b.consts.contains_key(n) =>
                GenericArg::Const(ConstVal::Lit(b.consts[n].val)),
            GenericArg::Type(t) => GenericArg::Type(self.subst_ty(t, b)),
            GenericArg::Const(cv) => GenericArg::Const(subst_cv(cv, b)),
        }
    }

    /// [`Self::subst_targ`] followed by the collapse to a [`ConcreteArg`], for
    /// the generic *type* args of a `Named` (a struct/enum instance) rather than
    /// a call's turbofish. Both paths need the same fixup: a bare-ident const
    /// argument (`Ring<T, N>`, `C::<N> { .. }`) parses as a type, because nothing
    /// in the grammar separates a type name from a const parameter's name, so a
    /// forwarded const param arrives here as `GenericArg::Type(Type::Param(n))`.
    /// Only the enclosing instance's bindings can tell the two apart, which is
    /// why the correction lands in mono and not in the parser.
    ///
    /// Going through `subst_targ` rather than matching inline is what keeps the
    /// two in step: routing a `Named`'s args around it is how `C::<N> { .. }`
    /// used to reach the kind-mismatch `unreachable!` in the instance builder
    /// while the identical `f::<N>()` went through fine.
    fn concrete_targ(&mut self, ga: &GenericArg<'a>, b: &Bindings<'a>) -> ConcreteArg<'a> {
        match self.subst_targ(ga, b) {
            GenericArg::Type(t) => ConcreteArg::Type(t),
            GenericArg::Const(cv) => ConcreteArg::Const(cv.expect_lit()),
        }
    }

    /// Specialize a receiver call `recv.m(args)` whose method came from a
    /// *generic* `extend` block, rewriting it into a direct call on the
    /// instance: `xs.show()` with `xs: [i32]` becomes `slice$show$i32(xs)`.
    ///
    /// `None` leaves the call alone, which is the right answer for every method
    /// of a concrete `extend` (`Point`, `i32`): those need no instance, and the
    /// post-mono typecheck resolves them through the member table exactly as
    /// before. Only a generic impl has to be rewritten here, because its
    /// instance's name exists nowhere until this function mints it.
    ///
    /// A `*self` method on a receiver that isn't a place produces `&<temporary>`,
    /// which the post-mono typecheck rejects with its usual "bind it to a `let`
    /// first" message. That is a real limitation of desugaring to a call rather
    /// than carrying the adjustment through to lowering, but the diagnostic
    /// lands on the offending call and says what to do.
    fn generic_method_call(
        &mut self,
        base: &Expr<'a>,
        field: &'a str,
        type_args: &[GenericArg<'a>],
        new_args: &[Expr<'a>],
        b: &Bindings<'a>,
        span: &Span,
    ) -> Option<ExprNode<'a>> {
        // what the receiver is *here*: its type in the template, with this
        // instantiation's parameters bound but generic types left generic, since
        // that is the form an impl is written against.
        let recv_ty = self.node_types.get(&base.id)?.clone();
        // `subst_params` may bind a parameter to a generic *instance* (`Vec$i32`)
        // rather than to `Vec<i32>`, when the caller's turbofish was itself a
        // generic type; `deinstance` restores template form so head dispatch can
        // find the impl and unification can recover its parameters.
        let recv_ty = self.deinstance(&self.subst_params(&recv_ty, b));

        // dispatch by head, then through one pointer level - the same two-step
        // the typechecker uses, so both agree on which impl a receiver picks.
        let (m, u, via_deref) = member_of(&self.members, self.defs.instances(), &recv_ty, field)?;
        if m.receiver == Receiver::Associated { return None; }
        // not a template: a method of a concrete `extend` that declares no
        // generics of its own needs no instance, and the post-mono typecheck
        // resolves it through the member table exactly as before.
        if !self.templates.contains_key(m.name) { return None; }

        // the instance's arguments, in the order the desugared function declares
        // its parameters: the impl's own list first, bound by unifying against
        // the receiver, then whatever the method declares for itself, which only
        // the turbofish can supply. The typechecker has already checked that the
        // two together account for every parameter.
        //
        // Each impl argument is put through `subst_ty` so a *nested* generic one
        // collapses to its instance (`Buf<Vec<i32>>` binds `T = Vec$i32`, not
        // `Vec<i32>`), matching how every other request spells its arguments -
        // two spellings of one type would otherwise mint the instance twice.
        let mut cargs: Vec<ConcreteArg<'a>> = Vec::with_capacity(m.generics.len());
        for g in &m.generics {
            cargs.push(match g {
                GenericParam::Type { name, .. } => {
                    let bound = u.types.get(name)?.clone();
                    ConcreteArg::Type(self.subst_ty(&bound, &Bindings::empty()))
                }
                GenericParam::Const(name, _) => ConcreteArg::Const(u.consts.get(name)?.expect_lit()),
            });
        }
        for ta in type_args {
            cargs.push(match self.subst_targ(ta, b) {
                GenericArg::Type(t) => ConcreteArg::Type(t),
                GenericArg::Const(cv) => ConcreteArg::Const(cv.expect_lit()),
            });
        }
        let mangled = self.request(m.name, cargs, span.clone());

        // form the `self` argument. A `*self` method found directly on the
        // receiver's head takes `self: *recv_ty`, so the base is address-taken -
        // whether that base is a value (`extend i32` on an `i32`) or is itself a
        // pointer (`extend *T` on a `*i32`, whose `self` is `**i32`). Reached
        // through the pointer fallback instead (`extend i32` on a `*i32`), the
        // base already *is* the `*self`, so it passes straight through.
        let recv = self.rebuild_expr(base, b);
        let recv = if m.receiver == Receiver::Pointer && !via_deref {
            Metadata::new(
                ExprNode::Unary { op: UnaryOp::AddrOf, operand: Box::new(recv) },
                base.span.clone(),
            )
        } else {
            recv
        };

        let mut args = Vec::with_capacity(new_args.len() + 1);
        args.push(recv);
        args.extend(new_args.iter().cloned());
        Some(ExprNode::Call {
            func: Box::new(Metadata::new(ExprNode::Var(mangled), base.span.clone())),
            type_args: Vec::new(),
            args,
        })
    }

    fn rebuild_expr(&mut self, expr: &Expr<'a>, b: &Bindings<'a>) -> Expr<'a> {
        let node = match &expr.value {
            ExprNode::Call { func, type_args, args } => {
                let new_args: Vec<Expr<'a>> =
                    args.iter().map(|a| self.rebuild_expr(a, b)).collect();

                // a method whose desugared function is a template needs its
                // instance minted and the call pointed at it; everything else
                // falls through to the ordinary paths below. A turbofish here
                // belongs to the method's own parameters, so it goes with it
                // rather than staying on the rewritten call.
                if let ExprNode::Access { base, field } = &func.value {
                    if let Some(call) =
                        self.generic_method_call(base, field, type_args, &new_args, b, &expr.span)
                    {
                        return Metadata::new(call, expr.span.clone());
                    }
                }
                // sub type params inside the turbofish (user generic calls +
                // intrinsics like `sizeof::<T>()`). A bare generic call carries an
                // empty turbofish on the node; the typechecker recovered its args
                // by inference, so fall back to those.
                let effective_targs: &[GenericArg<'a>] = if type_args.is_empty() {
                    self.inferred_type_args.get(&expr.id).map_or(type_args, Vec::as_slice)
                } else {
                    type_args
                };
                let subst_targs: Vec<GenericArg<'a>> =
                    effective_targs.iter().map(|ga| self.subst_targ(ga, b)).collect();

                // user generic call: mangle to the concrete instance, drop the
                // turbofish.
                if let ExprNode::Var(name) = &func.value {
                    if self.templates.contains_key(name) {
                        let concrete: Vec<ConcreteArg<'a>> = subst_targs.iter().map(|ga| match ga {
                            GenericArg::Type(t) => ConcreteArg::Type(t.clone()),
                            // subst_targ resolved every const param to a literal.
                            GenericArg::Const(cv) => ConcreteArg::Const(cv.expect_lit()),
                        }).collect();
                        let mangled = self.request(name, concrete, expr.span.clone());
                        let new_func = Metadata::new(ExprNode::Var(mangled), func.span.clone());
                        ExprNode::Call {
                            func: Box::new(new_func),
                            type_args: Vec::new(),
                            args: new_args,
                        }
                    } else {
                        // an intrinsic or an ordinary call: keep the (substituted)
                        // turbofish.
                        let new_func = self.rebuild_expr(func, b);
                        ExprNode::Call { func: Box::new(new_func), type_args: subst_targs, args: new_args }
                    }
                } else if let ExprNode::Path(path) = &func.value {
                    if path.path.segments.len() == 2 && b.types.contains_key(path.path.segments[0]) {
                        // an associated call through a type param, `P::new(args)`:
                        // substitute `P` to the concrete type and re-mangle to that
                        // type's associated function (`Gain$new`). The typechecker
                        // already validated it dispatches through `P`'s bound.
                        let subst = self.subst_ty(&Type::Param(path.path.segments[0]), b);
                        let concrete = self.deinstance(&subst);
                        let (m, u, _) = member_of(&self.members, self.defs.instances(),
                                                  &concrete, path.path.segments[1])
                            .expect("bounded associated fn resolves after substitution");
                        // an associated fn on a *generic* impl is itself a template
                        // (its instance carries the impl's bound params); a concrete
                        // impl's is already a final symbol.
                        let name = if self.templates.contains_key(m.name) {
                            let cargs: Vec<ConcreteArg<'a>> = m.generics.iter().map(|g| match g {
                                GenericParam::Type { name, .. } => ConcreteArg::Type(
                                    self.subst_ty(u.types.get(name)
                                        .expect("impl param bound by unification"), &Bindings::empty())),
                                GenericParam::Const(name, _) => ConcreteArg::Const(
                                    u.consts.get(name).expect("impl const bound").expect_lit()),
                            }).collect();
                            self.request(m.name, cargs, expr.span.clone())
                        } else {
                            m.name
                        };
                        ExprNode::Call {
                            func: Box::new(Metadata::new(ExprNode::Var(name), func.span.clone())),
                            type_args: Vec::new(),
                            args: new_args,
                        }
                    } else if self.enum_templates.contains_key(&path.def) {
                        // a generic-enum tuple/unit variant constructor with
                        // turbofish (`Option::Some::<i32>(5)`, `Option::None::<i32>()`):
                        // mangle to the concrete instance and rewrite the path's enum
                        // segment to it (`Option$i32::Some`), dropping the turbofish.
                        // Unlike a plain function call, the name to request against is
                        // the enum segment, not the whole path.
                        let concrete: Vec<ConcreteArg<'a>> = subst_targs.iter().map(|ga| match ga {
                            GenericArg::Type(t) => ConcreteArg::Type(t.clone()),
                            GenericArg::Const(cv) => ConcreteArg::Const(cv.expect_lit()),
                        }).collect();
                        let inst = self.request_enum(path.def, concrete);
                        let mut new_path = path.clone();
                        new_path.def = inst;
                        let new_func = Metadata::new(
                            ExprNode::Path(new_path), func.span.clone());
                        ExprNode::Call {
                            func: Box::new(new_func),
                            type_args: Vec::new(),
                            args: new_args,
                        }
                    } else {
                        // a non-generic enum-variant constructor: its turbofish is
                        // always empty (Phase 1/2 behavior, unchanged).
                        let new_func = self.rebuild_expr(func, b);
                        ExprNode::Call { func: Box::new(new_func), type_args: subst_targs, args: new_args }
                    }
                } else {
                    let new_func = self.rebuild_expr(func, b);
                    ExprNode::Call { func: Box::new(new_func), type_args: subst_targs, args: new_args }
                }
            }
            // a generic fn taken by value: mint its instance (the turbofish fully
            // determines it) and become a bare `Var` of the mangled symbol - a
            // function pointer. Mirrors the user-generic-call arm, minus the call.
            ExprNode::FnRef { name, type_args } => {
                let fname = name.path.as_single()
                    .expect("FnRef name resolved to a single symbol");
                let subst_targs: Vec<GenericArg<'a>> =
                    type_args.iter().map(|ga| self.subst_targ(ga, b)).collect();
                if self.templates.contains_key(fname) {
                    let concrete: Vec<ConcreteArg<'a>> = subst_targs.iter().map(|ga| match ga {
                        GenericArg::Type(t) => ConcreteArg::Type(t.clone()),
                        GenericArg::Const(cv) => ConcreteArg::Const(cv.expect_lit()),
                    }).collect();
                    let mangled = self.request(fname, concrete, expr.span.clone());
                    ExprNode::Var(mangled)
                } else {
                    // typecheck guarantees a generic fn here; a non-template name is
                    // unreachable, but stay total and emit the bare name.
                    ExprNode::Var(fname)
                }
            }
            ExprNode::Slice(elems) =>
                ExprNode::Slice(elems.iter().map(|e| self.rebuild_expr(e, b)).collect()),
            // the repeat count may be this instance's own const param (`[x; N]`
            // inside an `f<const N>`), so substitute it the same way an array
            // *type*'s length is substituted.
            ExprNode::Repeat { value, count } => ExprNode::Repeat {
                value: Box::new(self.rebuild_expr(value, b)),
                count: subst_cv(count, b),
            },
            ExprNode::Struct { name, type_args, fields } => {
                let new_fields: Vec<(&'a str, Expr<'a>)> =
                    fields.iter().map(|(f, e)| (*f, self.rebuild_expr(e, b))).collect();
                // a generic literal written without a turbofish carries an empty
                // one on the node; the typechecker recovered its args from the
                // field values, so fall back to those (same channel as a bare
                // generic call).
                let type_args: &[GenericArg<'a>] = if type_args.is_empty() {
                    self.inferred_type_args.get(&expr.id).map_or(type_args, Vec::as_slice)
                } else {
                    type_args
                };
                if type_args.is_empty() {
                    ExprNode::Struct { name: name.clone(), type_args: Vec::new(), fields: new_fields }
                } else if self.enum_templates.contains_key(&name.def) {
                    // a generic-enum struct-style variant literal with turbofish
                    // (`Result::Ok::<i32, str> { val: x }`): mangle to the concrete
                    // instance and rewrite the literal's enum segment to it
                    // (`Result$i32$str::Ok`), dropping the turbofish. The name to
                    // request against is the enum segment, not the whole path.
                    let cargs: Vec<ConcreteArg<'a>> = type_args.iter()
                        .map(|ga| self.concrete_targ(ga, b)).collect();
                    let inst = self.request_enum(name.def, cargs);
                    let mut new_name = name.clone();
                    new_name.def = inst;
                    ExprNode::Struct { name: new_name, type_args: Vec::new(), fields: new_fields }
                } else {
                    // a generic struct literal -> its concrete instance. mangle
                    // exactly like the generic *type* `Option<i32>`: substitute the
                    // args, request the instance, swap in the flat name and drop the
                    // turbofish (mil looks the fields up by this name).
                    let concrete = Type::Named { def: name.def, args: type_args.to_vec() };
                    let Type::Named { def: inst, .. } = self.subst_ty(&concrete, b) else {
                        unreachable!("subst_ty of a Named is always a Named")
                    };
                    let mut new_name = name.clone();
                    new_name.def = inst;
                    ExprNode::Struct { name: new_name, type_args: Vec::new(), fields: new_fields }
                }
            }
            ExprNode::Access { base, field } =>
                ExprNode::Access { base: Box::new(self.rebuild_expr(base, b)), field },
            ExprNode::Index { slice, index } => ExprNode::Index {
                slice: Box::new(self.rebuild_expr(slice, b)),
                index: Box::new(self.rebuild_expr(index, b)),
            },
            ExprNode::Unary { op, operand } =>
                ExprNode::Unary { op: *op, operand: Box::new(self.rebuild_expr(operand, b)) },
            ExprNode::Binary { op, left, right } => ExprNode::Binary {
                op: *op,
                left: Box::new(self.rebuild_expr(left, b)),
                right: Box::new(self.rebuild_expr(right, b)),
            },
            // leaves: copy as-is (fresh id)
            ExprNode::Bool(v) => ExprNode::Bool(*v),
            ExprNode::Int8(v) => ExprNode::Int8(*v),
            ExprNode::Int16(v) => ExprNode::Int16(*v),
            ExprNode::Int32(v) => ExprNode::Int32(*v),
            ExprNode::Int64(v) => ExprNode::Int64(*v),
            ExprNode::Uint8(v) => ExprNode::Uint8(*v),
            ExprNode::Uint16(v) => ExprNode::Uint16(*v),
            ExprNode::Uint32(v) => ExprNode::Uint32(*v),
            ExprNode::Uint64(v) => ExprNode::Uint64(*v),
            ExprNode::Float32(v) => ExprNode::Float32(*v),
            ExprNode::Float64(v) => ExprNode::Float64(*v),
            // a width-less literal keeps no type of its own here: the fresh node
            // is re-typed from context by the post-mono typecheck pass, exactly
            // as the original was.
            ExprNode::IntLit(v) => ExprNode::IntLit(*v),
            ExprNode::FloatLit(v) => ExprNode::FloatLit(*v),
            ExprNode::Str(s) => ExprNode::Str(s),
            // a bound const param used as a value becomes a typed literal; any
            // other name is copied through.
            ExprNode::Var(name) => match b.consts.get(name) {
                Some(cb) => const_literal(cb),
                None => ExprNode::Var(name),
            },
            // a unit enum variant. A bare path has no turbofish syntax, so a
            // generic enum's is only reachable here when the typechecker took
            // its arguments from the surrounding expectation (`let x: Option<i32>
            // = Option::None;`) and recorded them under the node id: mangle to
            // that instance and rewrite the enum segment to it, exactly as the
            // `Call` arm does for `Option::None::<i32>()`. Any other path is
            // copied through.
            ExprNode::Path(path) => {
                match self.inferred_type_args.get(&expr.id)
                    .filter(|_| self.enum_templates.contains_key(&path.def))
                {
                    Some(targs) => {
                        let concrete: Vec<ConcreteArg<'a>> = targs.iter()
                            .map(|ga| match self.subst_targ(ga, b) {
                                GenericArg::Type(t) => ConcreteArg::Type(t),
                                GenericArg::Const(cv) => ConcreteArg::Const(cv.expect_lit()),
                            }).collect();
                        let inst = self.request_enum(path.def, concrete);
                        let mut new_path = path.clone();
                        new_path.def = inst;
                        ExprNode::Path(new_path)
                    }
                    None => ExprNode::Path(path.clone()),
                }
            }
        };
        Metadata::new(node, expr.span.clone())
    }

    fn rebuild_stmt(&mut self, stmt: &Stmt<'a>, b: &Bindings<'a>) -> Stmt<'a> {
        let node = match &stmt.value {
            StmtNode::Expr(e) => StmtNode::Expr(self.rebuild_expr(e, b)),
            StmtNode::Block(stmts) =>
                StmtNode::Block(stmts.iter().map(|s| self.rebuild_stmt(s, b)).collect()),
            // a bare `let x = e;` gets the annotation it did not write, taken
            // from what the checker recorded for the binding. This is the seam:
            // the option is `None` only before this point, so every pass after
            // mono reads the type off the node (`Stmt::declared_ty`) without
            // caring whether the source spelled it out.
            //
            // The lookup is keyed by the *original* statement's id, which is why
            // it happens here rather than after the rebuild - `Metadata::new`
            // below mints a fresh one.
            StmtNode::Declare { name, ty, value } => StmtNode::Declare {
                name,
                ty: Some(self.subst_ty(ty.as_ref().unwrap_or_else(|| &self.node_types[&stmt.id]), b)),
                value: self.rebuild_expr(value, b),
            },
            StmtNode::Assign { left, value } => StmtNode::Assign {
                left: self.rebuild_expr(left, b),
                value: self.rebuild_expr(value, b),
            },
            StmtNode::If { condition, then_branch, else_branch } => StmtNode::If {
                condition: self.rebuild_expr(condition, b),
                then_branch: Box::new(self.rebuild_stmt(then_branch, b)),
                else_branch: else_branch.as_ref().map(|s| Box::new(self.rebuild_stmt(s, b))),
            },
            StmtNode::While { condition, body } => StmtNode::While {
                condition: self.rebuild_expr(condition, b),
                body: Box::new(self.rebuild_stmt(body, b)),
            },
            StmtNode::Match { scrutinee, arms } => StmtNode::Match {
                scrutinee: self.rebuild_expr(scrutinee, b),
                // patterns carry no substitutable types; clone them, rebuild bodies.
                arms: arms.iter()
                    .map(|(p, body)| (p.clone(), Box::new(self.rebuild_stmt(body, b))))
                    .collect(),
            },
            StmtNode::Return(e) => StmtNode::Return(e.as_ref().map(|e| self.rebuild_expr(e, b))),
            StmtNode::Continue => StmtNode::Continue,
            StmtNode::Break => StmtNode::Break,
        };
        Metadata::new(node, stmt.span.clone())
    }

    /// Rebuild a function with type params substituted per `b`, an optional new
    /// (mangled) name, and no generics. call sites inside get rewritten and any
    /// generic calls found get queued.
    fn rebuild_function(
        &mut self,
        tl: &TopLevel<'a>,
        b: &Bindings<'a>,
        name_override: Option<(DefId, &'a str)>,
    ) -> TopLevel<'a> {
        let TopLevelNode::Function { name, def, is_pub, attributes, params, return_type, body, .. } = &tl.value
        else { unreachable!("rebuild_function called on a non-function") };

        // any struct instance requested while substituting this function's types
        // reports against the function's span.
        self.cur_span = tl.span.clone();
        let new_params: Vec<(&'a str, Type<'a>)> =
            params.iter().map(|(pn, ty)| (*pn, self.subst_ty(ty, b))).collect();
        let new_return = self.subst_ty(return_type, b);
        let new_body: Vec<Stmt<'a>> = body.iter().map(|s| self.rebuild_stmt(s, b)).collect();

        Metadata::new(
            TopLevelNode::Function {
                name: name_override.map_or(*name, |(_, n)| n),
                def: name_override.map_or(*def, |(d, _)| d),
                is_pub: *is_pub,
                attributes: attributes.clone(),
                // a monomorphic instance has no parameters left, so nothing
                // remains for a clause to bound either.
                generics: Vec::new(),
                where_bounds: Vec::new(),
                params: new_params,
                return_type: new_return,
                body: new_body,
            },
            tl.span.clone(),
        )
    }

    /// Rebuild a struct with its field types substituted per `b`. A concrete
    /// generic-instance field (`buf: Vec<u8>`) collapses to its flat instance
    /// name (`Vec$u8`) and the instance is requested, exactly as it would inside
    /// a generic struct's own fields. Used for NON-generic structs: they are not
    /// templates, but a field that names a concrete generic type still has to be
    /// rewritten and its instance emitted, or downstream passes never see it. For
    /// a struct with only scalar / plain-struct fields this is a no-op copy.
    fn rebuild_struct(&mut self, tl: &TopLevel<'a>, b: &Bindings<'a>) -> TopLevel<'a> {
        let TopLevelNode::Struct { name, def, is_pub, attributes, fields, .. } = &tl.value
        else { unreachable!("rebuild_struct called on a non-struct") };

        // instances requested while substituting these fields report against the
        // struct's own declaration span.
        self.cur_span = tl.span.clone();
        let new_fields: Vec<(&'a str, Type<'a>)> = fields.iter()
            .map(|(fname, fty)| (*fname, self.subst_ty(fty, b)))
            .collect();

        Metadata::new(
            TopLevelNode::Struct {
                name,
                def: *def,
                is_pub: *is_pub,
                attributes: attributes.clone(),
                generics: Vec::new(),
                fields: new_fields,
            },
            tl.span.clone(),
        )
    }

    /// Rebuild an enum with its variant payload types substituted per `b`, the
    /// enum analogue of `rebuild_struct`: a non-generic enum carrying a concrete
    /// generic instance in a payload (`A(Vec<u8>)`) needs that field collapsed to
    /// its flat instance name and requested. Discriminants and field names pass
    /// through unchanged.
    fn rebuild_enum(&mut self, tl: &TopLevel<'a>, b: &Bindings<'a>) -> TopLevel<'a> {
        let TopLevelNode::Enum { name, def, is_pub, attributes, variants, .. } = &tl.value
        else { unreachable!("rebuild_enum called on a non-enum") };

        self.cur_span = tl.span.clone();
        let new_variants: Vec<(&'a str, Option<i64>, Vec<(&'a str, Type<'a>)>)> = variants.iter()
            .map(|(vname, disc, payload)| (
                *vname,
                *disc,
                payload.iter().map(|(fname, fty)| (*fname, self.subst_ty(fty, b))).collect(),
            ))
            .collect();

        Metadata::new(
            TopLevelNode::Enum {
                name,
                def: *def,
                is_pub: *is_pub,
                attributes: attributes.clone(),
                generics: Vec::new(),
                variants: new_variants,
            },
            tl.span.clone(),
        )
    }
}

/// Expand every generic function reachable from a concrete call site into
/// concrete instances, drop the templates. non-generic functions, externs and
/// structs stay (with call sites rewritten).
///
/// `defs` is extended in place: each instance minted here is registered against
/// the template it specializes, which is how the post-mono typecheck matches a
/// template-named match pattern to an instance-typed scrutinee, and how the
/// alloc check reports `alloc::<Vec2>` instead of `std.alloc$alloc$Vec2`.
/// `node_types` is the pre-mono typecheck's inference result: mono needs it only
/// to know what a method call's receiver is, which is what lets a generic
/// `extend` block's methods be instantiated at all (see `generic_method_call`).
pub fn monomorphize<'a>(
    program: &[TopLevel<'a>],
    defs: &mut Defs<'a>,
    arena: &'a Bump,
    node_types: &HashMap<usize, Type<'a>>,
    inferred_type_args: &HashMap<usize, Vec<GenericArg<'a>>>,
    impls: &[ImplDecl<'a>],
) -> Result<Vec<TopLevel<'a>>, Error> {
    // functions stay keyed by emitted name: a call site names its callee, and
    // there is no `Type` involved to carry an identity. Types key by identity.
    let templates: HashMap<&'a str, (DefId, &TopLevel<'a>)> = program.iter()
        .filter_map(|tl| match &tl.value {
            TopLevelNode::Function { name, def, generics, .. } if !generics.is_empty() =>
                Some((*name, (*def, tl))),
            _ => None,
        })
        .collect();

    let struct_templates: HashMap<DefId, &TopLevel<'a>> = program.iter()
        .filter_map(|tl| match &tl.value {
            TopLevelNode::Struct { def, generics, .. } if !generics.is_empty() =>
                Some((*def, tl)),
            _ => None,
        })
        .collect();

    let enum_templates: HashMap<DefId, &TopLevel<'a>> = program.iter()
        .filter_map(|tl| match &tl.value {
            TopLevelNode::Enum { def, generics, .. } if !generics.is_empty() =>
                Some((*def, tl)),
            _ => None,
        })
        .collect();

    let mut m = Mono {
        arena, templates, struct_templates, enum_templates,
        queue: VecDeque::new(), seen: HashMap::new(),
        struct_queue: VecDeque::new(), struct_seen: HashMap::new(),
        enum_queue: VecDeque::new(), enum_seen: HashMap::new(),
        cur_span: Span::unknown(),
        members: defs.members().clone(),
        defs,
        node_types,
        inferred_type_args,
        impls,
    };
    let empty = Bindings::empty();

    // rebuild concrete functions (their call sites seed the queue), keyed by
    // original position so we can re-emit in source order. Non-generic structs
    // and enums are rebuilt here too (into `concrete_aggregates`): a field that
    // names a concrete generic instance (`buf: Vec<u8>`) has to be collapsed to
    // its flat name and requested, and doing it here - before the struct/enum
    // queues drain below - means those requests are picked up in the same drain.
    let mut concrete: HashMap<usize, TopLevel<'a>> = HashMap::new();
    let mut concrete_aggregates: HashMap<usize, TopLevel<'a>> = HashMap::new();
    let mut concrete_globals: HashMap<usize, TopLevel<'a>> = HashMap::new();
    for (i, tl) in program.iter().enumerate() {
        match &tl.value {
            TopLevelNode::Function { generics, .. } if !generics.is_empty() => {} // template
            TopLevelNode::Function { .. } => {
                concrete.insert(i, m.rebuild_function(tl, &empty, None));
            }
            // generic struct/enum templates are handled by the drain loops below.
            TopLevelNode::Struct { generics, .. } if !generics.is_empty() => {}
            TopLevelNode::Enum { generics, .. } if !generics.is_empty() => {}
            TopLevelNode::Struct { .. } => {
                concrete_aggregates.insert(i, m.rebuild_struct(tl, &empty));
            }
            TopLevelNode::Enum { .. } => {
                concrete_aggregates.insert(i, m.rebuild_enum(tl, &empty));
            }
            // a global's initializer can reference a generic fn by value
            // (`const clap_entry = ... entry_init::<Gain> ...`); rebuild it so that
            // reference mints its instance (seeding the drain below) and is
            // rewritten to the mangled symbol. Globals are never generic themselves,
            // so `empty` bindings suffice.
            TopLevelNode::Global { name, def, is_pub, attributes, ty, value } => {
                concrete_globals.insert(i, Metadata::new(
                    TopLevelNode::Global {
                        name, def: *def, is_pub: *is_pub, attributes: attributes.clone(),
                        ty: ty.clone(), value: m.rebuild_expr(value, &empty),
                    },
                    tl.span.clone(),
                ));
            }
            TopLevelNode::Extern { .. } => {}
            // traits emit no code; they're dropped from the monomorphized output.
            TopLevelNode::Trait { .. } => {}
            TopLevelNode::Alias { .. } => unreachable!("aliases expanded before mono"),
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before mono"),
        }
    }

    // build each requested instantiation. instances can request more, so drain
    // till the queue is empty. group by template so they emit next to it.
    // NOTE: the depth/count guards below fire after pop, i.e. after this item was
    // already built+queued - so we overshoot the cutoff by one instance. fine as
    // a backstop, just not a tight bound.
    let mut instances: HashMap<DefId, Vec<TopLevel<'a>>> = HashMap::new();
    let mut materialized = 0usize;
    while let Some(inst) = m.queue.pop_front() {
        let base_name = m.defs.get(inst.base).source_name;
        materialized += 1;
        if let Some(deep) = inst.args.iter().find_map(|a| match a {
            ConcreteArg::Type(t) if type_depth(t) > TYPE_DEPTH_LIMIT => Some(t),
            _ => None,
        }) {
            return Err(Error::new(inst.span,
                format!("type argument nested deeper than {}", TYPE_DEPTH_LIMIT))
                .with_label(inst.span, format!(
                    "monomorphizing '{}' produced `{}`", base_name, deep))
                .with_note("this usually means unbounded generic recursion: a generic \
                            function calling itself at an ever-growing type"));
        }
        if materialized > INSTANTIATION_LIMIT {
            return Err(Error::new(inst.span, format!(
                "monomorphization exceeded {} instantiations at call to '{}'",
                INSTANTIATION_LIMIT, base_name,
            )));
        }
        let tl = m.template_of(inst.base);
        let TopLevelNode::Function { generics, .. } = &tl.value else { unreachable!() };
        // pair each declared generic param with its concrete arg (positional).
        let mut bindings = Bindings::empty();
        for (gp, arg) in generics.iter().zip(&inst.args) {
            match (gp, arg) {
                (GenericParam::Type { name: n, .. }, ConcreteArg::Type(t)) => {
                    bindings.types.insert(n, t.clone());
                }
                (GenericParam::Const(n, ty), ConcreteArg::Const(v)) => {
                    bindings.consts.insert(n, ConstBind { val: *v, ty: ty.clone() });
                }
                _ => unreachable!("generic param/arg kind mismatch - validated in typecheck"),
            }
        }
        let f = m.rebuild_function(tl, &bindings, Some((inst.def, inst.mangled)));
        instances.entry(inst.base).or_default().push(f);
    }

    // build each requested generic-struct/enum instance. substituting a field or
    // payload type can request further instances of EITHER kind (`Box<Option<i32>>`
    // needs `Option$i32`; `Option<Buf<i32,4>>` needs `Buf$i32$4`), so drain BOTH
    // queues together to a fixpoint - whichever has a pending item goes next. the
    // same depth/count guards backstop growing recursion (`struct Bad<T> { p:
    // *Bad<*T> }`). fn instances above already seeded these queues via their
    // param/return/body types.
    let mut struct_instances: HashMap<DefId, Vec<TopLevel<'a>>> = HashMap::new();
    let mut enum_instances: HashMap<DefId, Vec<TopLevel<'a>>> = HashMap::new();
    let mut materialized = 0usize;
    loop {
        if let Some(inst) = m.struct_queue.pop_front() {
            let base_name = m.defs.get(inst.base).source_name;
            materialized += 1;
            if let Some(deep) = inst.args.iter().find_map(|a| match a {
                ConcreteArg::Type(t) if type_depth(t) > TYPE_DEPTH_LIMIT => Some(t),
                _ => None,
            }) {
                return Err(Error::new(inst.span,
                    format!("type argument nested deeper than {}", TYPE_DEPTH_LIMIT))
                    .with_label(inst.span, format!(
                        "monomorphizing struct '{}' produced `{}`", base_name, deep))
                    .with_note("this usually means an unbounded generic struct: one whose \
                                field mentions itself at an ever-growing type"));
            }
            if materialized > INSTANTIATION_LIMIT {
                return Err(Error::new(inst.span, format!(
                    "monomorphization exceeded {} struct/enum instantiations at '{}'",
                    INSTANTIATION_LIMIT, base_name,
                )));
            }
            let tl = m.struct_templates[&inst.base];
            let TopLevelNode::Struct { generics, fields, attributes, is_pub, .. } = &tl.value
            else { unreachable!() };
            // bind each declared param to its concrete arg (positional): type params
            // to types, const params to values (so `[T; N]` fields substitute both).
            let mut bindings = Bindings::empty();
            for (gp, arg) in generics.iter().zip(&inst.args) {
                match (gp, arg) {
                    (GenericParam::Type { name: n, .. }, ConcreteArg::Type(t)) => {
                        bindings.types.insert(n, t.clone());
                    }
                    (GenericParam::Const(n, ty), ConcreteArg::Const(v)) => {
                        bindings.consts.insert(n, ConstBind { val: *v, ty: ty.clone() });
                    }
                    _ => unreachable!("struct generic param/arg kind mismatch - validated in typecheck"),
                }
            }
            // field types report against this instance's span while being substituted.
            m.cur_span = inst.span.clone();
            let new_fields: Vec<(&'a str, Type<'a>)> = fields.iter()
                .map(|(fname, fty)| (*fname, m.subst_ty(fty, &bindings)))
                .collect();
            let s = Metadata::new(
                TopLevelNode::Struct {
                    name: inst.mangled,
                    def: inst.def,
                    is_pub: *is_pub,
                    attributes: attributes.clone(),
                    generics: Vec::new(),
                    fields: new_fields,
                },
                tl.span.clone(),
            );
            struct_instances.entry(inst.base).or_default().push(s);
            continue;
        }
        if let Some(inst) = m.enum_queue.pop_front() {
            let base_name = m.defs.get(inst.base).source_name;
            materialized += 1;
            if let Some(deep) = inst.args.iter().find_map(|a| match a {
                ConcreteArg::Type(t) if type_depth(t) > TYPE_DEPTH_LIMIT => Some(t),
                _ => None,
            }) {
                return Err(Error::new(inst.span,
                    format!("type argument nested deeper than {}", TYPE_DEPTH_LIMIT))
                    .with_label(inst.span, format!(
                        "monomorphizing enum '{}' produced `{}`", base_name, deep))
                    .with_note("this usually means an unbounded generic enum: one whose \
                                payload mentions itself at an ever-growing type"));
            }
            if materialized > INSTANTIATION_LIMIT {
                return Err(Error::new(inst.span, format!(
                    "monomorphization exceeded {} struct/enum instantiations at '{}'",
                    INSTANTIATION_LIMIT, base_name,
                )));
            }
            let tl = m.enum_templates[&inst.base];
            let TopLevelNode::Enum { generics, variants, attributes, is_pub, .. } = &tl.value
            else { unreachable!() };
            let mut bindings = Bindings::empty();
            for (gp, arg) in generics.iter().zip(&inst.args) {
                match (gp, arg) {
                    (GenericParam::Type { name: n, .. }, ConcreteArg::Type(t)) => {
                        bindings.types.insert(n, t.clone());
                    }
                    (GenericParam::Const(n, ty), ConcreteArg::Const(v)) => {
                        bindings.consts.insert(n, ConstBind { val: *v, ty: ty.clone() });
                    }
                    _ => unreachable!("enum generic param/arg kind mismatch - validated in typecheck"),
                }
            }
            // discriminants don't depend on the bound params - only payload field
            // types are substituted; field names (real or synthesized "0"/"1") and
            // discriminant values pass through unchanged.
            m.cur_span = inst.span.clone();
            let new_variants: Vec<(&'a str, Option<i64>, Vec<(&'a str, Type<'a>)>)> = variants.iter()
                .map(|(vname, disc, payload)| (
                    *vname,
                    *disc,
                    payload.iter().map(|(fname, fty)| (*fname, m.subst_ty(fty, &bindings))).collect(),
                ))
                .collect();
            let e = Metadata::new(
                TopLevelNode::Enum {
                    name: inst.mangled,
                    def: inst.def,
                    is_pub: *is_pub,
                    attributes: attributes.clone(),
                    generics: Vec::new(),
                    variants: new_variants,
                },
                tl.span.clone(),
            );
            enum_instances.entry(inst.base).or_default().push(e);
            continue;
        }
        break;
    }

    // reassemble in source order: each template -> its instances (or nothing if
    // it was never instantiated).
    let mut output: Vec<TopLevel<'a>> = Vec::with_capacity(program.len());
    for (i, tl) in program.iter().enumerate() {
        match &tl.value {
            TopLevelNode::Function { def, generics, .. } if !generics.is_empty() => {
                if let Some(insts) = instances.remove(def) {
                    output.extend(insts);
                }
            }
            TopLevelNode::Function { .. } => output.push(concrete.remove(&i).unwrap()),
            // a generic struct template drops out (its `Param` fields never lay
            // out), replaced by its concrete instances - emitted here, next to it.
            TopLevelNode::Struct { def, generics, .. } if !generics.is_empty() => {
                if let Some(insts) = struct_instances.remove(def) {
                    output.extend(insts);
                }
            }
            // same story for a generic enum template: dropped, replaced by its
            // concrete instances (each a plain, fully-substituted data enum - the
            // typecheck/mil/backend pipeline handles it exactly like a hand-written
            // one, per the field-less/data-enum machinery already in place).
            TopLevelNode::Enum { def, generics, .. } if !generics.is_empty() => {
                if let Some(insts) = enum_instances.remove(def) {
                    output.extend(insts);
                }
            }
            // a non-generic struct/enum: emit the rebuilt version, whose concrete
            // generic-instance field types were collapsed and requested up front.
            TopLevelNode::Struct { .. } | TopLevelNode::Enum { .. } =>
                output.push(concrete_aggregates.remove(&i).unwrap()),
            TopLevelNode::Global { .. } => output.push(concrete_globals.remove(&i).unwrap()),
            TopLevelNode::Extern { .. } => output.push(tl.clone()),
            // traits emit no code and are not carried into the concrete program.
            TopLevelNode::Trait { .. } => {}
            TopLevelNode::Alias { .. } => unreachable!("aliases expanded before mono"),
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before mono"),
        }
    }

    Ok(output)
}
