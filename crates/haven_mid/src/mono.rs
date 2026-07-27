//! monomorphization: rewrite generic (type-param) functions into concrete ones.
//!
//! runs AST -> AST after the first typecheck (which tells us which functions are
//! generic) and before MIL lowering. every distinct instantiation reachable from
//! a concrete call site becomes its own function with a mangled name
//! (`id::<i32>` -> `id$i32`), and the call site is rewritten to call it directly.
//! the result gets re-typechecked so node_types is filled in for the new nodes.
//! MIL never sees a generic.
//!
//! type params only. const generics on user functions are rejected back in
//! typecheck, so they never get here.

use std::collections::{HashMap, VecDeque};

use bumpalo::Bump;

use haven_common::ast::*;
use haven_common::defs::Defs;

/// one requested instantiation: `base` specialized to `args`, emitted as
/// `mangled`. `span` is the call site that first asked for it (for errors).
struct Instantiation<'a> {
    base: &'a str,
    args: Vec<ConcreteArg<'a>>,
    mangled: &'a str,
    span: Span,
}

/// one requested generic-struct instantiation: `struct Buf<T, const N>` at `args`
/// (`[i32, 8]`), emitted as the concrete `struct Buf$i32$8`. args are types and/or
/// const values, matching the struct's declared params. `span` points at the
/// context that first named the type, for the depth-limit diagnostic.
struct StructInstantiation<'a> {
    base: &'a str,
    args: Vec<ConcreteArg<'a>>,
    mangled: &'a str,
    span: Span,
}

/// one requested generic-enum instantiation: `enum Option<T>` at `args` (`[i32]`),
/// emitted as the concrete `enum Option$i32`. Mirrors `StructInstantiation` - a
/// data-carrying enum is lowered as a struct-shaped aggregate, so once
/// monomorphized it goes through exactly the same downstream machinery.
struct EnumInstantiation<'a> {
    base: &'a str,
    args: Vec<ConcreteArg<'a>>,
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
    /// generic function templates, by name.
    templates: HashMap<&'a str, &'p TopLevel<'a>>,
    /// instantiations still to build.
    queue: VecDeque<Instantiation<'a>>,
    /// mangled name per instantiation we've already asked for, keyed by the
    /// mangled string, so repeated call sites reuse one instance (one alloc).
    seen: HashMap<InstanceKey<'a>, &'a str>,
    /// generic struct templates, by name (`Option` -> `struct Option<T> {...}`).
    struct_templates: HashMap<&'a str, &'p TopLevel<'a>>,
    /// generic-struct instantiations still to build.
    struct_queue: VecDeque<StructInstantiation<'a>>,
    /// mangled name per struct instance already requested, keyed by the mangled
    /// string, so every concrete `Option$i32` use shares one emitted struct.
    struct_seen: HashMap<InstanceKey<'a>, &'a str>,
    /// generic enum templates, by name (`Option` -> `enum Option<T> {...}`).
    enum_templates: HashMap<&'a str, &'p TopLevel<'a>>,
    /// generic-enum instantiations still to build.
    enum_queue: VecDeque<EnumInstantiation<'a>>,
    /// mangled name per enum instance already requested, keyed by the mangled
    /// string, so every concrete `Option$i32` use shares one emitted enum.
    enum_seen: HashMap<InstanceKey<'a>, &'a str>,
    /// best-effort span for the context currently being rebuilt, so a struct or
    /// enum instance requested deep inside `subst_ty` (which has no span of its
    /// own) still gets a source location for the depth-limit error.
    cur_span: Span,
    /// definition arena, extended as we go: every instance minted below is
    /// registered here with the template it came from, so post-mono passes can
    /// map an instance back to its template without taking its name apart.
    defs: &'p mut Defs<'a>,
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

/// What identifies one instantiation: the template's name plus the concrete
/// arguments it was specialized with. The instance caches key on *this* rather
/// than on the mangled string, so `mangle_ty` no longer has to be injective for
/// the compiler to be correct - see its doc comment.
type InstanceKey<'a> = (&'a str, Vec<ConcreteArg<'a>>);

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
        Type::Int32  => ExprNode::Int32(cb.val as i32),
        Type::Int64  => ExprNode::Int64(cb.val as i64),
        Type::Uint8  => ExprNode::Uint8(cb.val as u8),
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
fn mangle_ty(ty: &Type) -> String {
    match ty {
        Type::Void => "void".into(),
        Type::Bool => "bool".into(),
        Type::Int8 => "i8".into(),
        Type::Int32 => "i32".into(),
        Type::Int64 => "i64".into(),
        Type::Uint8 => "u8".into(),
        Type::Uint32 => "u32".into(),
        Type::Uint64 => "u64".into(),
        Type::Float32 => "f32".into(),
        Type::Float64 => "f64".into(),
        Type::Str => "str".into(),
        Type::Path { path, .. } => Type::unresolved(path),
        Type::Enum { name, .. } => (*name).into(),
        Type::Pointer(inner) => format!(".ptr{}", mangle_ty(inner)),
        Type::Array(inner, n) => format!(".arr{}.{}", n.expect_lit(), mangle_ty(inner)),
        Type::Slice(inner) => format!(".slice{}", mangle_ty(inner)),
        Type::Simd(inner, n) => format!(".simd{}.{}", n.expect_lit(), mangle_ty(inner)),
        Type::Struct { name, args } if args.is_empty() => (*name).into(),
        // a generic struct instance is normally flattened to a bare name before it
        // reaches here (subst_ty requests it); encode defensively with balanced
        // brackets so nested args stay unambiguous.
        Type::Struct { name, args } => {
            let inner = args.iter().map(mangle_generic_arg).collect::<Vec<_>>().join(".");
            format!(".struct.{}.{}", name, inner)
        }
        // Neither should appear in a fully-concrete instantiation; encode them
        // defensively rather than panicking so a bug surfaces as a bad symbol.
        Type::Param(name) => format!(".param.{}", name),
        Type::Function { params, return_type } => {
            let ps = params.iter().map(mangle_ty).collect::<Vec<_>>().join(".");
            format!(".fn{}.{}.ret{}", params.len(), ps, mangle_ty(return_type))
        }
    }
}

/// Mangle a generic argument in a struct type's arg list (a type or a const
/// value). Only reached on the defensive not-yet-flattened path in `mangle_ty`.
fn mangle_generic_arg(ga: &GenericArg) -> String {
    match ga {
        GenericArg::Type(t) => mangle_ty(t),
        GenericArg::Const(cv) => cv.to_string(),
    }
}

/// Identifier-safe fragment for one generic arg: a mangled type, or a const's
/// decimal value (`4`). Const values are pure digits and no type mangles to bare
/// digits, so the two never collide within a single arg list.
fn mangle_arg(arg: &ConcreteArg) -> String {
    match arg {
        ConcreteArg::Type(t) => mangle_ty(t),
        ConcreteArg::Const(n) => n.to_string(),
    }
}

/// `id`, `[i32]` -> `id$slice_i32`; `splat`, `f32`, `4` -> `splat$f32$4`.
fn mangle_name(base: &str, args: &[ConcreteArg]) -> String {
    let parts = args.iter().map(mangle_arg).collect::<Vec<_>>().join("$");
    format!("{}${}", base, parts)
}

/// Human-readable spelling of an instance, for diagnostics only. Strips the
/// module slug off `base` (`std.alloc$alloc` -> `alloc`, since the slug is a
/// linkage detail, not something the user wrote) and re-attaches the turbofish:
/// `alloc`, `[Vec2]` -> `alloc::<Vec2>`. Best-effort; never fed back into the
/// compiler.
fn display_name(base: &str, args: &[ConcreteArg]) -> String {
    let leaf = base.rsplit('$').next().unwrap_or(base);
    let targs = args.iter().map(|a| match a {
        ConcreteArg::Type(t) => t.to_string(),
        ConcreteArg::Const(n) => n.to_string(),
    }).collect::<Vec<_>>().join(", ");
    format!("{}::<{}>", leaf, targs)
}

impl<'p, 'a> Mono<'p, 'a> {
    /// Record an instantiation request, return its (stable) mangled name.
    /// de-dupes so each distinct instance is built exactly once.
    fn request(&mut self, base: &'a str, args: Vec<ConcreteArg<'a>>, span: Span) -> &'a str {
        let key = (base, args.clone());
        if let Some(&m) = self.seen.get(&key) {
            return m;
        }
        let mangled: &'a str = self.arena.alloc_str(&mangle_name(base, &args));
        self.seen.insert(key, mangled);
        self.defs.add_instance(mangled, base, display_name(base, &args));
        self.queue.push_back(Instantiation { base, args, mangled, span });
        mangled
    }

    /// Record a generic-struct instantiation request, return its mangled name
    /// (`Buf` + `[i32, 8]` -> `Buf$i32$8`). de-dupes so each concrete instance is
    /// built once. Enqueues onto the struct queue, drained after all functions.
    fn request_struct(&mut self, base: &'a str, args: Vec<ConcreteArg<'a>>) -> &'a str {
        let key = (base, args.clone());
        if let Some(&m) = self.struct_seen.get(&key) {
            return m;
        }
        let mangled: &'a str = self.arena.alloc_str(&mangle_name(base, &args));
        self.struct_seen.insert(key, mangled);
        self.defs.add_instance(mangled, base, display_name(base, &args));
        self.struct_queue.push_back(StructInstantiation {
            base, args, mangled, span: self.cur_span.clone(),
        });
        mangled
    }

    /// Record a generic-enum instantiation request, return its mangled name
    /// (`Option` + `[i32]` -> `Option$i32`). de-dupes so each concrete instance is
    /// built once. Enqueues onto the enum queue, drained (interleaved with the
    /// struct queue, since either can request the other) after all functions.
    fn request_enum(&mut self, base: &'a str, args: Vec<ConcreteArg<'a>>) -> &'a str {
        let key = (base, args.clone());
        if let Some(&m) = self.enum_seen.get(&key) {
            return m;
        }
        let mangled: &'a str = self.arena.alloc_str(&mangle_name(base, &args));
        self.enum_seen.insert(key, mangled);
        self.defs.add_instance(mangled, base, display_name(base, &args));
        self.enum_queue.push_back(EnumInstantiation {
            base, args, mangled, span: self.cur_span.clone(),
        });
        mangled
    }

    /// Substitute bound type and const params in `ty` with their concrete
    /// bindings. In the raw pre-typecheck AST a type param looks like
    /// `Type::Struct(name)` (parser can't tell it from a real struct), so both
    /// Struct and Param get substituted when bound.
    ///
    /// A concrete generic-struct type (`Option<i32>`, args non-empty) is
    /// monomorphized here: its args are substituted, the instance is requested,
    /// and the mangled flat name (`Option$i32`, no args) is returned. Every
    /// downstream pass therefore only ever sees ordinary no-arg structs.
    ///
    /// FIXME: a real struct named the same as a bound type param gets wrongly
    /// rewritten inside a generic body (e.g. `struct T` used inside `f<T>`). no
    /// scope check distinguishes them. edge case, but there's no guard.
    fn subst_ty(&mut self, ty: &Type<'a>, b: &Bindings<'a>) -> Type<'a> {
        match ty {
            Type::Param(n) if b.types.contains_key(n) => b.types[n].clone(),
            // a bare struct name that's actually a bound type param; substitute it.
            Type::Struct { name, args } if args.is_empty() && b.types.contains_key(name) =>
                b.types[name].clone(),
            // a concrete generic-struct instance: substitute inside its args
            // (types and const values), then mangle + request it, collapsing to the
            // flat instance name. The raw (pre-typecheck) AST never distinguishes a
            // struct name from an enum name - both parse as `Type::Struct{name,args}`
            // (see the front end's own comment on this) - so which queue to request
            // from is decided here by template-table membership. Either way the
            // result stays `Type::plain_struct(mangled)`: the SECOND typecheck pass
            // (post-mono) re-derives `Type::Enum` from this bare form via its own
            // `resolve_type`, once it sees the concrete instance's own declaration.
            Type::Struct { name, args } if !args.is_empty() => {
                let cargs: Vec<ConcreteArg<'a>> = args.iter().map(|a| match a {
                    GenericArg::Type(t) => ConcreteArg::Type(self.subst_ty(t, b)),
                    GenericArg::Const(cv) => ConcreteArg::Const(subst_cv(cv, b).expect_lit()),
                }).collect();
                let mangled = if self.enum_templates.contains_key(name) {
                    self.request_enum(name, cargs)
                } else {
                    self.request_struct(name, cargs)
                };
                Type::plain_struct(mangled)
            }
            // an ordinary (non-generic) struct: copy through.
            Type::Struct { name, .. } => Type::plain_struct(name),
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

    /// Substitute bound params in a turbofish argument. A const generic forwarded
    /// by name (`simd_load::<f32, N>`) reaches here as a bare-ident `Type` - but
    /// the name binds in `consts`, not `types` - so resolve it to a literal
    /// `Const` argument; the specialized call then re-typechecks against a
    /// concrete value.
    fn subst_targ(&mut self, ga: &GenericArg<'a>, b: &Bindings<'a>) -> GenericArg<'a> {
        match ga {
            GenericArg::Type(Type::Param(n)) if b.consts.contains_key(n) =>
                GenericArg::Const(ConstVal::Lit(b.consts[n].val)),
            GenericArg::Type(Type::Struct { name, args }) if args.is_empty() && b.consts.contains_key(name) =>
                GenericArg::Const(ConstVal::Lit(b.consts[name].val)),
            GenericArg::Type(t) => GenericArg::Type(self.subst_ty(t, b)),
            GenericArg::Const(cv) => GenericArg::Const(subst_cv(cv, b)),
        }
    }

    fn rebuild_expr(&mut self, expr: &Expr<'a>, b: &Bindings<'a>) -> Expr<'a> {
        let node = match &expr.value {
            ExprNode::Call { func, type_args, args } => {
                let new_args: Vec<Expr<'a>> =
                    args.iter().map(|a| self.rebuild_expr(a, b)).collect();
                // sub type params inside the turbofish (user generic calls +
                // intrinsics like `sizeof::<T>()`)
                let subst_targs: Vec<GenericArg<'a>> =
                    type_args.iter().map(|ga| self.subst_targ(ga, b)).collect();

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
                    if let Some((ename, _)) = path.as_variant()
                        .filter(|(en, _)| self.enum_templates.contains_key(en))
                    {
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
                        let mangled = self.request_enum(ename, concrete);
                        let new_func = Metadata::new(
                            ExprNode::Path(path.with_head(mangled)), func.span.clone());
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
            ExprNode::Slice(elems) =>
                ExprNode::Slice(elems.iter().map(|e| self.rebuild_expr(e, b)).collect()),
            ExprNode::Struct { name, type_args, fields } => {
                let new_fields: Vec<(&'a str, Expr<'a>)> =
                    fields.iter().map(|(f, e)| (*f, self.rebuild_expr(e, b))).collect();
                if type_args.is_empty() {
                    ExprNode::Struct { name: name.clone(), type_args: Vec::new(), fields: new_fields }
                } else if let Some((ename, _)) = name.as_variant()
                    .filter(|(en, _)| self.enum_templates.contains_key(en))
                {
                    // a generic-enum struct-style variant literal with turbofish
                    // (`Result::Ok::<i32, str> { val: x }`): mangle to the concrete
                    // instance and rewrite the literal's enum segment to it
                    // (`Result$i32$str::Ok`), dropping the turbofish. The name to
                    // request against is the enum segment, not the whole path.
                    let cargs: Vec<ConcreteArg<'a>> = type_args.iter().map(|ga| match ga {
                        GenericArg::Type(t) => ConcreteArg::Type(self.subst_ty(t, b)),
                        GenericArg::Const(cv) => ConcreteArg::Const(subst_cv(cv, b).expect_lit()),
                    }).collect();
                    let mangled = self.request_enum(ename, cargs);
                    ExprNode::Struct { name: name.with_head(mangled), type_args: Vec::new(), fields: new_fields }
                } else {
                    // a generic struct literal -> its concrete instance. mangle
                    // exactly like the generic *type* `Option<i32>`: substitute the
                    // args, request the instance, swap in the flat name and drop the
                    // turbofish (mil looks the fields up by this name).
                    let concrete = Type::Struct { name: name.last(), args: type_args.clone() };
                    let Type::Struct { name: mangled, .. } = self.subst_ty(&concrete, b) else {
                        unreachable!("subst_ty of a Struct is always a Struct")
                    };
                    ExprNode::Struct { name: Path::single(mangled), type_args: Vec::new(), fields: new_fields }
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
            ExprNode::Int32(v) => ExprNode::Int32(*v),
            ExprNode::Int64(v) => ExprNode::Int64(*v),
            ExprNode::Uint8(v) => ExprNode::Uint8(*v),
            ExprNode::Uint32(v) => ExprNode::Uint32(*v),
            ExprNode::Uint64(v) => ExprNode::Uint64(*v),
            ExprNode::Float32(v) => ExprNode::Float32(*v),
            ExprNode::Float64(v) => ExprNode::Float64(*v),
            ExprNode::Str(s) => ExprNode::Str(s),
            // a bound const param used as a value becomes a typed literal; any
            // other name is copied through.
            ExprNode::Var(name) => match b.consts.get(name) {
                Some(cb) => const_literal(cb),
                None => ExprNode::Var(name),
            },
            // a unit enum variant. A generic enum's is only reachable through the
            // call form (`Option::None::<i32>()`), handled in the `Call` arm above,
            // so nothing here needs substituting.
            ExprNode::Path(path) => ExprNode::Path(path.clone()),
        };
        Metadata::new(node, expr.span.clone())
    }

    fn rebuild_stmt(&mut self, stmt: &Stmt<'a>, b: &Bindings<'a>) -> Stmt<'a> {
        let node = match &stmt.value {
            StmtNode::Expr(e) => StmtNode::Expr(self.rebuild_expr(e, b)),
            StmtNode::Block(stmts) =>
                StmtNode::Block(stmts.iter().map(|s| self.rebuild_stmt(s, b)).collect()),
            StmtNode::Declare { name, ty, value } => StmtNode::Declare {
                name,
                ty: self.subst_ty(ty, b),
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
            StmtNode::Return(e) => StmtNode::Return(self.rebuild_expr(e, b)),
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
        name_override: Option<&'a str>,
    ) -> TopLevel<'a> {
        let TopLevelNode::Function { name, is_pub, attributes, params, return_type, body, .. } = &tl.value
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
                name: name_override.unwrap_or(name),
                is_pub: *is_pub,
                attributes: attributes.clone(),
                generics: Vec::new(),
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
        let TopLevelNode::Struct { name, is_pub, attributes, fields, .. } = &tl.value
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
        let TopLevelNode::Enum { name, is_pub, attributes, variants, .. } = &tl.value
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
pub fn monomorphize<'a>(program: &[TopLevel<'a>], defs: &mut Defs<'a>, arena: &'a Bump)
    -> Result<Vec<TopLevel<'a>>, Error>
{
    let templates: HashMap<&'a str, &TopLevel<'a>> = program.iter()
        .filter_map(|tl| match &tl.value {
            TopLevelNode::Function { name, generics, .. } if !generics.is_empty() =>
                Some((*name, tl)),
            _ => None,
        })
        .collect();

    let struct_templates: HashMap<&'a str, &TopLevel<'a>> = program.iter()
        .filter_map(|tl| match &tl.value {
            TopLevelNode::Struct { name, generics, .. } if !generics.is_empty() =>
                Some((*name, tl)),
            _ => None,
        })
        .collect();

    let enum_templates: HashMap<&'a str, &TopLevel<'a>> = program.iter()
        .filter_map(|tl| match &tl.value {
            TopLevelNode::Enum { name, generics, .. } if !generics.is_empty() =>
                Some((*name, tl)),
            _ => None,
        })
        .collect();

    let mut m = Mono {
        arena, templates, struct_templates, enum_templates,
        queue: VecDeque::new(), seen: HashMap::new(),
        struct_queue: VecDeque::new(), struct_seen: HashMap::new(),
        enum_queue: VecDeque::new(), enum_seen: HashMap::new(),
        cur_span: Span::unknown(),
        defs,
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
            TopLevelNode::Extern { .. } | TopLevelNode::Global { .. } => {}
            // traits emit no code; they're dropped from the monomorphized output.
            TopLevelNode::Trait { .. } => {}
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before mono"),
        }
    }

    // build each requested instantiation. instances can request more, so drain
    // till the queue is empty. group by template so they emit next to it.
    // NOTE: the depth/count guards below fire after pop, i.e. after this item was
    // already built+queued - so we overshoot the cutoff by one instance. fine as
    // a backstop, just not a tight bound.
    let mut instances: HashMap<&'a str, Vec<TopLevel<'a>>> = HashMap::new();
    let mut materialized = 0usize;
    while let Some(inst) = m.queue.pop_front() {
        materialized += 1;
        if let Some(deep) = inst.args.iter().find_map(|a| match a {
            ConcreteArg::Type(t) if type_depth(t) > TYPE_DEPTH_LIMIT => Some(t),
            _ => None,
        }) {
            return Err(Error {
                msg: format!(
                    "monomorphization of '{}' produced a type argument nested deeper \
                     than {} (`{}`); this usually means unbounded generic recursion \
                     (a generic function calling itself at an ever-growing type)",
                    inst.base, TYPE_DEPTH_LIMIT, deep,
                ),
                span: inst.span,
            });
        }
        if materialized > INSTANTIATION_LIMIT {
            return Err(Error {
                msg: format!(
                    "monomorphization exceeded {} instantiations at call to '{}'",
                    INSTANTIATION_LIMIT, inst.base,
                ),
                span: inst.span,
            });
        }
        let tl = m.templates[inst.base];
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
        let f = m.rebuild_function(tl, &bindings, Some(inst.mangled));
        instances.entry(inst.base).or_default().push(f);
    }

    // build each requested generic-struct/enum instance. substituting a field or
    // payload type can request further instances of EITHER kind (`Box<Option<i32>>`
    // needs `Option$i32`; `Option<Buf<i32,4>>` needs `Buf$i32$4`), so drain BOTH
    // queues together to a fixpoint - whichever has a pending item goes next. the
    // same depth/count guards backstop growing recursion (`struct Bad<T> { p:
    // *Bad<*T> }`). fn instances above already seeded these queues via their
    // param/return/body types.
    let mut struct_instances: HashMap<&'a str, Vec<TopLevel<'a>>> = HashMap::new();
    let mut enum_instances: HashMap<&'a str, Vec<TopLevel<'a>>> = HashMap::new();
    let mut materialized = 0usize;
    loop {
        if let Some(inst) = m.struct_queue.pop_front() {
            materialized += 1;
            if let Some(deep) = inst.args.iter().find_map(|a| match a {
                ConcreteArg::Type(t) if type_depth(t) > TYPE_DEPTH_LIMIT => Some(t),
                _ => None,
            }) {
                return Err(Error {
                    msg: format!(
                        "monomorphization of struct '{}' produced a type argument nested \
                         deeper than {} (`{}`); this usually means an unbounded generic \
                         struct (one whose field mentions itself at an ever-growing type)",
                        inst.base, TYPE_DEPTH_LIMIT, deep,
                    ),
                    span: inst.span,
                });
            }
            if materialized > INSTANTIATION_LIMIT {
                return Err(Error {
                    msg: format!(
                        "monomorphization exceeded {} struct/enum instantiations at '{}'",
                        INSTANTIATION_LIMIT, inst.base,
                    ),
                    span: inst.span,
                });
            }
            let tl = m.struct_templates[inst.base];
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
            materialized += 1;
            if let Some(deep) = inst.args.iter().find_map(|a| match a {
                ConcreteArg::Type(t) if type_depth(t) > TYPE_DEPTH_LIMIT => Some(t),
                _ => None,
            }) {
                return Err(Error {
                    msg: format!(
                        "monomorphization of enum '{}' produced a type argument nested \
                         deeper than {} (`{}`); this usually means an unbounded generic \
                         enum (one whose payload mentions itself at an ever-growing type)",
                        inst.base, TYPE_DEPTH_LIMIT, deep,
                    ),
                    span: inst.span,
                });
            }
            if materialized > INSTANTIATION_LIMIT {
                return Err(Error {
                    msg: format!(
                        "monomorphization exceeded {} struct/enum instantiations at '{}'",
                        INSTANTIATION_LIMIT, inst.base,
                    ),
                    span: inst.span,
                });
            }
            let tl = m.enum_templates[inst.base];
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
            TopLevelNode::Function { name, generics, .. } if !generics.is_empty() => {
                if let Some(insts) = instances.remove(name) {
                    output.extend(insts);
                }
            }
            TopLevelNode::Function { .. } => output.push(concrete.remove(&i).unwrap()),
            // a generic struct template drops out (its `Param` fields never lay
            // out), replaced by its concrete instances - emitted here, next to it.
            TopLevelNode::Struct { name, generics, .. } if !generics.is_empty() => {
                if let Some(insts) = struct_instances.remove(name) {
                    output.extend(insts);
                }
            }
            // same story for a generic enum template: dropped, replaced by its
            // concrete instances (each a plain, fully-substituted data enum - the
            // typecheck/mil/backend pipeline handles it exactly like a hand-written
            // one, per the field-less/data-enum machinery already in place).
            TopLevelNode::Enum { name, generics, .. } if !generics.is_empty() => {
                if let Some(insts) = enum_instances.remove(name) {
                    output.extend(insts);
                }
            }
            // a non-generic struct/enum: emit the rebuilt version, whose concrete
            // generic-instance field types were collapsed and requested up front.
            TopLevelNode::Struct { .. } | TopLevelNode::Enum { .. } =>
                output.push(concrete_aggregates.remove(&i).unwrap()),
            TopLevelNode::Extern { .. } | TopLevelNode::Global { .. } => output.push(tl.clone()),
            // traits emit no code and are not carried into the concrete program.
            TopLevelNode::Trait { .. } => {}
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before mono"),
        }
    }

    Ok(output)
}
