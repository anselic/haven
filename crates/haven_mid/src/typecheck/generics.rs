use std::collections::HashMap;
use haven_common::ast::*;
use crate::intrinsics::{Intrinsic, IntrinsicSig, TyConstraint, ConstBound};
use super::context::{Context, GenericFnSig};
use super::infer::check_expr;

/// Resolves a turbofish type argument (generic params → `Type::Param`), checks
/// any referenced structs exist, then checks it against the parameter's kind
/// constraint. Type params pass kind checks optimistically - their kind is only
/// known once a generic function is monomorphized (not yet implemented), and
/// such bodies never reach codegen.
fn check_type_arg<'a>(
    cx: &Context<'a>,
    intrinsic: Intrinsic,
    kind: TyConstraint,
    ty: &Type<'a>,
    span: &Span,
) -> Result<Type<'a>, Error> {
    let ty = ty.clone();
    if let Err(msg) = check_type_resolves(cx, &ty) {
        return Err(Error { msg: format!("{}(): {}", intrinsic, msg), span: span.clone() });
    }
    match kind {
        TyConstraint::Any => {}
        TyConstraint::Numeric => {
            if !ty.is_numeric() && !matches!(ty, Type::Param(_)) {
                return Err(Error {
                    msg: format!("{}() expects a numeric type, got `{}`", intrinsic, cx.show(&ty)),
                    span: span.clone(),
                });
            }
        }
        TyConstraint::Pointer => {
            if !matches!(ty, Type::Pointer(_) | Type::Param(_)) {
                return Err(Error {
                    msg: format!("{}() expects a pointer type, got `{}`", intrinsic, cx.show(&ty)),
                    span: span.clone(),
                });
            }
        }
    }
    Ok(ty)
}

/// Checks a turbofish const argument against the parameter's bound.
fn check_const_arg(
    intrinsic: Intrinsic,
    bound: ConstBound,
    value: i64,
    span: &Span,
) -> Result<usize, Error> {
    if value >= bound.min && value <= bound.max && value % bound.multiple_of as i64 == 0 {
        Ok(value as usize)
    } else {
        let mut msg = format!(
            "{}() const argument must be an integer literal in {}..={}",
            intrinsic, bound.min, bound.max,
        );
        if bound.multiple_of != 1 {
            msg += &format!(" and a multiple of {}", bound.multiple_of);
        }
        Err(Error { msg, span: span.clone() })
    }
}

/// Validates the turbofish and value arities of an intrinsic call against its
/// signature, and binds the type/const arguments. The trailing value arguments
/// stay in `args` and are checked by the caller.
pub(crate) fn bind_generics<'a>(
    cx: &Context<'a>,
    intrinsic: Intrinsic,
    sig: &IntrinsicSig,
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
    span: &Span,
) -> Result<(Vec<Type<'a>>, Vec<ConstVal<'a>>), Error> {
    let n_type = sig.type_params.len();
    let n_const = sig.const_params.len();

    if type_args.len() != n_type + n_const {
        return Err(Error {
            msg: format!(
                "{}() expects {} type argument{} in `::<...>`, got {}",
                intrinsic, n_type + n_const,
                if n_type + n_const == 1 { "" } else { "s" }, type_args.len(),
            ),
            span: span.clone(),
        });
    }
    if args.len() != sig.value_arity {
        return Err(Error {
            msg: format!(
                "{}() takes exactly {} argument{}, got {}",
                intrinsic, sig.value_arity,
                if sig.value_arity == 1 { "" } else { "s" }, args.len(),
            ),
            span: span.clone(),
        });
    }

    let mut tys = Vec::with_capacity(n_type);
    for (i, kind) in sig.type_params.iter().enumerate() {
        match &type_args[i] {
            GenericArg::Type(ty) => tys.push(check_type_arg(cx, intrinsic, *kind, ty, span)?),
            GenericArg::Const(_) => return Err(Error {
                msg: format!("{}() expects a type for type argument {}, got a const", intrinsic, i + 1),
                span: span.clone(),
            }),
        }
    }
    let mut consts = Vec::with_capacity(n_const);
    for (j, bound) in sig.const_params.iter().enumerate() {
        let cv = match &type_args[n_type + j] {
            GenericArg::Const(ConstVal::Lit(n)) => {
                check_const_arg(intrinsic, *bound, *n as i64, span)?;
                ConstVal::Lit(*n)
            }
            // already-symbolic (not produced by the parser today, but kept total)
            GenericArg::Const(ConstVal::Param(name)) => ConstVal::Param(name),
            // a bare ident forwarded from an enclosing `const N` parses as a type;
            // accept it as a symbolic const and leave the range check to the
            // post-mono re-typecheck, when the value is a concrete literal.
            GenericArg::Type(ty) if const_param_name(ty)
                .is_some_and(|n| cx.const_generics.contains(&n)) =>
            {
                ConstVal::Param(const_param_name(ty).unwrap())
            }
            GenericArg::Type(_) => return Err(Error {
                msg: format!("{}() expects a const for type argument {}, got a type", intrinsic, n_type + j + 1),
                span: span.clone(),
            }),
        };
        consts.push(cv);
    }
    Ok((tys, consts))
}

/// If `ty` is a bare identifier (`Type::Struct` before resolution, or `Type::Param`
/// after), returns that name - the two shapes a forwarded const generic parameter
/// can take in a turbofish argument. Compound types are never const params.
fn const_param_name<'a>(ty: &Type<'a>) -> Option<&'a str> {
    match ty {
        // a forwarded const param arrives as a bare ident, which name resolution
        // could not tell from a type parameter and so resolved to `Param`.
        Type::Param(name) => Some(name),
        _ => None,
    }
}

/// Substitute `Type::Param(name)` with its concrete binding, recursing through
/// compound types. inverse of `resolve_type`: used when a generic sig (which
/// holds `Param`s) is specialized at a call site.
// TODO: this, resolve_type, and mono.rs::subst_ty are three near-identical walks
// over the same compound-type arms, so maybe in the future it could be generalized
// into a single `Type::walk_mut` or `Type::map` function that takes a closure to
// apply to each leaf type
pub(crate) fn subst_param_type<'a>(
    types: &HashMap<&'a str, Type<'a>>,
    consts: &HashMap<&'a str, ConstVal<'a>>,
    ty: &Type<'a>,
) -> Type<'a> {
    // a const param binds to either a concrete literal or - when it was forwarded
    // from an enclosing generic's own `const` param - another symbolic `Param`,
    // which mono resolves once the outer instance is specialized.
    let sub_cv = |cv: &ConstVal<'a>| match cv {
        ConstVal::Param(n) => consts.get(n).cloned().unwrap_or_else(|| cv.clone()),
        ConstVal::Lit(_) => cv.clone(),
    };
    match ty {
        Type::Param(name) => types.get(name).cloned().unwrap_or_else(|| ty.clone()),
        // a generic instance can mention params in its args (`Option<T>`,
        // `Buf<T, N>`); recurse so both type and const args get substituted.
        // Structs and enums needed separate copies of this when they were
        // separate variants; now the identity is opaque and one arm does both.
        Type::Named { def, args } => Type::Named {
            def: *def,
            args: args.iter().map(|a| match a {
                GenericArg::Type(t) => GenericArg::Type(subst_param_type(types, consts, t)),
                GenericArg::Const(cv) => GenericArg::Const(sub_cv(cv)),
            }).collect(),
        },
        Type::Pointer(inner)  => Type::Pointer(Box::new(subst_param_type(types, consts, inner))),
        Type::Array(inner, n) => Type::Array(Box::new(subst_param_type(types, consts, inner)), sub_cv(n)),
        Type::Slice(inner)    => Type::Slice(Box::new(subst_param_type(types, consts, inner))),
        Type::Simd(inner, n)  => Type::Simd(Box::new(subst_param_type(types, consts, inner)), sub_cv(n)),
        Type::Function { params, return_type } => Type::Function {
            params: params.iter().map(|p| subst_param_type(types, consts, p)).collect(),
            return_type: Box::new(subst_param_type(types, consts, return_type)),
        },
        other => other.clone(),
    }
}

/// Typecheck a call to a user generic function: bind the turbofish type args to
/// the callee's type params, substitute them into the sig, check the value args,
/// and return the substituted result type. mono materializes the instance later.
pub(crate) fn check_generic_call<'a>(
    cx: &mut Context<'a>,
    name: &'a str,
    sig: &GenericFnSig<'a>,
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
    span: &Span,
) -> Result<Type<'a>, Error> {
    if type_args.len() != sig.generics.len() {
        return Err(Error {
            msg: format!(
                "{}() expects {} generic argument{} in `::<...>`, got {}",
                name, sig.generics.len(),
                if sig.generics.len() == 1 { "" } else { "s" }, type_args.len(),
            ),
            span: span.clone(),
        });
    }

    // bind each turbofish arg to its generic param, matched positionally.
    let mut type_bindings: HashMap<&'a str, Type<'a>> = HashMap::new();
    let mut const_bindings: HashMap<&'a str, ConstVal<'a>> = HashMap::new();
    for (gp, ta) in sig.generics.iter().zip(type_args) {
        match (gp, ta) {
            (GenericParam::Type { name: pname, .. }, GenericArg::Type(ty)) => {
                // resolve against the caller's own type params (a generic body
                // can forward its `T`), then check any structs exist.
                let ty = ty.clone();
                if let Err(msg) = check_type_resolves(cx, &ty) {
                    return Err(Error { msg: format!("{}(): {}", name, msg), span: span.clone() });
                }
                type_bindings.insert(pname, ty);
            }
            (GenericParam::Const(pname, _), GenericArg::Const(ConstVal::Lit(v))) => {
                const_bindings.insert(pname, ConstVal::Lit(*v));
            }
            // forwarding an enclosing generic's own `const` param by name: bind it
            // symbolically. `subst_param_type` keeps the `Param` in the result type,
            // and mono resolves it to a literal when the outer proc is specialized
            // (the forwarded name is always concrete by then). Range/kind bounds are
            // re-checked post-mono, exactly as for the intrinsic turbofish path.
            (GenericParam::Const(pname, _), GenericArg::Const(ConstVal::Param(fwd))) => {
                if !cx.const_generics.contains(fwd) {
                    return Err(Error {
                        msg: format!("{}(): unknown const parameter '{}'", name, fwd),
                        span: span.clone(),
                    });
                }
                const_bindings.insert(pname, ConstVal::Param(fwd));
            }
            (GenericParam::Type { name: pname, .. }, GenericArg::Const(_)) => return Err(Error {
                msg: format!("{}(): expected a type argument for '{}', got a const value", name, pname),
                span: span.clone(),
            }),
            (GenericParam::Const(pname, _), GenericArg::Type(ty)) => {
                // a bare-ident turbofish arg (`N`) parses as a type; if it names an
                // in-scope const param it's a forward (bind symbolically, as above),
                // otherwise it's a real type wrongly placed in a const slot.
                match const_param_name(ty).filter(|n| cx.const_generics.contains(n)) {
                    Some(fwd) => { const_bindings.insert(pname, ConstVal::Param(fwd)); }
                    None => return Err(Error {
                        msg: format!("{}(): expected a const argument for '{}', got a type", name, pname),
                        span: span.clone(),
                    }),
                }
            }
        }
    }

    // verify trait bounds: each bounded type param's concrete argument must
    // implement the required traits (nominal, via an `extend T: Trait` impl). A
    // forwarded type param satisfies a bound iff it carries the same bound in the
    // enclosing generic (bound propagation); a concrete struct/enum must have a
    // matching recorded impl; anything else can't satisfy a bound.
    for gp in &sig.generics {
        if let GenericParam::Type { name: pname, bounds } = gp {
            if bounds.is_empty() { continue; }
            let Some(arg_ty) = type_bindings.get(pname) else { continue };
            for bound in bounds {
                let ok = match arg_ty {
                    // a forwarded type param satisfies a bound only by carrying
                    // it; there is no impl to consult until it is substituted.
                    Type::Param(fp) =>
                        cx.generic_bounds.get(fp).is_some_and(|bs| bs.contains(&bound.def)),
                    // anything with a head can be covered by an impl - which now
                    // includes primitives and structural types, so `[T]` may
                    // satisfy a `Display` bound just as a struct does.
                    concrete => cx.implements(concrete, bound.def),
                };
                if !ok {
                    return Err(Error {
                        msg: format!(
                            "type `{}` does not implement trait `{}`, required by `{}`'s bound `{}: {}`",
                            cx.show(arg_ty), bound, name, pname, bound),
                        span: span.clone(),
                    });
                }
            }
        }
    }

    let params: Vec<Type<'a>> = sig.params.iter()
        .map(|p| subst_param_type(&type_bindings, &const_bindings, p)).collect();
    let return_type = subst_param_type(&type_bindings, &const_bindings, &sig.return_type);

    if args.len() != params.len() {
        return Err(Error {
            msg: format!("{}() expects {} argument{}, got {}",
                name, params.len(), if params.len() == 1 { "" } else { "s" }, args.len()),
            span: span.clone(),
        });
    }
    for (param_ty, arg) in params.iter().zip(args) {
        check_expr(cx, param_ty, arg)?;
    }

    Ok(return_type)
}

/// Bind a generic struct's applied args to its declared params (`Buf<i32, 8>` ->
/// `{T: i32}`, `{N: 8}`), validating arity and each arg's kind. Type args are
/// resolved against the enclosing function's own generics and checked to exist;
/// const args must be concrete literals. Returns the type/const substitutions
/// plus the resolved args (for the resulting `Type::Struct`). Mirrors
/// `check_generic_call`'s binding, for the struct case.
pub(crate) fn bind_struct_generics<'a>(
    cx: &Context<'a>,
    name: &str,
    params: &[GenericParam<'a>],
    args: &[GenericArg<'a>],
    span: &Span,
) -> Result<(HashMap<&'a str, Type<'a>>, HashMap<&'a str, ConstVal<'a>>, Vec<GenericArg<'a>>), Error> {
    if args.len() != params.len() {
        return Err(Error {
            msg: format!(
                "struct '{}' expects {} type argument{}, got {}",
                name, params.len(),
                if params.len() == 1 { "" } else { "s" }, args.len(),
            ),
            span: span.clone(),
        });
    }
    let mut type_subst: HashMap<&'a str, Type<'a>> = HashMap::new();
    let mut const_subst: HashMap<&'a str, ConstVal<'a>> = HashMap::new();
    let mut resolved: Vec<GenericArg<'a>> = Vec::with_capacity(params.len());
    for (gp, ga) in params.iter().zip(args) {
        match (gp, ga) {
            (GenericParam::Type { name: pname, .. }, GenericArg::Type(ty)) => {
                let ty = ty.clone();
                if let Err(msg) = check_type_resolves(cx, &ty) {
                    return Err(Error { msg: format!("struct '{}': {}", name, msg), span: span.clone() });
                }
                type_subst.insert(pname, ty.clone());
                resolved.push(GenericArg::Type(ty));
            }
            (GenericParam::Const(pname, _), GenericArg::Const(ConstVal::Lit(v))) => {
                const_subst.insert(pname, ConstVal::Lit(*v));
                resolved.push(GenericArg::Const(ConstVal::Lit(*v)));
            }
            (GenericParam::Const(pname, _), GenericArg::Const(ConstVal::Param(fwd))) => return Err(Error {
                msg: format!("struct '{}': forwarding const parameter '{}' to '{}' is not supported yet", name, fwd, pname),
                span: span.clone(),
            }),
            (GenericParam::Type { name: pname, .. }, GenericArg::Const(_)) => return Err(Error {
                msg: format!("struct '{}': expected a type argument for '{}', got a const value", name, pname),
                span: span.clone(),
            }),
            (GenericParam::Const(pname, _), GenericArg::Type(ty)) => {
                // a bare-ident arg (`N`) parses as a type; if it names an in-scope
                // const param it's a (currently unsupported) forward, else it's a
                // real type wrongly placed in a const slot.
                if const_param_name(ty).is_some_and(|n| cx.const_generics.contains(&n)) {
                    return Err(Error {
                        msg: format!("struct '{}': forwarding const parameter '{}' to '{}' is not supported yet",
                            name, const_param_name(ty).unwrap(), pname),
                        span: span.clone(),
                    });
                }
                return Err(Error {
                    msg: format!("struct '{}': expected a const argument for '{}', got a type", name, pname),
                    span: span.clone(),
                });
            }
        }
    }
    Ok((type_subst, const_subst, resolved))
}

/// Substitute the special `Self` type (which name resolution leaves as
/// `Type::Param("Self")`, it having no definition of its own) with `self_ty`,
/// recursing through compound
/// types. Used to specialize a trait method signature to a concrete implementing
/// type (conformance) or to a bounded type param (bounded call resolution).
pub(crate) fn subst_self<'a>(ty: &Type<'a>, self_ty: &Type<'a>) -> Type<'a> {
    match ty {
        Type::Param(name) if *name == "Self" => self_ty.clone(),
        Type::Pointer(inner)  => Type::Pointer(Box::new(subst_self(inner, self_ty))),
        Type::Array(inner, n) => Type::Array(Box::new(subst_self(inner, self_ty)), n.clone()),
        Type::Slice(inner)    => Type::Slice(Box::new(subst_self(inner, self_ty))),
        Type::Simd(inner, n)  => Type::Simd(Box::new(subst_self(inner, self_ty)), n.clone()),
        Type::Function { params, return_type } => Type::Function {
            params: params.iter().map(|p| subst_self(p, self_ty)).collect(),
            return_type: Box::new(subst_self(return_type, self_ty)),
        },
        other => other.clone(),
    }
}

/// Verifies that every `ConstVal::Param` in `ty` names a const generic parameter
/// in `in_scope`. Ignores type params/structs entirely - those are handled by
/// `resolve_type`/`check_type_resolves` - so it can run on raw (unresolved) types.
pub(crate) fn check_const_scope<'a>(in_scope: &[&'a str], ty: &Type<'a>) -> Result<(), String> {
    fn check_cv<'a>(in_scope: &[&'a str], cv: &ConstVal<'a>) -> Result<(), String> {
        match cv {
            ConstVal::Param(n) if !in_scope.contains(n) =>
                Err(format!("unknown const parameter '{}'", n)),
            _ => Ok(()),
        }
    }
    match ty {
        Type::Array(inner, n) | Type::Simd(inner, n) => {
            check_cv(in_scope, n)?;
            check_const_scope(in_scope, inner)
        }
        Type::Pointer(inner) | Type::Slice(inner) => check_const_scope(in_scope, inner),
        Type::Function { params, return_type } => {
            for p in params { check_const_scope(in_scope, p)?; }
            check_const_scope(in_scope, return_type)
        }
        _ => Ok(()),
    }
}

/// Recursively verifies that every named type in `ty` exists, and that its
/// generic arguments match the declaration in count and kind.
///
/// Structs and enums used to need two near-identical copies of this, because
/// they were separate `Type` variants carrying separate name tables. A resolved
/// named type no longer says which it is - that is a property of its definition
/// - so one walk covers both, and the only thing the two cases still disagree
/// about is the word to use in the message.
pub(crate) fn check_type_resolves<'a>(cx: &Context<'a>, ty: &Type<'a>) -> Result<(), String> {
    match ty {
        Type::Named { def, args } => {
            let is_enum = cx.enums.contains_key(def);
            if !is_enum && !cx.types.contains_key(def) {
                return Err(format!("unknown type '{}'", cx.name_of(*def)));
            }
            // recurse into type arguments (`Option<Unknown>` must still error).
            for a in args {
                if let GenericArg::Type(t) = a { check_type_resolves(cx, t)?; }
            }
            let kind = if is_enum { "enum" } else { "struct" };
            let name = cx.name_of(*def);
            // validate the applied argument count against the declared arity:
            // 0 for a non-generic type, `params.len()` for a generic one.
            let params = if is_enum { cx.generic_enums.get(def) } else { cx.generic_structs.get(def) };
            let arity = params.map_or(0, |p| p.len());
            if args.len() != arity {
                return Err(if arity == 0 {
                    format!("{} '{}' is not generic; no type arguments expected", kind, name)
                } else {
                    format!(
                        "{} '{}' expects {} type argument{}, got {}",
                        kind, name, arity,
                        if arity == 1 { "" } else { "s" }, args.len(),
                    )
                });
            }
            // each applied arg's kind must match its param (type vs const);
            // forwarding a const param into a named type isn't supported yet.
            if let Some(params) = params {
                for (gp, ga) in params.iter().zip(args) {
                    match (gp, ga) {
                        (GenericParam::Type { .. }, GenericArg::Type(_)) => {}
                        (GenericParam::Const(_, _), GenericArg::Const(ConstVal::Lit(_))) => {}
                        (GenericParam::Const(pn, _), GenericArg::Const(ConstVal::Param(f))) =>
                            return Err(format!("{} '{}': forwarding const parameter '{}' to '{}' is not supported yet", kind, name, f, pn)),
                        (GenericParam::Type { name: pn, .. }, GenericArg::Const(_)) =>
                            return Err(format!("{} '{}': expected a type argument for '{}', got a const value", kind, name, pn)),
                        (GenericParam::Const(pn, _), GenericArg::Type(t)) => {
                            if const_param_name(t).is_some_and(|n| cx.const_generics.contains(&n)) {
                                return Err(format!("{} '{}': forwarding const parameter '{}' to '{}' is not supported yet", kind, name, const_param_name(t).unwrap(), pn));
                            }
                            return Err(format!("{} '{}': expected a const argument for '{}', got a type", kind, name, pn));
                        }
                    }
                }
            }
            // A concrete generic use (`Option<i32>`, `Buf<i32, 8>`) is valid:
            // monomorphization rewrites it to a flat instance before any later
            // stage.
            Ok(())
        }
        Type::Pointer(inner)
        | Type::Array(inner, _)
        | Type::Slice(inner)
        | Type::Simd(inner, _) => check_type_resolves(cx, inner),
        Type::Function { params, return_type } => {
            for p in params { check_type_resolves(cx, p)?; }
            check_type_resolves(cx, return_type)
        }
        _ => Ok(()),
    }
}
