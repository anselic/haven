use std::collections::HashMap;
use haven_common::ast::*;
use crate::intrinsics::{Intrinsic, IntrinsicSig, TyConstraint, ConstBound};
use super::context::{Context, GenericFnSig, param_names};
use super::infer::{check_expr, infer};

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
        return Err(Error::new(span.clone(), format!("{}(): {}", intrinsic, msg)));
    }
    match kind {
        TyConstraint::Any => {}
        TyConstraint::Numeric => {
            if !ty.is_numeric() && !matches!(ty, Type::Param(_)) {
                return Err(Error::new(span.clone(), format!(
                    "{}() expects a numeric type, got `{}`", intrinsic, cx.show(&ty))));
            }
        }
        TyConstraint::Pointer => {
            // `str` is a raw `const char*` - a single machine pointer - so it is a
            // valid pointer type for `null`/`ptr_cast` (e.g. a null C string, or
            // casting a `*u8` to `str` and back).
            if !matches!(ty, Type::Pointer(_) | Type::Param(_) | Type::Str) {
                return Err(Error::new(span.clone(), format!(
                    "{}() expects a pointer type, got `{}`", intrinsic, cx.show(&ty))));
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
        Err(Error::new(span.clone(), msg))
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
        return Err(Error::new(span.clone(), format!(
            "{}() expects {} type argument{} in `::<...>`, got {}",
            intrinsic, n_type + n_const,
            if n_type + n_const == 1 { "" } else { "s" }, type_args.len(),
        )));
    }
    if args.len() != sig.value_arity {
        return Err(Error::new(span.clone(), format!(
            "{}() takes exactly {} argument{}, got {}",
            intrinsic, sig.value_arity,
            if sig.value_arity == 1 { "" } else { "s" }, args.len(),
        )));
    }

    let mut tys = Vec::with_capacity(n_type);
    for (i, kind) in sig.type_params.iter().enumerate() {
        match &type_args[i] {
            GenericArg::Type(ty) => tys.push(check_type_arg(cx, intrinsic, *kind, ty, span)?),
            GenericArg::Const(_) => return Err(Error::new(span.clone(), format!(
                "{}() expects a type for type argument {}, got a const", intrinsic, i + 1))),
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
            GenericArg::Type(_) => return Err(Error::new(span.clone(), format!(
                "{}() expects a const for type argument {}, got a type", intrinsic, n_type + j + 1))),
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

/// Bind turbofish arguments to generic parameters, positionally, and verify each
/// bounded parameter's argument implements the traits it was declared to.
///
/// `generics` and `type_args` must already be the same length — the arity error
/// belongs to the caller, which knows whether it is checking a whole call
/// (`f::<A, B>()`) or only the tail a method declares for itself, after
/// unification has bound its `extend` block's share.
pub(crate) fn bind_turbofish<'a>(
    cx: &Context<'a>,
    name: &str,
    generics: &[GenericParam<'a>],
    type_args: &[GenericArg<'a>],
    span: &Span,
) -> Result<(HashMap<&'a str, Type<'a>>, HashMap<&'a str, ConstVal<'a>>), Error> {
    let mut type_bindings: HashMap<&'a str, Type<'a>> = HashMap::new();
    let mut const_bindings: HashMap<&'a str, ConstVal<'a>> = HashMap::new();
    for (gp, ta) in generics.iter().zip(type_args) {
        match (gp, ta) {
            (GenericParam::Type { name: pname, .. }, GenericArg::Type(ty)) => {
                // resolve against the caller's own type params (a generic body
                // can forward its `T`), then check any structs exist.
                let ty = ty.clone();
                if let Err(msg) = check_type_resolves(cx, &ty) {
                    return Err(Error::new(span.clone(), format!("{}(): {}", name, msg)));
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
                    return Err(Error::new(span.clone(), format!(
                        "{}(): unknown const parameter '{}'", name, fwd)));
                }
                const_bindings.insert(pname, ConstVal::Param(fwd));
            }
            (GenericParam::Type { name: pname, .. }, GenericArg::Const(_)) => return Err(Error::new(span.clone(), format!("{}(): expected a type argument for '{}', got a const value", name, pname))),
            (GenericParam::Const(pname, _), GenericArg::Type(ty)) => {
                // a bare-ident turbofish arg (`N`) parses as a type; if it names an
                // in-scope const param it's a forward (bind symbolically, as above),
                // otherwise it's a real type wrongly placed in a const slot.
                match const_param_name(ty).filter(|n| cx.const_generics.contains(n)) {
                    Some(fwd) => { const_bindings.insert(pname, ConstVal::Param(fwd)); }
                    None => return Err(Error::new(span.clone(), format!(
                        "{}(): expected a const argument for '{}', got a type", name, pname))),
                }
            }
        }
    }

    check_bounds(cx, name, generics, &type_bindings, span)?;
    Ok((type_bindings, const_bindings))
}

/// Verify that each bounded type parameter's binding implements the traits it was
/// declared to. Nominal: satisfaction means an `extend T: Trait` impl exists.
///
/// Shared by the two ways a parameter acquires a binding. A turbofish binds one
/// positionally (`show::<f64>()`), and unifying an `extend` target against a
/// receiver binds one by matching (`p.display()` on a `Pair<f64>`, where
/// `extend Pair<T>: Display where T: Display`). Both have to be checked, and
/// checking them in the same place is what makes the second produce an error
/// about the call rather than one about the template body it would otherwise
/// fail inside - "`f64` does not implement `Display`" instead of "cannot access
/// field 'display' on type f64", pointing at a line the caller never wrote.
///
/// `who` names whatever imposed the bound, for the message: a function, a method,
/// or an `extend` target.
pub(crate) fn check_bounds<'a>(
    cx: &Context<'a>,
    who: &str,
    generics: &[GenericParam<'a>],
    bindings: &HashMap<&'a str, Type<'a>>,
    span: &Span,
) -> Result<(), Error> {
    for gp in generics {
        let GenericParam::Type { name: pname, bounds } = gp else { continue };
        if bounds.is_empty() { continue; }
        let Some(arg_ty) = bindings.get(pname) else { continue };
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
                return Err(Error::new(span.clone(), format!(
                    "type `{}` does not implement trait `{}`", cx.show(arg_ty), bound))
                    .with_label(span.clone(), format!("required by `{}`", who))
                    .with_note(format!("`{}` declares the bound `{}: {}`", who, pname, bound)));
            }
        }
    }
    Ok(())
}

/// Typecheck a call to a user generic function: bind the callee's type params,
/// substitute them into the sig, check the value args, and return the substituted
/// result type. mono materializes the instance later.
///
/// The type params are bound one of two ways. An explicit turbofish
/// (`printf::<str>(s)`) binds them positionally. An omitted one (`printf(s)`) is
/// inferred from the argument types; when that happens the recovered args are
/// returned as `Some(..)` so the caller can stash them for mono, which otherwise
/// sees a bare call with nothing to instantiate.
pub(crate) fn check_generic_call<'a>(
    cx: &mut Context<'a>,
    name: &'a str,
    sig: &GenericFnSig<'a>,
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
    span: &Span,
) -> Result<(Type<'a>, Option<Vec<GenericArg<'a>>>), Error> {
    // value-arg arity is the same both ways, and the inference path relies on
    // args lining up one-to-one with params, so check it once up front.
    if args.len() != sig.params.len() {
        return Err(Error::new(span.clone(), format!("{}() expects {} argument{}, got {}",
            name, sig.params.len(), if sig.params.len() == 1 { "" } else { "s" }, args.len())));
    }

    // Partial turbofish (some but not all args) stays an arity error: inference
    // is all-or-nothing, either every param is written or every param is inferred.
    let (type_bindings, const_bindings, inferred) = if type_args.is_empty() {
        let (tb, cb) = infer_type_args(cx, name, sig, args, span)?;
        let materialized = materialize_targs(name, &sig.generics, &tb, &cb, TargSite::Call, span)?;
        (tb, cb, Some(materialized))
    } else if type_args.len() == sig.generics.len() {
        let (tb, cb) = bind_turbofish(cx, name, &sig.generics, type_args, span)?;
        (tb, cb, None)
    } else {
        return Err(Error::new(span.clone(), format!(
            "{}() expects {} generic argument{} in `::<...>`, got {}",
            name, sig.generics.len(),
            if sig.generics.len() == 1 { "" } else { "s" }, type_args.len(),
        )));
    };

    let params: Vec<Type<'a>> = sig.params.iter()
        .map(|p| subst_param_type(&type_bindings, &const_bindings, p)).collect();
    let return_type = subst_param_type(&type_bindings, &const_bindings, &sig.return_type);

    for (param_ty, arg) in params.iter().zip(args) {
        check_expr(cx, param_ty, arg)?;
    }

    Ok((return_type, inferred))
}

/// Recover the callee's type-param bindings from the argument types, for a call
/// written without a turbofish (`printf(x)` rather than `printf::<T>(x)`). Each
/// declared parameter type is a pattern over the callee's generics; unifying it
/// against the inferred argument type binds whatever generics appear in it, so
/// `arg: T` against an `str` argument binds `T = str`.
///
/// Unification is best-effort per argument: a param that shares no structure with
/// its argument simply binds nothing (a generic that appears in *no* parameter is
/// then caught by `materialize_targs`), and a structural mismatch on a param that
/// *does* mention a generic is left for the `check_expr` pass to report against
/// the substituted type, where the message names the concrete types rather than a
/// bare "could not unify".
fn infer_type_args<'a>(
    cx: &mut Context<'a>,
    name: &str,
    sig: &GenericFnSig<'a>,
    args: &[Expr<'a>],
    span: &Span,
) -> Result<(HashMap<&'a str, Type<'a>>, HashMap<&'a str, ConstVal<'a>>), Error> {
    // the free names unify may bind are exactly the callee's own generics.
    let params = param_names(&sig.generics);
    let mut u = Unified::default();
    for (param_ty, arg) in sig.params.iter().zip(args) {
        let arg_ty = infer(cx, arg)?;
        unify(param_ty, &arg_ty, &params, &mut u);
    }
    // check bounds on whatever we managed to bind; an unbound param is reported
    // by `materialize_targs`, and `check_bounds` skips it in the meantime.
    check_bounds(cx, name, &sig.generics, &u.types, span)?;
    Ok((u.types, u.consts))
}

/// Recover a generic struct's type arguments from the values its literal gives
/// its fields, for a literal written without a turbofish (`Serial { a: x, b: y }`
/// rather than `Serial::<A, B> { .. }`).
///
/// The struct-literal counterpart of [`infer_type_args`], and deliberately the
/// same shape: each declared field type is a pattern over the struct's params,
/// and unifying it against the inferred field-value type binds whatever params
/// appear in it. `a: A` against a `Gain` value binds `A = Gain`.
///
/// Best-effort per field, for the same reason: a field sharing no structure with
/// its value binds nothing and is reported by [`materialize_targs`], while a
/// structural mismatch is left to the `check_expr` pass below, whose message
/// names the concrete types.
///
/// Fields are matched positionally against the declaration, which is what the
/// caller enforces anyway - a literal must list every field in order - so a
/// literal with the wrong arity or a misspelled field simply binds less and
/// falls through to that check.
pub(crate) fn infer_struct_type_args<'a>(
    cx: &mut Context<'a>,
    name: &str,
    params: &[GenericParam<'a>],
    def: &[(&'a str, Type<'a>)],
    fields: &[(&'a str, Expr<'a>)],
    span: &Span,
) -> Result<Vec<GenericArg<'a>>, Error> {
    let pnames = param_names(params);
    let mut u = Unified::default();
    for ((def_name, def_ty), (lit_name, lit_value)) in def.iter().zip(fields) {
        if def_name != lit_name { break; }
        // a field whose type mentions no parameter can teach us nothing, and
        // inferring its value would only risk a spurious error ahead of the real
        // per-field check.
        if !mentions_param(def_ty, &pnames) { continue; }
        let Ok(arg_ty) = infer(cx, lit_value) else { continue };
        unify(def_ty, &arg_ty, &pnames, &mut u);
    }
    check_bounds(cx, name, params, &u.types, span)?;
    materialize_targs(name, params, &u.types, &u.consts, TargSite::StructLit, span)
}

/// Whether `ty` mentions any of `params` anywhere inside it.
fn mentions_param<'a>(ty: &Type<'a>, params: &[&'a str]) -> bool {
    match ty {
        Type::Param(n) => params.contains(n),
        Type::Pointer(t) | Type::Slice(t) => mentions_param(t, params),
        Type::Array(t, n) | Type::Simd(t, n) =>
            mentions_param(t, params)
                || matches!(n, ConstVal::Param(p) if params.contains(p)),
        Type::Named { args, .. } => args.iter().any(|a| match a {
            GenericArg::Type(t) => mentions_param(t, params),
            GenericArg::Const(ConstVal::Param(p)) => params.contains(p),
            GenericArg::Const(_) => false,
        }),
        Type::Function { params: ps, return_type } =>
            ps.iter().any(|t| mentions_param(t, params)) || mentions_param(return_type, params),
        _ => false,
    }
}

/// Which syntax the args were being recovered from, so an "unbound parameter"
/// message can name what the reader actually wrote. The two differ only in
/// wording; a call is inferred from arguments and spelled with parentheses, a
/// literal from field values and spelled with braces.
#[derive(Clone, Copy)]
pub(crate) enum TargSite { Call, StructLit }

impl TargSite {
    /// What failed to determine the parameter.
    fn source(self) -> &'static str {
        match self {
            TargSite::Call => "not determined by these arguments",
            TargSite::StructLit => "not determined by these field values",
        }
    }
    /// How to write the turbofish explicitly instead.
    fn example(self, name: &str) -> String {
        match self {
            TargSite::Call => format!("specify it explicitly, e.g. `{}::<...>(...)`", name),
            TargSite::StructLit =>
                format!("specify it explicitly, e.g. `{}::<...> {{ ... }}`", name),
        }
    }
}

/// Assemble inferred bindings into a positional turbofish, in the callee's
/// declared param order, so monomorphization consumes it exactly as if the user
/// had written `name::<...>`. Errors if any generic went unbound.
fn materialize_targs<'a>(
    name: &str,
    generics: &[GenericParam<'a>],
    types: &HashMap<&'a str, Type<'a>>,
    consts: &HashMap<&'a str, ConstVal<'a>>,
    site: TargSite,
    span: &Span,
) -> Result<Vec<GenericArg<'a>>, Error> {
    let mut out = Vec::with_capacity(generics.len());
    for gp in generics {
        let arg = match gp {
            GenericParam::Type { name: pn, .. } =>
                types.get(pn).cloned().map(GenericArg::Type),
            GenericParam::Const(pn, _) =>
                consts.get(pn).cloned().map(GenericArg::Const),
        };
        match arg {
            Some(a) => out.push(a),
            None => {
                let pn = match gp {
                    GenericParam::Type { name, .. } => *name,
                    GenericParam::Const(name, _) => *name,
                };
                return Err(Error::new(span.clone(),
                    format!("cannot infer type argument `{}` for `{}`", pn, name))
                    .with_label(span.clone(), site.source())
                    .with_note(site.example(name)));
            }
        }
    }
    Ok(out)
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
        return Err(Error::new(span.clone(), format!(
            "struct '{}' expects {} type argument{}, got {}",
            name, params.len(),
            if params.len() == 1 { "" } else { "s" }, args.len(),
        )));
    }
    let mut type_subst: HashMap<&'a str, Type<'a>> = HashMap::new();
    let mut const_subst: HashMap<&'a str, ConstVal<'a>> = HashMap::new();
    let mut resolved: Vec<GenericArg<'a>> = Vec::with_capacity(params.len());
    for (gp, ga) in params.iter().zip(args) {
        match (gp, ga) {
            (GenericParam::Type { name: pname, .. }, GenericArg::Type(ty)) => {
                let ty = ty.clone();
                if let Err(msg) = check_type_resolves(cx, &ty) {
                    return Err(Error::new(span.clone(), format!("struct '{}': {}", name, msg)));
                }
                type_subst.insert(pname, ty.clone());
                resolved.push(GenericArg::Type(ty));
            }
            (GenericParam::Const(pname, _), GenericArg::Const(ConstVal::Lit(v))) => {
                const_subst.insert(pname, ConstVal::Lit(*v));
                resolved.push(GenericArg::Const(ConstVal::Lit(*v)));
            }
            (GenericParam::Const(pname, _), GenericArg::Const(ConstVal::Param(fwd))) => return Err(Error::new(span.clone(), format!("struct '{}': forwarding const parameter '{}' to '{}' is not supported yet", name, fwd, pname))),
            (GenericParam::Type { name: pname, .. }, GenericArg::Const(_)) => return Err(Error::new(span.clone(), format!("struct '{}': expected a type argument for '{}', got a const value", name, pname))),
            (GenericParam::Const(pname, _), GenericArg::Type(ty)) => {
                // a bare-ident arg (`N`) parses as a type; if it names an in-scope
                // const param it's a (currently unsupported) forward, else it's a
                // real type wrongly placed in a const slot.
                if const_param_name(ty).is_some_and(|n| cx.const_generics.contains(&n)) {
                    return Err(Error::new(span.clone(), format!(
                        "struct '{}': forwarding const parameter '{}' to '{}' is not supported yet",
                        name, const_param_name(ty).unwrap(), pname)));
                }
                return Err(Error::new(span.clone(), format!(
                    "struct '{}': expected a const argument for '{}', got a type", name, pname)));
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

/// Like [`subst_self`], but also replaces each associated-type parameter with the
/// implementing type's binding for it. Inside a trait method signature `Self`
/// stands for the implementing type and a `Self::Item` projection was resolved to
/// `Param("Item")`; `assoc` maps each such name (`"Item"`) to the type the impl
/// bound it to. Used to turn a trait signature into the concrete signature the
/// impl must match. A `Named` type's generic arguments are descended into so
/// `Option<Self::Item>` becomes `Option<i32>`.
pub(crate) fn subst_self_assoc<'a>(
    ty: &Type<'a>,
    self_ty: &Type<'a>,
    assoc: &std::collections::HashMap<&'a str, Type<'a>>,
) -> Type<'a> {
    match ty {
        Type::Param(name) if *name == "Self" => self_ty.clone(),
        Type::Param(name) => match assoc.get(name) {
            Some(bound) => bound.clone(),
            None => ty.clone(),
        },
        Type::Pointer(inner)  => Type::Pointer(Box::new(subst_self_assoc(inner, self_ty, assoc))),
        Type::Array(inner, n) => Type::Array(Box::new(subst_self_assoc(inner, self_ty, assoc)), n.clone()),
        Type::Slice(inner)    => Type::Slice(Box::new(subst_self_assoc(inner, self_ty, assoc))),
        Type::Simd(inner, n)  => Type::Simd(Box::new(subst_self_assoc(inner, self_ty, assoc)), n.clone()),
        Type::Function { params, return_type } => Type::Function {
            params: params.iter().map(|p| subst_self_assoc(p, self_ty, assoc)).collect(),
            return_type: Box::new(subst_self_assoc(return_type, self_ty, assoc)),
        },
        Type::Named { def, args } => Type::Named {
            def: *def,
            args: args.iter().map(|a| match a {
                GenericArg::Type(t) => GenericArg::Type(subst_self_assoc(t, self_ty, assoc)),
                other => other.clone(),
            }).collect(),
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
