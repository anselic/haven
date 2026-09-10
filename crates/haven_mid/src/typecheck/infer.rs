use std::collections::{HashMap, HashSet};
use haven_common::ast::*;
use crate::intrinsics::Intrinsic;
use haven_common::defs::{DefId, Member};
use super::context::{Context, MethodCall, RecvAdjust};
use super::generics::{bind_generics, bind_turbofish, check_bounds, subst_param_type, check_generic_call, bind_struct_generics, check_const_scope, subst_self, infer_struct_type_args, infer_enum_type_args, TargSite, ResolvedGenerics};
use super::enums::{enum_variant, split_enum_variant, enum_variant_ctor, check_variant_pattern};

/// A method's mangled name, parameter types, and return type for one call site.
type MethodSig<'a> = (&'a str, Vec<Type<'a>>, Type<'a>);


/// The `extend` method `field` a receiver of type `base_ty` dispatches to, with
/// the bindings that specialize the impl to it.
///
/// The receiver is tried as written first, then through one level of pointer, so
/// `p.area()` on a `*Point` still finds `extend Point`'s method — while an
/// `extend *T` block, which the head machinery now makes expressible, gets first
/// refusal on a pointer receiver.
pub(crate) fn receiver_member<'a>(cx: &Context<'a>, base_ty: &Type<'a>, field: &str)
    -> Option<(Member<'a>, Unified<'a>)>
{
    let direct = cx.member_for(base_ty, field);
    let found = match (direct, base_ty) {
        (None, Type::Pointer(inner)) => cx.member_for(inner, field),
        (found, _) => found,
    };
    found.map(|(m, u)| (m.clone(), u))
}

/// Specialize a member signature for one call site. Receiver matching supplies
/// the `extend` block's arguments; the call's turbofish supplies any method
/// arguments. The desugared signature stores those groups in that order.
fn method_signature<'a>(
    cx: &Context<'a>,
    m: &Member<'a>,
    u: &Unified<'a>,
    field: &str,
    type_args: &[GenericArg<'a>],
    span: &Span,
) -> Result<Option<MethodSig<'a>>, Error> {
    // the impl's parameters were bound by unifying its target against the
    // receiver, so its `where` clause is checked here rather than by
    // `bind_turbofish` - nothing was written at the call site for that to look at.
    check_bounds(cx, &cx.show(&m.self_ty), &m.generics, &u.types, span)?;

    // a method of a *concrete* `extend` that declares no generics of its own is
    // an ordinary function, already in final form in the value scope.
    let Some(sig) = cx.generic_fns.get(m.name) else {
        if !type_args.is_empty() {
            return Err(Error::new(*span, format!("method '{}' takes no generic arguments", field)));
        }
        return Ok(match cx.lookup(m.name) {
            Some((_, Type::Function { params, return_type })) =>
                Some((m.name, params.clone(), (**return_type).clone())),
            _ => None,
        });
    };

    // the parameters the receiver could not determine: everything the desugared
    // function declares past the impl's own list.
    let own = &sig.generics[sig.generics.len().min(m.generics.len())..];
    if type_args.len() != own.len() {
        return Err(Error::new(*span, format!(
            "method '{}' expects {} generic argument{} in `::<...>`, got {}",
            field, own.len(), if own.len() == 1 { "" } else { "s" }, type_args.len()))
            .with_label(*span, format!(
                "declares {} generic parameter{} of its own",
                own.len(), if own.len() == 1 { "" } else { "s" }))
            .with_note(format!(
                "supply {} with a turbofish, e.g. `.{}::<...>(...)` - they cannot be \
                 inferred from the arguments",
                if own.len() == 1 { "it" } else { "them" }, field)));
    }

    let (mut types, mut consts) = bind_turbofish(cx, field, own, type_args, span)?;
    types.extend(u.types.iter().map(|(k, v)| (*k, v.clone())));
    consts.extend(u.consts.iter().map(|(k, v)| (*k, v.clone())));
    Ok(Some((
        m.name,
        sig.params.iter().map(|p| subst_param_type(cx, &types, &consts, p)).collect(),
        subst_param_type(cx, &types, &consts, &sig.return_type),
    )))
}

/// Resolve a method call `base.field(args)` when `base` is a (possibly
/// pointer-wrapped) type parameter, dispatching through the param's trait bounds.
/// Returns `Ok(Some(result_type))` on success, `Ok(None)` if `base_ty` is not a
/// type param at all (the caller then tries concrete method resolution), or an
/// error if the base *is* a type param but no bound provides `field` (a bare type
/// param has no methods of its own) or the arguments don't match.
fn resolve_bounded_method<'a>(
    cx: &mut Context<'a>,
    base_ty: &Type<'a>,
    field: &str,
    args: &[Expr<'a>],
    span: &Span,
) -> Result<Option<Type<'a>>, Error> {
    let param = match base_ty {
        Type::Param(n) => *n,
        Type::Pointer(inner) => match inner.as_ref() {
            Type::Param(n) => *n,
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };

    // find the first bound trait that declares a method named `field`.
    let bounds = cx.generic_bounds.get(param).cloned().unwrap_or_default();
    let mut sig = None;
    for tr in &bounds {
        if let Some(def) = cx.traits.get(tr)
            && let Some(m) = def.methods.get(field) {
                sig = Some((m.params.clone(), m.return_type.clone()));
                break;
            }
    }
    let Some((params, return_type)) = sig else {
        return Err(Error::new(*span,
            format!("no method '{}' on type parameter '{}'", field, param))
            .with_note(format!(
                "add a trait bound that provides it, e.g. `<{}: SomeTrait>`", param)));
    };

    // `Self` in the trait signature refers to the param type `T` here.
    let self_ty = Type::Param(param);
    if args.len() != params.len() {
        return Err(Error::new(*span, format!("method '{}' expects {} argument(s), got {}",
            field, params.len(), args.len())));
    }
    for (pty, arg) in params.iter().zip(args) {
        let expected = subst_self(pty, &self_ty);
        check_expr(cx, &expected, arg)?;
    }
    Ok(Some(subst_self(&return_type, &self_ty)))
}

/// Resolve an associated (no-`self`) call made *through a type parameter*,
/// `P::assoc(args)`, dispatching through `P`'s trait bounds. The resolver leaves
/// such a path unresolved (it can't name a concrete symbol); this checks it, and
/// monomorphization later re-mangles `P::assoc` to the concrete `Gain$assoc`.
/// `Ok(None)` when `cname` is not `Param::sym` for an in-scope type param (the
/// caller then tries its other Path interpretations, e.g. an enum constructor).
fn resolve_bounded_assoc<'a>(
    cx: &mut Context<'a>,
    cname: &NameRef<'a>,
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
    span: &Span,
) -> Result<Option<Type<'a>>, Error> {
    let segs = &cname.path.segments;
    if segs.len() != 2 || !cx.generics.contains(&segs[0]) {
        return Ok(None);
    }
    let param = segs[0];
    let method = segs[1];

    // find the first bound trait that declares `method`.
    let bounds = cx.generic_bounds.get(param).cloned().unwrap_or_default();
    let mut sig = None;
    for tr in &bounds {
        if let Some(def) = cx.traits.get(tr)
            && let Some(m) = def.methods.get(method) {
                sig = Some((m.receiver, m.params.clone(), m.return_type.clone()));
                break;
            }
    }
    let Some((receiver, params, return_type)) = sig else {
        return Err(Error::new(*span,
            format!("no associated function '{}' on type parameter '{}'", method, param))
            .with_note(format!(
                "add a trait bound that provides it, e.g. `<{}: SomeTrait>`", param)));
    };
    // `P::m()` names it without a receiver, so `m` must actually be associated.
    if receiver != Receiver::Associated {
        return Err(Error::new(*span,
            format!("'{}::{}' takes a receiver", param, method))
            .with_note(format!("call it as a method, `x.{}(...)`", method)));
    }
    if !type_args.is_empty() {
        return Err(Error::new(*span, format!(
            "associated function '{}' takes no generic arguments", method)));
    }

    // `Self` in the trait signature is the param type `P` here.
    let self_ty = Type::Param(param);
    if args.len() != params.len() {
        return Err(Error::new(*span, format!("associated function '{}' expects {} argument(s), got {}",
            method, params.len(), args.len())));
    }
    for (pty, arg) in params.iter().zip(args) {
        let expected = subst_self(pty, &self_ty);
        check_expr(cx, &expected, arg)?;
    }
    Ok(Some(subst_self(&return_type, &self_ty)))
}

/// Check that `arg` is a `*T` for the turbofished element type `T`. Shared by
/// the two intrinsics that address a slot rather than take one by value; both
/// name the pointee in the turbofish, so the pointer type is derived, never
/// written, and a mismatch is worth spelling out in full.
fn expect_pointer_to<'a>(
    cx: &mut Context<'a>,
    intrinsic: Intrinsic,
    pointee: &Type<'a>,
    arg: &Expr<'a>,
    which: &str,
) -> Result<(), Error> {
    let want = Type::Pointer(Box::new(pointee.clone()));
    let got = infer(cx, arg)?;
    if got != want {
        return Err(Error::new(arg.span, format!("{}() {} argument must be `{}`, got `{}`",
                     intrinsic, which, cx.show(&want), cx.show(&got))));
    }
    Ok(())
}

fn typecheck_intrinsic<'a>(
    cx: &mut Context<'a>,
    intrinsic: Intrinsic,
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
    span: Span,
    expr_id: usize,
) -> Result<Type<'a>, Error> {
    let sig = intrinsic.signature();
    // validates turbofish/value arity and binds the type/const arguments
    // value arguments remain in `args` (indexed from 0) and are checked per-intrinsic
    let (tys, consts) = bind_generics(cx, intrinsic, &sig, type_args, args, &span)?;

    match intrinsic {
        Intrinsic::Null => {
            // null::<*T>() -> *T. turbofish validated as a pointer by bind_generics
            let target_ty = tys[0].clone();
            cx.node_types.insert(expr_id, target_ty.clone());
            Ok(target_ty)
        }

        // Intrinsic::Len => {
        //     let arg_ty = infer(cx, &args[0])?;
        //     if !matches!(arg_ty, Type::Slice(_) | Type::Array(_, _) | Type::Pointer(_) | Type::Str) {
        //         return Err(Error {
        //             msg: format!("len() expects a slice, array, pointer or str, got {}", arg_ty),
        //             span,
        //         });
        //     }
        //     cx.node_types.insert(expr_id, Type::Int32);
        //     Ok(Type::Int32)
        // }
        Intrinsic::NumericalCast => {
            // numerical_cast::<T>(value) -> T
            let target_ty = tys[0].clone();
            let value_ty = infer(cx, &args[0])?;
            // an enum casts to/from its integer repr, so it is allowed here too.
            if !value_ty.is_numeric() && !cx.enums.contains_key(&value_ty.def().unwrap_or(DefId::UNRESOLVED)) {
                return Err(Error::new(span, format!(
                    "numerical_cast() argument must be a numeric type, got {}", cx.show(&value_ty))));
            }
            cx.node_types.insert(expr_id, target_ty.clone());
            Ok(target_ty)
        }
        Intrinsic::Sizeof => {
            // sizeof::<T>() -> u64. The type argument is validated by bind_generics.
            cx.node_types.insert(expr_id, Type::Uint64);
            Ok(Type::Uint64)
        }
        Intrinsic::PtrCast => {
            // ptr_cast::<*T>(p: *U) -> *T. The turbofish (validated as a pointer
            // by bind_generics) is the result type, the argument must be a pointer
            let target_ty = tys[0].clone();
            let value_ty = infer(cx, &args[0])?;
            // `str` is a raw pointer too, so it may be reinterpreted like any other.
            if !matches!(value_ty, Type::Pointer(_) | Type::Str) {
                return Err(Error::new(span, format!(
                    "ptr_cast() argument must be a pointer, got {}", cx.show(&value_ty))));
            }
            cx.node_types.insert(expr_id, target_ty.clone());
            Ok(target_ty)
        }
        Intrinsic::PtrWrite => {
            // ptr_write::<T>(dst: *T, value: T) -> void
            let ty = tys[0].clone();
            expect_pointer_to(cx, intrinsic, &ty, &args[0], "first")?;
            // checked rather than inferred, so an untyped literal takes `T`.
            check_expr(cx, &ty, &args[1])?;
            cx.node_types.insert(expr_id, Type::Void);
            Ok(Type::Void)
        }
        Intrinsic::DropInPlace => {
            // drop_in_place::<T>(ptr: *T, count) -> void
            let ty = tys[0].clone();
            expect_pointer_to(cx, intrinsic, &ty, &args[0], "first")?;
            let count_ty = infer(cx, &args[1])?;
            // any integer width: the expansion counts in whatever type it is
            // given, so a `u64` length and a `u32` one both work unconverted.
            if !matches!(count_ty, Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
                                 | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64) {
                return Err(Error::new(span, format!("{}() second argument must be an integer count, got `{}`",
                             intrinsic, cx.show(&count_ty))));
            }
            cx.node_types.insert(expr_id, Type::Void);
            Ok(Type::Void)
        }
        Intrinsic::SimdSplat => {
            let ty = tys[0].clone();
            let size = consts[0].clone();

            let value_ty = infer(cx, &args[0])?;
            if value_ty != ty {
                return Err(Error::new(span, format!(
                    "simd_splat() value argument must be of the element type, got {}", cx.show(&value_ty))));
            }

            let simd_ty = Type::Simd(Box::new(ty), size);
            cx.node_types.insert(expr_id, simd_ty.clone());
            Ok(simd_ty)
        }
        Intrinsic::SimdLoad => {
            let ty = tys[0].clone();
            let size = consts[0].clone();

            let slice_ty = infer(cx, &args[0])?;
            let offset_ty = infer(cx, &args[1])?;

            if !offset_ty.is_integer() {
                return Err(Error::new(span, format!(
                    "simd_load() offset argument must be an integer type, got {}", cx.show(&offset_ty))));
            }

            match slice_ty {
                Type::Slice(inner) | Type::Pointer(inner) if *inner == ty => {
                    let simd_ty = Type::Simd(Box::new(ty), size);
                    cx.node_types.insert(expr_id, simd_ty.clone());
                    Ok(simd_ty)
                },
                _ => {
                    Err(Error::new(span, "simd_load() first argument has the wrong type")
                        .with_label(span, format!("expected `[{0}]` or `*{0}`, got `{1}`",
                            cx.show(&ty), cx.show(&slice_ty))))
                }
            }
        }
        Intrinsic::SimdStore => {
            let ty = tys[0].clone();
            let size = consts[0].clone();

            let slice_ty = infer(cx, &args[0])?;
            let offset_ty = infer(cx, &args[1])?;
            let value_ty = infer(cx, &args[2])?;

            if !offset_ty.is_integer() {
                return Err(Error::new(span, format!(
                    "simd_store() offset argument must be an integer type, got {}", cx.show(&offset_ty))));
            }

            let expected_value_ty = Type::Simd(Box::new(ty.clone()), size);
            if value_ty != expected_value_ty {
                return Err(Error::new(span, "simd_store() value argument has the wrong type")
                    .with_label(span, format!("expected `{}`, got `{}`",
                        cx.show(&expected_value_ty), cx.show(&value_ty))));
            }

            match slice_ty {
                Type::Slice(inner) | Type::Pointer(inner) if *inner == ty => Ok(Type::Void),
                _ => {
                    Err(Error::new(span, "simd_store() first argument has the wrong type")
                        .with_label(span, format!("expected `[{0}]` or `*{0}`, got `{1}`",
                            cx.show(&ty), cx.show(&slice_ty))))
                }
            }
        }
        Intrinsic::SimdConcat => {
            let ty = tys[0].clone();
            let size = consts[0].clone();

            let value1_ty = infer(cx, &args[0])?;
            let value2_ty = infer(cx, &args[1])?;

            // both operands must be half-width vectors of the element type. the
            // exact half-size relation is only statically checkable when the size
            // is a literal; for a symbolic const param we verify the shape and
            // defer the width check to the post-mono re-typecheck.
            let expected_value_ty = match &size {
                ConstVal::Lit(n) => Some(Type::Simd(Box::new(ty.clone()), ConstVal::Lit(n / 2))),
                ConstVal::Param(_) => None,
            };
            let half_ok = |v: &Type<'a>| match &expected_value_ty {
                Some(expected) => v == expected,
                None => matches!(v, Type::Simd(inner, _) if **inner == ty),
            };
            if !half_ok(&value1_ty) {
                return Err(Error::new(span, "simd_concat() first value argument has the wrong type")
                    .with_label(span, format!("got `{}`", cx.show(&value1_ty)))
                    .with_note(format!(
                        "both operands must be SIMD vectors of `{}` at half the result size",
                        cx.show(&ty))));
            }
            if !half_ok(&value2_ty) {
                return Err(Error::new(span, "simd_concat() second value argument has the wrong type")
                    .with_label(span, format!("got `{}`", cx.show(&value2_ty)))
                    .with_note(format!(
                        "both operands must be SIMD vectors of `{}` at half the result size",
                        cx.show(&ty))));
            }

            let result_ty = Type::Simd(Box::new(ty.clone()), size.clone());
            cx.node_types.insert(expr_id, result_ty.clone());
            Ok(result_ty)
        }
        Intrinsic::SimdLow | Intrinsic::SimdHigh => {
            // simd_low/high::<T, N>(value: simd[T, M]) -> simd[T, N] where N < M
            let ty = tys[0].clone();
            let size = consts[0].clone();

            let value_ty = infer(cx, &args[0])?;
            // the input must be a wider vector of the element type. `N < M` is only
            // checkable statically when both are literals; a symbolic const param
            // defers the width comparison to the post-mono re-typecheck.
            let wider = |inner_size: &ConstVal<'a>| match (inner_size, &size) {
                (ConstVal::Lit(m), ConstVal::Lit(n)) => m > n,
                _ => true,
            };
            match value_ty {
                Type::Simd(ref inner_ty, ref inner_size) if **inner_ty == ty && wider(inner_size) => {
                    let result_ty = Type::Simd(Box::new(ty.clone()), size.clone());
                    cx.node_types.insert(expr_id, result_ty.clone());
                    Ok(result_ty)
                }
                _ => {
                    Err(Error::new(span,
                        format!("{}() value argument has the wrong type", intrinsic))
                        .with_label(span, format!("got `{}`", cx.show(&value_ty)))
                        .with_note(format!(
                            "it must be a SIMD vector of `{}` wider than the result",
                            cx.show(&ty))))
                }
            }
        }
    }
}

/// The inclusive range of `ty`, if it is an integer type. Every bound fits in
/// `i128`, which is what makes a width-less literal checkable against any of
/// them - including `u64`, whose top half is out of `i64`'s reach.
fn int_range(ty: &Type<'_>) -> Option<(i128, i128)> {
    Some(match ty {
        Type::Int8   => (i8::MIN as i128, i8::MAX as i128),
        Type::Int16  => (i16::MIN as i128, i16::MAX as i128),
        Type::Int32  => (i32::MIN as i128, i32::MAX as i128),
        Type::Int64  => (i64::MIN as i128, i64::MAX as i128),
        Type::Uint8  => (0, u8::MAX as i128),
        Type::Uint16 => (0, u16::MAX as i128),
        Type::Uint32 => (0, u32::MAX as i128),
        Type::Uint64 => (0, u64::MAX as i128),
        _ => return None,
    })
}

/// Can a width-less literal of value `v` be the type `expected` asks for?
///
/// An integer literal fits an integer type whose range contains it, and any
/// float type - `let x: f64 = 1;` is allowed, since the intent is unambiguous.
/// A *float* literal (`v` is `None`) fits only a float type: an integer target
/// would have to discard digits the source wrote.
fn literal_fits(v: Option<i128>, expected: &Type<'_>) -> bool {
    match (v, expected) {
        (_, Type::Float32 | Type::Float64) => true,
        (Some(v), ty) => int_range(ty).is_some_and(|(lo, hi)| (lo..=hi).contains(&v)),
        (None, _) => false,
    }
}

/// The value of a width-less literal leaf, negation folded in: `Some(v)` for an
/// integer literal, `None` for a float one. Any other expression - a suffixed
/// literal included - is not context-typed and yields `None` via the outer
/// `Option`.
///
/// Negation has to be folded rather than checked through, because the range test
/// is asymmetric: `128` does not fit `i8` but `-128` does.
fn untyped_lit(expr: &Expr<'_>) -> Option<Option<i128>> {
    match &expr.value {
        ExprNode::IntLit(v) => Some(Some(*v)),
        ExprNode::FloatLit(_) => Some(None),
        ExprNode::Unary { op: UnaryOp::Neg, operand } => match &operand.value {
            ExprNode::IntLit(v) => Some(Some(-*v)),
            ExprNode::FloatLit(_) => Some(None),
            _ => None,
        },
        _ => None,
    }
}

/// Is `expr` built only out of width-less literals, so that it has no type of
/// its own for context to disagree with?
///
/// A literal, a negated one, or the two joined by an operator that yields the
/// operand type: `1 << 40` is no more an `i32` than `1` is, so a `u64` slot
/// should get it as a `u64` rather than watch it default and then overflow.
/// Comparisons are excluded - they yield `bool` regardless of their operands.
fn untyped_lit_shape(expr: &Expr<'_>) -> bool {
    use haven_common::ast::BinaryOp::*;
    match &expr.value {
        ExprNode::Binary { op, left, right } =>
            matches!(op, Add | Sub | Mul | Div | Mod | BitAnd | BitOr | BitXor | Shl | Shr)
                && untyped_lit_shape(left) && untyped_lit_shape(right),
        _ => untyped_lit(expr).is_some(),
    }
}

/// Give every node of a width-less literal expression the type `expected`, after
/// checking each leaf's value survives it. Assumes [`untyped_lit_shape`].
///
/// Every node is recorded, not just the root: MIL reads a binary operation's
/// type from its left operand's entry, and a negation's from the literal inside
/// it, so a half-typed tree would lower at the wrong width.
fn type_untyped_lit<'a>(
    cx: &mut Context<'a>,
    expected: &Type<'a>,
    expr: &Expr<'a>,
) -> Result<(), Error> {
    match &expr.value {
        ExprNode::Binary { left, right, .. } => {
            type_untyped_lit(cx, expected, left)?;
            type_untyped_lit(cx, expected, right)?;
        }
        _ => {
            let v = untyped_lit(expr).expect("untyped_lit_shape checked this leaf");
            if !literal_fits(v, expected) {
                return Err(match v {
                    Some(v) => Error::new(expr.span,
                        format!("integer literal {} does not fit in {}", v, cx.show(expected)))
                        .with_label(expr.span,
                            format!("out of range for `{}`", cx.show(expected))),
                    None => Error::new(expr.span, "type mismatch")
                        .with_label(expr.span,
                            format!("expected `{}`, got a float literal", cx.show(expected))),
                });
            }
            // `-<literal>` is one unit: the literal inside it is typed too.
            if let ExprNode::Unary { operand, .. } = &expr.value {
                cx.node_types.insert(operand.id, expected.clone());
            }
        }
    }
    cx.node_types.insert(expr.id, expected.clone());
    Ok(())
}

/// The operand type shared by both sides of a binary operator, with the other
/// side checked against it.
///
/// The left operand normally decides. But a width-less literal has no type to
/// decide with - it would default to `i32` and then reject a perfectly good
/// `i64` on the right - so when exactly one side is such a literal, the *other*
/// side picks. That is what makes `1 + n` work as well as `n + 1`. With literals
/// on both sides nothing has changed: the left one defaults, and the right one
/// follows it.
fn binary_operand_ty<'a>(
    cx: &mut Context<'a>,
    left: &Expr<'a>,
    right: &Expr<'a>,
) -> Result<Type<'a>, Error> {
    if untyped_lit_shape(left) && !untyped_lit_shape(right) {
        let ty = infer(cx, right)?;
        check_expr(cx, &ty, left)?;
        Ok(ty)
    } else {
        let ty = infer(cx, left)?;
        check_expr(cx, &ty, right)?;
        Ok(ty)
    }
}

/// The type arguments a generic enum's constructor is built with, and the
/// substitutions they induce. Three sources, in order of authority: an explicit
/// turbofish always wins; a constructor without one takes the arguments of the
/// type its position expects (`hint`, from `check_expr`); failing that, they
/// are recovered from the values the constructor gives its payload, the way a
/// bare generic call recovers a callee's from its arguments. Whatever was not
/// written is recorded under the node id for mono, which reads that alongside
/// an empty turbofish exactly as it does for a bare generic call.
///
/// A non-generic enum takes no arguments and rejects any it is given.
// these functions need every part of an enum constructor call. keeping those
// parts as arguments is clearer than wrapping them in a one-use struct.
// buy me a coffee if you think otherwise.
#[allow(clippy::too_many_arguments)]
fn enum_ctor_targs<'a>(
    cx: &mut Context<'a>,
    node_id: usize,
    ename: DefId,
    ctor: &NameRef<'a>,
    type_args: &[GenericArg<'a>],
    hint: Option<&[GenericArg<'a>]>,
    payload: &[(&Type<'a>, &Expr<'a>)],
    site: TargSite,
    span: &Span,
) -> Result<ResolvedGenerics<'a>, Error> {
    let Some(params) = cx.generic_enums.get(&ename).cloned() else {
        if !type_args.is_empty() {
            return Err(Error::new(*span, format!(
                "enum constructor '{}' takes no type arguments", ctor)));
        }
        return Ok((HashMap::new(), HashMap::new(), Vec::new()));
    };
    let args: Vec<GenericArg<'a>> = if !type_args.is_empty() {
        type_args.to_vec()
    } else {
        let inferred = match hint {
            Some(hint) => hint.to_vec(),
            None => infer_enum_type_args(cx, &ctor.to_string(), &params, payload, site, span)?,
        };
        cx.inferred_type_args.insert(node_id, inferred.clone());
        inferred
    };
    bind_struct_generics(cx, &cx.name_of(ename), &params, &args, span)
}

/// A tuple or unit variant constructor call, `E::V(args)`: arity, then each
/// argument against its (substituted) payload field type, yielding the enum
/// type with its concrete arguments.
#[allow(clippy::too_many_arguments)]
fn check_enum_call_ctor<'a>(
    cx: &mut Context<'a>,
    expr: &Expr<'a>,
    cname: &NameRef<'a>,
    ename: DefId,
    payload_tys: &[Type<'a>],
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
    hint: Option<&[GenericArg<'a>]>,
    span: &Span,
) -> Result<Type<'a>, Error> {
    // arity first: inference pairs payload types with arguments positionally.
    if args.len() != payload_tys.len() {
        return Err(Error::new(*span, format!("variant '{}' expects {} field(s), got {}",
            cname, payload_tys.len(), args.len())));
    }
    let payload: Vec<(&Type<'a>, &Expr<'a>)> = payload_tys.iter().zip(args).collect();
    let site = if payload.is_empty() { TargSite::UnitVariant } else { TargSite::Call };
    let (type_subst, const_subst, resolved_args) =
        enum_ctor_targs(cx, expr.id, ename, cname, type_args, hint, &payload, site, span)?;
    for (pty, arg) in payload_tys.iter().zip(args) {
        let expected = subst_param_type(cx, &type_subst, &const_subst, pty);
        check_expr(cx, &expected, arg)?;
    }
    let ty = Type::Named { def: ename, args: resolved_args };
    cx.node_types.insert(expr.id, ty.clone());
    Ok(ty)
}

/// A struct-style variant constructor, `E::V { id: .., val: .. }`: it looks
/// like a struct literal but names a variant, so the literal is checked against
/// the variant's payload struct - every field, in declaration order - and
/// yields the enum type with its concrete arguments.
#[allow(clippy::too_many_arguments)]
fn check_enum_struct_ctor<'a>(
    cx: &mut Context<'a>,
    expr: &Expr<'a>,
    name: &NameRef<'a>,
    ename: DefId,
    variant: &'a str,
    type_args: &[GenericArg<'a>],
    fields: &[(&'a str, Expr<'a>)],
    hint: Option<&[GenericArg<'a>]>,
    span: &Span,
) -> Result<Type<'a>, Error> {
    let pstruct = cx.payloads[&(ename, variant)];
    let pdef = match cx.types.get(&pstruct) {
        Some(d) if !d.fields.is_empty() => d.fields.clone(),
        _ => return Err(Error::new(*span, format!("variant '{}' has no fields", name))
            .with_note(format!("construct it as `{}`", name))),
    };
    // a tuple variant's fields are named "0", "1", ...; those can't be
    // written in a `{ }` literal, so point the user at the `( )` form.
    if pdef.first().is_some_and(|(n, _)| n.bytes().all(|b| b.is_ascii_digit())) {
        return Err(Error::new(*span, format!("variant '{}' is a tuple variant", name))
            .with_note(format!("construct it with `{}(...)`", name)));
    }
    if fields.len() != pdef.len() {
        return Err(Error::new(*span, format!("variant '{}' expects {} field(s), got {}",
            name, pdef.len(), fields.len())));
    }
    // fields pair with the declaration positionally for inference, up to the
    // first misnamed one; that one is reported by the per-field check below.
    let payload: Vec<(&Type<'a>, &Expr<'a>)> = pdef.iter().zip(fields)
        .take_while(|((dn, _), (ln, _))| dn == ln)
        .map(|((_, dt), (_, lv))| (dt, lv))
        .collect();
    let (type_subst, const_subst, resolved_args) =
        enum_ctor_targs(cx, expr.id, ename, name, type_args, hint, &payload, TargSite::StructLit, span)?;
    // field order must match the declaration (same rule as a struct
    // literal); each field checks against its (param-substituted)
    // declared payload type.
    for ((def_name, def_ty), (lit_name, lit_value)) in pdef.iter().zip(fields.iter()) {
        if def_name != lit_name {
            return Err(Error::new(lit_value.span, format!(
                "In variant '{}': expected field '{}', got '{}'",
                name, def_name, lit_name)));
        }
        let expected = subst_param_type(cx, &type_subst, &const_subst, def_ty);
        check_expr(cx, &expected, lit_value)?;
    }
    let ty = Type::Named { def: ename, args: resolved_args };
    cx.node_types.insert(expr.id, ty.clone());
    Ok(ty)
}

/// A bare `Enum::Variant` used as a value. A unit variant is one: a field-less
/// enum's scalar discriminant, or (for a data enum) an aggregate with no
/// payload. A tuple/struct variant used bare is a missing constructor call -
/// `Msg::Note` needs `Msg::Note(...)`.
///
/// A bare path has no syntax to attach a turbofish, so a generic enum's unit
/// variant is only writable this way where the context supplies the arguments
/// (`let x: Option<i32> = Option::None;`); anywhere else it needs the call form
/// with a turbofish, `Option::None::<i32>()`, and the error says so.
fn check_enum_unit_path<'a>(
    cx: &mut Context<'a>,
    expr: &Expr<'a>,
    path: &NameRef<'a>,
    ename: DefId,
    hint: Option<&[GenericArg<'a>]>,
    span: &Span,
) -> Result<Type<'a>, Error> {
    let variant = path.variant();
    if cx.enums[&ename].payloads.get(variant).is_some_and(|p| !p.is_empty()) {
        return Err(Error::new(*span, format!("variant '{}' carries a payload", path))
            .with_note(format!("construct it with `{}(...)`", path)));
    }
    let (_, _, resolved_args) =
        enum_ctor_targs(cx, expr.id, ename, path, &[], hint, &[], TargSite::UnitVariant, span)?;
    Ok(Type::Named { def: ename, args: resolved_args })
}

/// A generic enum's constructor, written without a turbofish, in a position
/// that expects that very enum: the expected type's arguments are the
/// constructor's, so `Option::None` under `let x: Option<i32>` is
/// `Option::None::<i32>()`, and the `42` in `Option::Some(42)` under
/// `Option<i64>` is an `i64`, as a width-less literal under a plain `i64`
/// would be. Returns `None` for any other expression, or a constructor of a
/// different enum, or one that already carries its own turbofish, so the caller
/// falls back to `infer` and the ordinary comparison.
fn enum_ctor_with_hint<'a>(
    cx: &mut Context<'a>,
    expected: &Type<'a>,
    expr: &Expr<'a>,
) -> Result<Option<Type<'a>>, Error> {
    let Type::Named { def: want, args: hint } = expected else { return Ok(None) };
    if !cx.generic_enums.contains_key(want) { return Ok(None); }
    let span = expr.span;
    match &expr.value {
        ExprNode::Call { func, type_args, args } if type_args.is_empty() => {
            let ExprNode::Path(cname) = &func.value else { return Ok(None) };
            let Some((ename, payload_tys)) = enum_variant_ctor(cx, cname) else { return Ok(None) };
            if ename != *want { return Ok(None); }
            check_enum_call_ctor(cx, expr, cname, ename, &payload_tys, type_args, args, Some(hint), &span)
                .map(Some)
        }
        ExprNode::Struct { name, type_args, fields } if type_args.is_empty() => {
            let Some((ename, variant)) = split_enum_variant(cx, name) else { return Ok(None) };
            if ename != *want { return Ok(None); }
            check_enum_struct_ctor(cx, expr, name, ename, variant, type_args, fields, Some(hint), &span)
                .map(Some)
        }
        ExprNode::Path(path) => {
            let Some((ename, _, _)) = enum_variant(cx, path) else { return Ok(None) };
            if ename != *want { return Ok(None); }
            check_enum_unit_path(cx, expr, path, ename, Some(hint), &span).map(Some)
        }
        _ => Ok(None),
    }
}

pub(crate) fn check_expr<'a>(
    cx: &mut Context<'a>,
    expected: &Type<'a>,
    expr: &Expr<'a>,
) -> Result<(), Error> {
    let metadata = expr;
    let value = &metadata.value;
    let span = metadata.span;

    // an expression written entirely in width-less literals has no type of its
    // own: it takes the one being asked for here, provided each value survives
    // it. `infer` only sees such a literal where there is no expectation to read,
    // and defaults it to `i32`/`f32` there.
    //
    // A non-numeric expectation falls through instead, so `let b: bool = 1;`
    // still reports the ordinary mismatch rather than a range complaint.
    if expected.is_numeric() && untyped_lit_shape(expr) {
        return type_untyped_lit(cx, expected, expr);
    }

    let actual = match value {
        ExprNode::Bool(_)    => Type::Bool,
        ExprNode::Int8(_)    => Type::Int8,
        ExprNode::Int16(_)   => Type::Int16,
        ExprNode::Int32(_)   => Type::Int32,
        ExprNode::Int64(_)   => Type::Int64,
        ExprNode::Uint8(_)   => Type::Uint8,
        ExprNode::Uint16(_)  => Type::Uint16,
        ExprNode::Uint32(_)  => Type::Uint32,
        ExprNode::Uint64(_)  => Type::Uint64,
        ExprNode::Float32(_) => Type::Float32,
        ExprNode::Float64(_) => Type::Float64,
        ExprNode::Str(_)     => Type::Str,

        // `[value; N]` in a known array position: check the element against the
        // element type being asked for rather than inferring it on its own, for
        // the same reason the list form does - otherwise the width-less literal
        // in `let xs: [i64; 8] = [0; 8];` would settle on `i32` and then the
        // array would mismatch as a whole. A length disagreement falls through to
        // `infer`, which reports it.
        ExprNode::Repeat { value, count } => {
            let elem = match expected {
                Type::Array(elem, n) if n == count => Some(elem),
                Type::Slice(elem) => Some(elem),
                _ => None,
            };
            match elem {
                Some(elem) => {
                    let elem = (**elem).clone();
                    check_expr(cx, &elem, value)?;
                    Type::Array(Box::new(elem), count.clone())
                }
                None => infer(cx, expr)?,
            }
        },

        // let xs: []i32 = []; so the type of [] is i32
        // else, if [...] is populated, infer it
        ExprNode::Slice(inner) if inner.is_empty() => {
            match expected {
                Type::Slice(elem_ty) => Type::Slice(elem_ty.clone()),
                _ => {
                    return Err(Error::new(span, "type mismatch")
                        .with_label(span, format!("expected `{}`, got an empty slice literal",
                            cx.show(expected))));
                }
            }
        },

        // a populated array literal in a known array/slice position: check each
        // element against the element type being asked for, instead of inferring
        // the whole literal from its first element. Without this the elements are
        // typed before anything says what they should be, so the width-less
        // literals in `let xs: [i64; 3] = [1, 2, 3];` would default to `i32` and
        // the array would then mismatch as a whole.
        //
        // A length disagreement, a const-param length, or any other expected type
        // falls through to `infer`, which reports the mismatch as before.
        ExprNode::Slice(inner) => {
            let elem = match expected {
                Type::Array(elem, ConstVal::Lit(n)) if *n == inner.len() => Some(elem),
                Type::Slice(elem) => Some(elem),
                _ => None,
            };
            match elem {
                Some(elem) => {
                    let elem = (**elem).clone();
                    for e in inner {
                        check_expr(cx, &elem, e)?;
                    }
                    // an array literal is an Array even where a slice is wanted;
                    // the coercion below is what accepts it, exactly as when the
                    // literal is inferred.
                    Type::Array(Box::new(elem), ConstVal::Lit(inner.len()))
                }
                None => infer(cx, expr)?,
            }
        },

        // a generic enum's constructor in a position that already knows which
        // instance it wants takes its type arguments from there, which is what
        // lets `let x: Option<i32> = Option::None;` say nothing twice. Anything
        // else is inferred on its own and compared below.
        _ => match enum_ctor_with_hint(cx, expected, expr)? {
            Some(ty) => ty,
            None => infer(cx, expr)?,
        },
    };

    // Resolve any associated type whose base is concrete, and attach the
    // uniquely selected trait to symbolic projections. This makes a written
    // `T::Item` compare equal to the `Self::Item` returned by a bounded trait
    // method without relying on the associated name alone.
    let expected = cx.normalize_type(expected)
        .map_err(|msg| Error::new(span, msg))?;
    let actual = cx.normalize_type(&actual)
        .map_err(|msg| Error::new(span, msg))?;
    let compatible = actual == expected ||
        // `!` (the type of a diverging expression like `abort(...)`) coerces to
        // any expected type: control never reaches the surrounding context, so
        // there is no value to be type-incompatible.
        actual == Type::Never ||
        matches!((&actual, &expected),
            (Type::Array(inner_actual, _), Type::Slice(inner_expected))
                if inner_actual == inner_expected
        );

    if !compatible {
        // header stays short and the types go on the underline: a monomorphized
        // type name can be long enough on its own to wrap the header line.
        return Err(Error::new(span, "type mismatch")
            .with_label(span, format!("expected `{}`, got `{}`",
                cx.show(&expected), cx.show(&actual))));
    }

    cx.node_types.insert(metadata.id, actual);
    Ok(())
}

pub(crate) fn infer<'a>(
    cx: &mut Context<'a>,
    expr: &Expr<'a>,
) -> Result<Type<'a>, Error> {
    let metadata = expr;
    let value = &metadata.value;
    let span = metadata.span;

    let ty = match value {
        ExprNode::Bool(_)    => Type::Bool,
        ExprNode::Int8(_)    => Type::Int8,
        ExprNode::Int16(_)   => Type::Int16,
        ExprNode::Int32(_)   => Type::Int32,
        ExprNode::Int64(_)   => Type::Int64,
        ExprNode::Uint8(_)   => Type::Uint8,
        ExprNode::Uint16(_)  => Type::Uint16,
        ExprNode::Uint32(_)  => Type::Uint32,
        ExprNode::Uint64(_)  => Type::Uint64,
        ExprNode::Float32(_) => Type::Float32,
        ExprNode::Float64(_) => Type::Float64,
        ExprNode::Str(_)     => Type::Str,

        // a width-less literal reaching `infer` is one with no expectation to
        // take its type from - `let n = 0;`, or the left operand of a binary
        // operator. It defaults to 32 bits, which is what a bare literal always
        // meant; a value too big for that has to say which type it wants.
        ExprNode::IntLit(v) => {
            if !literal_fits(Some(*v), &Type::Int32) {
                return Err(Error::new(span, format!(
                    "integer literal {} does not fit in i32", v))
                    .with_label(span, "a bare literal defaults to i32")
                    .with_note(format!(
                        "add a width suffix (`{}i64`) or a type annotation to say which \
                         integer type it is", v)));
            }
            Type::Int32
        },
        ExprNode::FloatLit(_) => Type::Float32,

        // an `Enum::Variant` reference - the only qualified path name resolution
        // leaves standing. A unit variant is a value: a field-less enum's scalar
        // discriminant, or (for a data enum) an aggregate with no payload. A
        // tuple/struct variant used bare is a missing constructor call -
        // `Msg::Note` needs `Msg::Note(...)`.
        ExprNode::Path(path) => {
            let Some((ename, _val, _repr)) = enum_variant(cx, path) else {
                return Err(Error::new(span, format!("Undefined variable '{}'", path)));
            };
            check_enum_unit_path(cx, expr, path, ename, None, &span)?
        },

        ExprNode::Var(name) => {
            let Some((binding, ty)) = cx.lookup(name) else {
                let msg = format!("Undefined variable '{}'", name);
                return Err(Error::new(span, msg));
            };
            let binding = *binding;
            let ty = ty.clone();
            // record which param/local this use resolves to (globals -> None)
            if let Some(b) = binding {
                cx.resolved.insert(metadata.id, b);
            }
            ty
        },

        // A generic function taken by value: `foo::<Concrete>`. The turbofish
        // fully specifies the instance, so its type is the *substituted* function
        // signature - a plain function pointer. Monomorphization mints `foo$...`
        // and rewrites this node to a bare `Var` of that symbol, so nothing past
        // mono sees a `FnRef`.
        ExprNode::FnRef { name, type_args } => {
            let fname = name.path.as_single().ok_or_else(|| Error::new(span, format!(
                "'{}' is not a function", name)))?;
            let Some(sig) = cx.generic_fns.get(fname).cloned() else {
                // a non-generic function reference carries no turbofish, so a name
                // here that isn't a generic fn is either undefined or a plain fn
                // wrongly given type arguments.
                let msg = if cx.lookup(fname).is_some() {
                    format!("'{}' takes no type arguments", fname)
                } else {
                    format!("unknown function '{}'", fname)
                };
                return Err(Error::new(span, msg));
            };
            if type_args.len() != sig.generics.len() {
                return Err(Error::new(span, format!(
                    "{}() expects {} generic argument{} in `::<...>`, got {}",
                    fname, sig.generics.len(),
                    if sig.generics.len() == 1 { "" } else { "s" }, type_args.len(),
                )));
            }
            let (type_bindings, const_bindings) =
                bind_turbofish(cx, fname, &sig.generics, type_args, &span)?;
            let params: Vec<Type<'a>> = sig.params.iter()
                .map(|p| subst_param_type(cx, &type_bindings, &const_bindings, p)).collect();
            let ret = subst_param_type(cx, &type_bindings, &const_bindings, &sig.return_type);
            let fn_ty = Type::Function { params, return_type: Box::new(ret) };
            cx.node_types.insert(metadata.id, fn_ty.clone());
            fn_ty
        },

        // `[value; N]` -> `[T; N]`, where `T` is the element's own type. `N` is
        // either a literal or a const generic parameter of the enclosing proc;
        // an identifier that names neither is caught here rather than surviving
        // to monomorphization as an unresolvable length.
        ExprNode::Repeat { value, count } => {
            if let ConstVal::Param(n) = count
                && !cx.const_generics.contains(n) {
                    return Err(Error::new(span, format!(
                        "unknown const parameter '{}' as a repeat count", n))
                        .with_note("a repeat count must be an integer literal or a \
                                    `const` generic parameter of the enclosing proc"));
                }
            if let ConstVal::Lit(0) = count {
                return Err(Error::new(span, "a repeat count must be at least 1")
                    .with_label(span, "this would produce an array of no elements")
                    .with_note("haven has no zero-length array type; use an empty slice \
                                literal (`[]`) with an annotation if that is what you want"));
            }
            let elem = infer(cx, value)?;
            Type::Array(Box::new(elem), count.clone())
        },

        ExprNode::Slice(inner) if inner.is_empty() => {
            return Err(Error::new(span, "cannot infer the type of an empty slice literal")
                .with_note("annotate it, e.g. `let xs: []i32 = [];`"));
        },

        // Infer [...] as Array first, convert to slice later if needed
        ExprNode::Slice(inner) => {
            let first_ty = infer(cx, &inner[0])?;
            for elem in inner.iter().skip(1) {
                check_expr(cx, &first_ty, elem)?;
            }
            Type::Array(Box::new(first_ty), ConstVal::Lit(inner.len()))
        },
        // ExprNode::Slice(inner) => {
        //     let first_ty = infer(cx, &inner[0])?;
        //     for elem in inner.iter().skip(1) {
        //         check_expr(cx, &first_ty, elem)?;
        //     }
        //     Type::Slice(Box::new(first_ty))
        // },

        ExprNode::Index { slice, index } => {
            let slice_ty = infer(cx, slice)?;
            let index_ty = infer(cx, index)?;
            if !index_ty.is_integer() {
                return Err(Error::new(span, "index must be an integer")
                    .with_label(span, format!("got `{}`", cx.show(&index_ty))));
            }
            match slice_ty {
                Type::Slice(inner)
                | Type::Array(inner, _)
                | Type::Pointer(inner) => *inner,
                _ => {
                    return Err(Error::new(span, "this type cannot be indexed")
                        .with_label(span, format!("got `{}`", cx.show(&slice_ty)))
                        .with_note("indexing needs a slice, array, or pointer"));
                }
            }
        },

        ExprNode::Unary { op, operand } => {
            let operand_ty = infer(cx, operand)?;
            match op {
                UnaryOp::AddrOf => {
                    // `&place` yields the place's address. `&<temporary>` (a call
                    // result, literal, arithmetic, ...) has no storage of its own,
                    // so MIL spills the value into a fresh slot and addresses that
                    // (`spill_temporary`). An *owning* temporary is rejected in the
                    // ownership pass, where the leak - a slot no scope destroys -
                    // is caught alongside the same case for `*self` receivers.
                    Type::Pointer(Box::new(operand_ty))
                },
                UnaryOp::Deref => match operand_ty {
                    Type::Pointer(inner) => *inner,
                    _ => {
                        return Err(Error::new(span, "this type cannot be dereferenced")
                            .with_label(span, format!("expected a pointer, got `{}`",
                                cx.show(&operand_ty))));
                    }
                },
                UnaryOp::Neg => if operand_ty.is_numeric() {
                    operand_ty
                } else {
                    return Err(Error::new(span, "this type cannot be negated")
                        .with_label(span, format!("expected a numeric type, got `{}`",
                            operand_ty)));
                },
                UnaryOp::Not => Type::Bool,
            }
        },

        ExprNode::Binary { op, left, right } => {
            use haven_common::ast::BinaryOp::*;
            match op {
                Eq | Ne => {
                    binary_operand_ty(cx, left, right)?;
                    Type::Bool
                },
                Add | Sub | Mul | Div | Mod
                | Lt | Gt | Le | Ge => {
                    let left_ty = binary_operand_ty(cx, left, right)?;

                    // check if both are numeric (scalar or SIMD)
                    if !left_ty.is_numeric_or_numeric_simd() {
                        return Err(Error::new(span, "binary operator needs numeric operands")
                            .with_label(span, format!("got `{}`", cx.show(&left_ty))));
                    }

                    match op {
                        Add | Sub | Mul | Div | Mod => left_ty, // returns scalar or SIMD
                        Lt | Gt | Le | Ge => Type::Bool,
                        _ => unreachable!(),
                    }
                },
                And | Or => {
                    let expected = Type::Bool;
                    check_expr(cx, &expected, left)?;
                    check_expr(cx, &expected, right)?;
                    Type::Bool
                },
                // bitwise and shifts: integer operands, same type on both sides
                // (LLVM requires the shift amount to match the value type), result
                // is that integer type.
                BitAnd | BitOr | BitXor | Shl | Shr => {
                    let left_ty = binary_operand_ty(cx, left, right)?;
                    if !left_ty.is_integer() {
                        return Err(Error::new(span,
                            format!("bitwise operator '{}' needs integer operands", op))
                            .with_label(span, format!("got `{}`", cx.show(&left_ty))));
                    }
                    left_ty
                },
            }
        },

        ExprNode::Struct { name, type_args, fields } => {
            // a struct-style enum-variant constructor `E::V { id: .., val: .. }`
            // looks like a struct literal but names a variant. Check the literal
            // against the variant's payload struct and yield the aggregate enum
            // type. Guarded before ordinary struct-literal handling.
            if let Some((ename, variant)) = split_enum_variant(cx, name) {
                return check_enum_struct_ctor(cx, expr, name, ename, variant, type_args, fields, None, &span);
            }

            // the literal named an enum but not one of its variants. (A path with
            // two segments is not itself the test: a module-qualified struct
            // literal, `osc::Osc { .. }`, keeps both segments and is perfectly
            // ordinary - which is why this asks what the name *resolved to*
            // rather than how it was spelled.)
            if cx.enums.contains_key(&name.def) {
                return Err(Error::new(span, format!("Unknown enum variant '{}'", name)));
            }
            let tdef = name.def;
            let name = cx.name_of(tdef);

            let def = match cx.types.get(&tdef) {
                Some(d) => d.fields.clone(),
                None => return Err(Error::new(span, format!("Unknown struct '{}'", name))),
            };

            // Bind the struct's params to the turbofish args so `Param` field types
            // check against a concrete type (and `[T; N]` sizes against a concrete
            // count). Non-generic structs take no args; a generic struct either
            // carries a turbofish or has its args recovered from the field values,
            // exactly as a bare call recovers a callee's from its arguments.
            // `struct_args` are the resolved args in declaration order.
            let (type_subst, const_subst, struct_args) = match cx.generic_structs.get(&tdef).cloned() {
                Some(params) => {
                    let args: Vec<GenericArg<'a>> = if type_args.is_empty() {
                        let inferred =
                            infer_struct_type_args(cx, &name, &params, &def, fields, &span)?;
                        // a literal with no turbofish keeps an empty `type_args`,
                        // so record the recovered args under the node id for mono
                        // to pick up - the same channel a bare generic call uses.
                        cx.inferred_type_args.insert(expr.id, inferred.clone());
                        inferred
                    } else {
                        type_args.to_vec()
                    };
                    bind_struct_generics(cx, &name, &params, &args, &span)?
                }
                None => {
                    if !type_args.is_empty() {
                        return Err(Error::new(span, format!(
                            "struct '{}' is not generic; no type arguments expected",
                            name,
                        )));
                    }
                    (HashMap::new(), HashMap::new(), Vec::new())
                }
            };

            if fields.len() != def.len() {
                return Err(Error::new(span, format!(
                    "Struct '{}' expects {} fields, got {}",
                    name, def.len(), fields.len()
                )));
            }

            // field order must match definition; each field checks against its
            // (param-substituted) declared type.
            for ((def_name, def_ty), (lit_name, lit_value)) in def.iter().zip(fields.iter()) {
                if def_name != lit_name {
                    return Err(Error::new(lit_value.span, format!(
                        "In struct '{}': expected field '{}', got '{}'",
                        name, def_name, lit_name
                    )));
                }
                let expected = subst_param_type(cx, &type_subst, &const_subst, def_ty);
                check_expr(cx, &expected, lit_value)?;
            }

            // The literal's type carries its concrete args (`Option<i32>`,
            // `Buf<i32, 8>`); monomorphization rewrites both this literal and the
            // type to the flat instance before codegen. Non-generic structs get
            // the usual no-args form.
            Type::Named { def: tdef, args: struct_args }
        },

        ExprNode::Access { base, field } => {
            let base_ty = infer(cx, base)?;
            let (struct_def, struct_args) = match &base_ty {
                Type::Named { def, args } => (*def, args.as_slice()),
                // auto-deref one level of pointer to a struct (like C's `->`)
                Type::Pointer(inner) => match inner.as_ref() {
                    Type::Named { def, args } => (*def, args.as_slice()),
                    _ => return Err(Error::new(span, format!(
                        "Cannot access field '{}' on type {}", field, cx.show(&base_ty)))),
                },
                _ => return Err(Error::new(span, format!(
                    "Cannot access field '{}' on type {}", field, cx.show(&base_ty)))),
            };

            let def = match cx.types.get(&struct_def).map(|i| &i.fields) {
                Some(d) => d,
                None => return Err(Error::new(span, format!("Unknown struct '{}'", cx.name_of(struct_def)))),
            };

            let field_ty = match def.iter().find(|(n, _)| n == field) {
                Some((_, ty)) => ty.clone(),
                None => return Err(Error::new(span, format!(
                    "Struct '{}' has no field '{}'", cx.name_of(struct_def), field))),
            };
            // for a generic-struct instance (`Option<i32>`, `Buf<i32, 8>`), the
            // field type is stored with `Param`s (`value: T`, `[T; N]`); substitute
            // the struct's args so the access yields the concrete field type. mono
            // later flattens both.
            if struct_args.is_empty() {
                field_ty
            } else {
                let mut type_subst: HashMap<&'a str, Type<'a>> = HashMap::new();
                let mut const_subst: HashMap<&'a str, ConstVal<'a>> = HashMap::new();
                if let Some(params) = cx.generic_structs.get(&struct_def) {
                    for (gp, ga) in params.iter().zip(struct_args.iter()) {
                        match (gp, ga) {
                            (GenericParam::Type { name: n, .. }, GenericArg::Type(t)) => { type_subst.insert(*n, t.clone()); }
                            (GenericParam::Const(n, _), GenericArg::Const(ConstVal::Lit(v))) => { const_subst.insert(*n, ConstVal::Lit(*v)); }
                            _ => {}
                        }
                    }
                }
                subst_param_type(cx, &type_subst, &const_subst, &field_ty)
            }
        },

        ExprNode::Call { func, type_args, args }
            if matches!(&func.value, ExprNode::Var(name)
                if Intrinsic::lookup(name).is_some()) => {
            let ExprNode::Var(name) = &func.value else { unreachable!() };
            let intrinsic = Intrinsic::lookup(name).unwrap();
            typecheck_intrinsic(cx, intrinsic, type_args, args, span, metadata.id)?
        },
        ExprNode::Call { func, type_args, args } => {
            // a receiver method call `recv.method(args)`: `func` is a field access
            // whose base is a struct/enum that has a method `Type$method`. Resolved
            // here (only the typechecker knows the receiver's type) into
            // `method_calls` for MIL lowering. When there's no such method this
            // falls through to the ordinary access-then-call path below (which
            // handles a function-pointer struct field called as `x.f()`).
            if let ExprNode::Access { base, field } = &func.value {
                let base_ty = infer(cx, base)?;
                // a bounded type-param receiver: `x.m(...)` where `x: T` (or
                // `*T`) and `T: SomeTrait`. Resolve through the bound and yield
                // the trait method's result type. The generic body itself is
                // never lowered - monomorphization substitutes `T` and the
                // concrete call re-resolves via the path below - so we only
                // typecheck here and record nothing in `method_calls`.
                if let Some(ret) = resolve_bounded_method(cx, &base_ty, field, args, &span)? {
                    // a trait declares no generic methods, so there is nothing a
                    // turbofish here could bind - and silently dropping it would
                    // typecheck a different call than the one written.
                    if !type_args.is_empty() {
                        return Err(Error::new(span, format!(
                            "trait method '{}' takes no generic arguments", field)));
                    }
                    cx.node_types.insert(metadata.id, ret.clone());
                    return Ok(ret);
                }
                // an associated fn (no `self`) is not callable as
                // `value.assoc()`, so it falls through rather than
                // misbinding. The member record says which it is outright;
                // this used to be inferred by checking whether the first
                // parameter looked like a `self` of the right type.
                let resolved = match receiver_member(cx, &base_ty, field) {
                    Some((m, u)) if m.receiver != Receiver::Associated => {
                        // a private method is reachable only from its own module,
                        // the same rule top-level functions follow (trait-impl
                        // methods are always public - see `Member::is_pub`).
                        if !m.is_pub && m.module != cx.current_module {
                            return Err(Error::new(span,
                                format!("method '{}' is private to its module", field))
                                .with_note("mark it `pub` to call it from another module"));
                        }
                        method_signature(cx, &m, &u, field, type_args, &span)?
                    }
                    _ => None,
                };
                if let Some((target, params, return_type)) = resolved {
                    // params[0] is the receiver `self`; args match the rest.
                    let arg_params = &params[1..];
                    if args.len() != arg_params.len() {
                        return Err(Error::new(span, format!("method '{}' expects {} argument(s), got {}",
                            field, arg_params.len(), args.len())));
                    }
                    for (param_ty, arg) in arg_params.iter().zip(args.iter()) {
                        check_expr(cx, param_ty, arg)?;
                    }

                    // The base must become a `self` of type `params[0]`. If it
                    // already has that type it passes straight through; if `self`
                    // is one pointer deeper, take its address. Comparing the exact
                    // types (rather than just "is either a pointer?") is what tells
                    // an auto-`->` receiver - `extend Point`'s `*self` reached via a
                    // `*Point` base, where base already *is* `*Point` - apart from
                    // an `extend *T` receiver, where the base *is* the `*T` self and
                    // the method's `*self` wants `**T`.
                    let want = &params[0];
                    let adjust = if &base_ty == want {
                        RecvAdjust::AsIs
                    } else if matches!(want, Type::Pointer(inner) if inner.as_ref() == &base_ty) {
                        RecvAdjust::AddrOf
                    } else {
                        RecvAdjust::AsIs
                    };
                    cx.method_calls.insert(metadata.id, MethodCall {
                        target, adjust, param_tys: params, return_type: return_type.clone(),
                    });
                    cx.node_types.insert(metadata.id, return_type.clone());
                    return Ok(return_type);
                }
            }

            // an associated call through a type parameter, `P::new(args)`:
            // dispatch through `P`'s trait bound. Tried before the enum path since
            // a two-segment path headed by a type param is never an enum variant.
            if let ExprNode::Path(cname) = &func.value
                && let Some(ret) = resolve_bounded_assoc(cx, cname, type_args, args, &span)? {
                    cx.node_types.insert(expr.id, ret.clone());
                    return Ok(ret);
                }

            // a data-enum constructor `E::V(args...)` looks like a call but names
            // no function; check arity + each arg against the payload field types
            // and yield the aggregate enum type. Guarded before ordinary dispatch.
            if let ExprNode::Path(cname) = &func.value
                && let Some((ename, payload_tys)) = enum_variant_ctor(cx, cname) {
                    return check_enum_call_ctor(cx, expr, cname, ename, &payload_tys, type_args, args, None, &span);
                }
            // user generic call (`foo::<T>(...)`) check against the generic sig
            // mono emits the instance later.
            // TODO: `.cloned()` copies the whole sig on every generic call site
            // (twice, since we typecheck again after mono) just to dodge the
            // borrow of cx
            if let ExprNode::Var(name) = &func.value
                && let Some(sig) = cx.generic_fns.get(*name).cloned() {
                    let name = *name;
                    let (ty, inferred) = check_generic_call(cx, name, &sig, type_args, args, &span)?;
                    // a bare call with inferred turbofish keeps an empty `type_args`
                    // on the AST node; record the recovered args so mono knows which
                    // instance to mint (it reads this alongside `node_types`).
                    if let Some(targs) = inferred {
                        cx.inferred_type_args.insert(expr.id, targs);
                    }
                    cx.node_types.insert(expr.id, ty.clone());
                    return Ok(ty);
                }
            if !type_args.is_empty() {
                // a turbofished name that reached here is neither a generic fn (the
                // `generic_fns` check above) nor an intrinsic (its own arm). Before
                // blaming the turbofish syntax, check whether the name resolves at
                // all: an unresolved one is almost always an undefined or, more
                // often, un-imported symbol (imports aren't re-exported), which the
                // syntax message hides
                if let ExprNode::Var(name) = &func.value
                    && cx.lookup(name).is_none() {
                        return Err(Error::new(span, format!("unknown function '{}'", name))
                            .with_note("is it defined and imported into this module?"));
                    }
                return Err(Error::new(span, "unexpected type arguments")
                    .with_note("`::<...>` is only valid on a generic or intrinsic call"));
            }
            let callee_ty = infer(cx, func)?;
            match callee_ty {
                Type::Function { params, return_type } => {
                    if params.len() != args.len() {
                        return Err(Error::new(span, format!(
                            "expected {} argument(s), got {}", params.len(), args.len())));
                    }

                    for (param_ty, arg_expr) in params.iter().zip(args.iter()) {
                        check_expr(cx, param_ty, arg_expr)?;
                    }

                    *return_type
                }
                _ => {
                    return Err(Error::new(span, "this is not callable")
                        .with_label(span, format!("expected a function, got `{}`",
                            cx.show(&callee_ty))));
                }
            }
        },
    };

    cx.node_types.insert(expr.id, ty.clone());
    Ok(ty)
}

/// Whether every path through `stmt` returns or diverges. Loops are treated
/// conservatively because there is no break analysis. Expressions typed `!`
/// also prevent fallthrough.
pub(crate) fn always_returns(stmt: &Stmt, node_types: &HashMap<usize, Type>) -> bool {
    match &stmt.value {
        StmtNode::Return(_) => true,
        // a bare expression of type `!` (currently only `abort(...)`) diverges, so
        // nothing after it in the block is reachable - it counts as a return.
        StmtNode::Expr(e) => matches!(node_types.get(&e.id), Some(Type::Never)),
        // a block returns if any statement in it returns (anything after the
        // first returning statement is dead, which is fine for this check)
        StmtNode::Block(stmts) => stmts.iter().any(|s| always_returns(s, node_types)),
        // an `if` guarantees a return only with an `else` where BOTH branches
        // return; a bare `if` falls through when the condition is false.
        StmtNode::If { then_branch, else_branch: Some(else_branch), .. } =>
            always_returns(then_branch, node_types)
            && always_returns(else_branch, node_types),
        // a match is exhaustive (typecheck guarantees it), so it returns on every
        // path iff every arm body does.
        StmtNode::Match { arms, .. } => !arms.is_empty()
            && arms.iter().all(|(_, body)| always_returns(body, node_types)),
        _ => false,
    }
}

pub(crate) fn check_stmt<'a>(
    cx: &mut Context<'a>,
    return_ty: &Type<'a>,
    stmt: &Stmt<'a>,
) -> Result<(), Error> {
    match &stmt.value {
        StmtNode::Expr(expr) => {
            infer(cx, expr)?;
        },

        StmtNode::Block(stmts) => {
            cx.push_scope();
            for stmt in stmts {
                check_stmt(cx, return_ty, stmt)?;
            }
            cx.pop_scope();
        },

        StmtNode::Declare { name, ty, value } => {
            let ty = match ty {
                // `let x: T = e;` - the annotation is the expectation `e` is
                // checked against, exactly as before.
                Some(ty) => {
                    if let Err(msg) = check_const_scope(&cx.const_generics, ty) {
                        return Err(Error::new(stmt.span, format!(
                            "in declaration of '{}': {}", name, msg)));
                    }
                    let ty = ty.clone();
                    check_expr(cx, &ty, value)?;
                    ty
                }
                // `let x = e;` - the initializer decides. There is nothing to
                // check it against, so a width-less literal in it defaults
                // (`let n = 0;` is an `i32`), and an expression that cannot be
                // typed on its own asks for the annotation back.
                None => infer(cx, value)
                    .map_err(|e| e.context(format!("cannot infer the type of '{}'", name)))?,
            };
            // a local has to have storage and a value; neither is true of `void`,
            // which is what a call to a procedure that returns nothing yields.
            if ty == Type::Void {
                return Err(Error::new(stmt.span, format!(
                    "'{}' cannot be declared with type void", name)));
            }
            // the local's binding identity is this Declare stmt's node id, which
            // is globally unique - so shadowed same-named locals stay distinct.
            // The type is recorded under that same id: for a bare `let` it is the
            // only record of what the binding turned out to be, and it is where
            // monomorphization reads the annotation it writes back into the AST.
            cx.node_types.insert(stmt.id, ty.clone());
            cx.insert(name, Some(Binding::Local(stmt.id)), ty);
        },

        StmtNode::Assign { left, value } => {
            // a `const` global is read-only: reject a direct `GLOBAL = ...`.
            // (mutating through a pointer/field is still the pointee's business.)
            if let ExprNode::Var(name) = &left.value
                && cx.global_consts.contains(name) {
                    return Err(Error::new(left.span, format!(
                        "cannot assign to constant global '{}'", name)));
                }
            let left_ty = infer(cx, left)?;
            check_expr(cx, &left_ty, value)?;
        },

        StmtNode::If { condition, then_branch, else_branch } => {
            check_expr(cx, &Type::Bool, condition)?;

            cx.push_scope();
            check_stmt(cx, return_ty, then_branch)?;
            cx.pop_scope();

            if let Some(else_branch) = else_branch {
                cx.push_scope();
                check_stmt(cx, return_ty, else_branch)?;
                cx.pop_scope();
            }
        },

        StmtNode::While { condition, body } => {
            check_expr(cx, &Type::Bool, condition)?;

            cx.push_scope();
            check_stmt(cx, return_ty, body)?;
            cx.pop_scope();
        },

        StmtNode::Match { scrutinee, arms } => {
            let scrut_ty = infer(cx, scrutinee)?;
            // the scrutinee must be an enum or an integer.
            let enum_name: Option<DefId> = match &scrut_ty {
                Type::Named { def, .. } if cx.enums.contains_key(def) => Some(*def),
                t if t.is_integer() => None,
                other => return Err(Error::new(scrutinee.span, format!(
                    "match scrutinee must be an enum or integer type, got {}", cx.show(other)))),
            };
            // `cx.enums[&en].payloads` holds the enum's OWN declared payload types
            // (`Type::Param("T")` for a generic enum's field) - if the scrutinee is
            // a concrete instance of a generic enum (`Option<i32>`, args non-empty),
            // substitute those params with the scrutinee's actual args before using
            // any payload type below, mirroring how struct field access substitutes
            // a generic struct's `Param` fields via its own `args`.
            let (type_subst, const_subst): (HashMap<&'a str, Type<'a>>, HashMap<&'a str, ConstVal<'a>>) =
                match &scrut_ty {
                    Type::Named { def, args } if !args.is_empty() => {
                        let mut ts = HashMap::new();
                        let mut cs = HashMap::new();
                        if let Some(params) = cx.generic_enums.get(def) {
                            for (gp, ga) in params.iter().zip(args.iter()) {
                                match (gp, ga) {
                                    (GenericParam::Type { name: n, .. }, GenericArg::Type(t)) => { ts.insert(*n, t.clone()); }
                                    (GenericParam::Const(n, _), GenericArg::Const(ConstVal::Lit(v))) => { cs.insert(*n, ConstVal::Lit(*v)); }
                                    _ => {}
                                }
                            }
                        }
                        (ts, cs)
                    }
                    _ => (HashMap::new(), HashMap::new()),
                };

            let mut has_wildcard = false;
            let mut covered_variants: HashSet<&'a str> = HashSet::new();
            let mut covered_ints: HashSet<i64> = HashSet::new();

            for (pat, body) in arms {
                if has_wildcard {
                    return Err(Error::new(pat.span, "unreachable match arm after `_`"));
                }
                // payload bindings this arm introduces into its own scope.
                let mut arm_bindings: Vec<(&'a str, Binding<'a>, Type<'a>)> = Vec::new();
                match &pat.value {
                    PatternNode::Wildcard => has_wildcard = true,
                    PatternNode::Int(n) => {
                        if let Some(en) = enum_name {
                            return Err(Error::new(pat.span, format!(
                                "integer pattern in a match on enum '{}'", cx.name_of(en))));
                        }
                        if !covered_ints.insert(*n) {
                            return Err(Error::new(pat.span, format!(
                                "duplicate match arm for `{}`", n)));
                        }
                    }
                    PatternNode::Path(p) => {
                        let Some(en) = enum_name else {
                            return Err(Error::new(pat.span, format!(
                                "enum-variant pattern `{}` in a match on integer type", p)));
                        };
                        let variant = check_variant_pattern(cx, en, p, &pat.span)?;
                        // a bare `E::V` on a data variant would leave the payload
                        // unbound - require the destructuring form `E::V(..)`.
                        let arity = cx.enums[&en].payloads.get(variant).map_or(0, |p| p.len());
                        if arity != 0 {
                            return Err(Error::new(pat.span, format!("variant `{}` carries {} field(s); destructure it as `{}(..)`", p, arity, p)));
                        }
                        if !covered_variants.insert(variant) {
                            return Err(Error::new(pat.span, format!(
                                "duplicate match arm for `{}`", p)));
                        }
                    }
                    PatternNode::Variant { path, fields } => {
                        let Some(en) = enum_name else {
                            return Err(Error::new(pat.span, format!(
                                "enum-variant pattern `{}` in a match on integer type", path)));
                        };
                        let variant = check_variant_pattern(cx, en, path, &pat.span)?;
                        let payload = cx.enums[&en].payloads.get(variant).cloned().unwrap_or_default();
                        if fields.len() != payload.len() {
                            return Err(Error::new(pat.span, format!(
                                "variant `{}` has {} field(s) but the pattern binds {}",
                                path, payload.len(), fields.len())));
                        }
                        if !covered_variants.insert(variant) {
                            return Err(Error::new(pat.span, format!(
                                "duplicate match arm for `{}`", path)));
                        }
                        for (fpat, (_, fty)) in fields.iter().zip(payload.iter()) {
                            match &fpat.value {
                                PatternNode::Wildcard => {}
                                // each `Bind` is keyed by its own node id (its
                                // binding identity), so shadowing/reuse is distinct.
                                PatternNode::Bind(bname) => arm_bindings.push((
                                    *bname, Binding::Local(fpat.id),
                                    subst_param_type(cx, &type_subst, &const_subst, fty),
                                )),
                                other => return Err(Error::new(fpat.span, format!(
                                    "unsupported payload sub-pattern `{}`", other))),
                            }
                        }
                    }
                    PatternNode::StructVariant { path, fields } => {
                        let Some(en) = enum_name else {
                            return Err(Error::new(pat.span, format!(
                                "enum-variant pattern `{}` in a match on integer type", path)));
                        };
                        let variant = check_variant_pattern(cx, en, path, &pat.span)?;
                        let payload = cx.enums[&en].payloads.get(variant).cloned().unwrap_or_default();
                        if payload.is_empty() {
                            return Err(Error::new(pat.span, format!(
                                "variant `{}` has no fields to destructure with `{{ }}`", path)));
                        }
                        if fields.len() != payload.len() {
                            return Err(Error::new(pat.span, format!(
                                "variant `{}` has {} field(s) but the pattern binds {}",
                                path, payload.len(), fields.len())));
                        }
                        if !covered_variants.insert(variant) {
                            return Err(Error::new(pat.span, format!(
                                "duplicate match arm for `{}`", path)));
                        }
                        // by-name: each named field must exist on the payload struct;
                        // a `Bind` view is keyed by its node id, a `_` ignores it.
                        let mut seen: HashSet<&str> = HashSet::new();
                        for (fname, fpat) in fields {
                            let Some((_, fty)) = payload.iter().find(|(n, _)| n == fname) else {
                                return Err(Error::new(fpat.span, format!(
                                    "variant `{}` has no field `{}`", path, fname)));
                            };
                            if !seen.insert(*fname) {
                                return Err(Error::new(fpat.span, format!(
                                    "field `{}` bound more than once in `{}`", fname, path)));
                            }
                            match &fpat.value {
                                PatternNode::Wildcard => {}
                                PatternNode::Bind(bname) => arm_bindings.push((
                                    *bname, Binding::Local(fpat.id),
                                    subst_param_type(cx, &type_subst, &const_subst, fty),
                                )),
                                other => return Err(Error::new(fpat.span, format!(
                                    "unsupported payload sub-pattern `{}`", other))),
                            }
                        }
                    }
                    PatternNode::Bind(name) => return Err(
                        Error::new(pat.span,
                            format!("bare binding `{}` is not a valid match pattern", name))
                            .with_note("bindings appear inside a variant destructure")),
                }
                cx.push_scope();
                for (bname, binding, bty) in &arm_bindings {
                    cx.insert(bname, Some(*binding), bty.clone());
                }
                check_stmt(cx, return_ty, body)?;
                cx.pop_scope();
            }

            // exhaustiveness: an enum needs every variant or a `_`; an integer
            // scrutinee always needs a `_` (its domain can't be enumerated).
            if !has_wildcard {
                match enum_name {
                    Some(en) => {
                        let mut missing: Vec<&str> = cx.enums[&en].variants.keys()
                            .filter(|v| !covered_variants.contains(**v)).cloned().collect();
                        if !missing.is_empty() {
                            missing.sort();
                            return Err(Error::new(stmt.span,
                                format!("non-exhaustive match on enum '{}'", cx.name_of(en)))
                                .with_label(stmt.span,
                                    format!("missing {}", missing.join(", ")))
                                .with_note("cover every variant, or add a `_` arm"));
                        }
                    }
                    None => return Err(Error::new(stmt.span, "non-exhaustive match on an integer type")
                        .with_note("an integer match needs a `_` arm")),
                }
            }
        },

        StmtNode::Continue | StmtNode::Break => {
            // nothing to check
        },

        StmtNode::Return(Some(expr)) => {
            check_expr(cx, return_ty, expr)?;
        },

        // `return;` carries nothing out, so it only means anything in a proc that
        // returns nothing. Anywhere else the value the caller is owed is missing.
        StmtNode::Return(None) => {
            if *return_ty != Type::Void {
                return Err(Error::new(stmt.span,
                    format!("this proc returns '{}', so `return` needs a value", return_ty))
                    .with_label(stmt.span, "expected `return <value>;`"));
            }
        },
    }

    Ok(())
}
