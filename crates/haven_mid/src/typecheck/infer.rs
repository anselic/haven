use std::collections::{HashMap, HashSet};
use haven_common::ast::*;
use crate::intrinsics::Intrinsic;
use super::context::{Context, MethodCall, RecvAdjust, enum_payload_struct_name};
use super::generics::{bind_generics, subst_param_type, check_generic_call, bind_struct_generics, resolve_type, check_const_scope, subst_self};
use super::enums::{enum_variant, split_enum_variant, enum_variant_ctor, check_variant_pattern};

/// Whether `expr` denotes a place (an addressable location) rather than a
/// temporary. Must stay in sync with what MIL's `lower_lvalue` can handle: a
/// variable, a field access, an index, or a pointer dereference. Everything else
/// (a call result, a literal, an arithmetic result, a struct/array literal) is a
/// temporary with no storage to take the address of.
fn is_place(expr: &Expr<'_>) -> bool {
    matches!(&expr.value,
        ExprNode::Var(_)
        | ExprNode::Access { .. }
        | ExprNode::Index { .. }
        | ExprNode::Unary { op: UnaryOp::Deref, .. })
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
        if let Some(def) = cx.traits.get(tr) {
            if let Some(m) = def.methods.get(field) {
                sig = Some((m.params.clone(), m.return_type.clone()));
                break;
            }
        }
    }
    let Some((params, return_type)) = sig else {
        return Err(Error {
            msg: format!(
                "no method '{}' on type parameter '{}'; add a trait bound that provides it \
                 (e.g. `<{}: SomeTrait>`)",
                field, param, param),
            span: span.clone(),
        });
    };

    // `Self` in the trait signature refers to the param type `T` here.
    let self_ty = Type::Param(param);
    if args.len() != params.len() {
        return Err(Error {
            msg: format!("method '{}' expects {} argument(s), got {}",
                field, params.len(), args.len()),
            span: span.clone(),
        });
    }
    for (pty, arg) in params.iter().zip(args) {
        let expected = subst_self(pty, &self_ty);
        check_expr(cx, &expected, arg)?;
    }
    Ok(Some(subst_self(&return_type, &self_ty)))
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
            if !value_ty.is_numeric() && !matches!(value_ty, Type::Enum { .. }) {
                return Err(Error {
                    msg: format!("numerical_cast() argument must be a numeric type, got {}", value_ty),
                    span,
                });
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
            if !matches!(value_ty, Type::Pointer(_)) {
                return Err(Error {
                    msg: format!("ptr_cast() argument must be a pointer, got {}", value_ty),
                    span,
                });
            }
            cx.node_types.insert(expr_id, target_ty.clone());
            Ok(target_ty)
        }
        Intrinsic::SimdSplat => {
            let ty = tys[0].clone();
            let size = consts[0].clone();

            let value_ty = infer(cx, &args[0])?;
            if value_ty != ty {
                return Err(Error {
                    msg: format!("simd_splat() value argument must be of the element type, got {}", value_ty),
                    span,
                });
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
                return Err(Error {
                    msg: format!("simd_load() offset argument must be an integer type, got {}", offset_ty),
                    span,
                });
            }

            match slice_ty {
                Type::Slice(inner) | Type::Pointer(inner) if *inner == ty => {
                    let simd_ty = Type::Simd(Box::new(ty), size);
                    cx.node_types.insert(expr_id, simd_ty.clone());
                    Ok(simd_ty)
                },
                _ => {
                    return Err(Error {
                        msg: format!("simd_load() first argument must be a slice or pointer to the element type, got {}", slice_ty),
                        span,
                    });
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
                return Err(Error {
                    msg: format!("simd_store() offset argument must be an integer type, got {}", offset_ty),
                    span,
                });
            }

            let expected_value_ty = Type::Simd(Box::new(ty.clone()), size);
            if value_ty != expected_value_ty {
                return Err(Error {
                    msg: format!("simd_store() value argument must be a SIMD vector of the element type and size, got {}", value_ty),
                    span,
                });
            }

            match slice_ty {
                Type::Slice(inner) | Type::Pointer(inner) if *inner == ty => Ok(Type::Void),
                _ => {
                    return Err(Error {
                        msg: format!("simd_store() first argument must be a slice or pointer to the element type, got {}", slice_ty),
                        span,
                    });
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
                return Err(Error {
                    msg: format!("simd_concat() first value argument must be a SIMD vector of the element type and half the size, got {}", value1_ty),
                    span,
                });
            }
            if !half_ok(&value2_ty) {
                return Err(Error {
                    msg: format!("simd_concat() second value argument must be a SIMD vector of the element type and half the size, got {}", value2_ty),
                    span,
                });
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
                    return Err(Error {
                        msg: format!("{}() value argument must be a SIMD vector of the element type and larger size, got {}", intrinsic, value_ty),
                        span,
                    });
                }
            }
        }
    }
}

pub(crate) fn check_expr<'a>(
    cx: &mut Context<'a>,
    expected: &Type<'a>,
    expr: &Expr<'a>,
) -> Result<(), Error> {
    let metadata = expr;
    let value = &metadata.value;
    let span = metadata.span.clone();

    let actual = match value {
        ExprNode::Bool(_)    => Type::Bool,
        ExprNode::Int8(_)    => Type::Int8,
        ExprNode::Int32(_)   => Type::Int32,
        ExprNode::Int64(_)   => Type::Int64,
        ExprNode::Uint8(_)   => Type::Uint8,
        ExprNode::Uint32(_)  => Type::Uint32,
        ExprNode::Uint64(_)  => Type::Uint64,
        ExprNode::Float32(_) => Type::Float32,
        ExprNode::Float64(_) => Type::Float64,
        ExprNode::Str(_)     => Type::Str,

        // let xs: []i32 = []; so the type of [] is i32
        // else, if [...] is populated, infer it
        ExprNode::Slice(inner) if inner.len() == 0 => {
            match expected {
                Type::Slice(elem_ty) => Type::Slice(elem_ty.clone()),
                _ => {
                    let msg = format!("Expected {expected}, got []");
                    return Err(Error {
                        msg,
                        span,
                    });
                }
            }
        },

        _ => infer(cx, expr)?,
    };

    let compatible = actual == *expected ||
        matches!((&actual, expected),
            (Type::Array(inner_actual, _), Type::Slice(inner_expected))
                if inner_actual == inner_expected
        );

    if !compatible {
        let msg = format!("Expected {expected}, got {actual}");
        return Err(Error {
            msg,
            span,
        });
    }

    cx.node_types.insert(metadata.id, actual);
    Ok(())
}

fn infer<'a>(
    cx: &mut Context<'a>,
    expr: &Expr<'a>,
) -> Result<Type<'a>, Error> {
    let metadata = expr;
    let value = &metadata.value;
    let span = metadata.span.clone();

    let ty = match value {
        ExprNode::Bool(_)    => Type::Bool,
        ExprNode::Int8(_)    => Type::Int8,
        ExprNode::Int32(_)   => Type::Int32,
        ExprNode::Int64(_)   => Type::Int64,
        ExprNode::Uint8(_)   => Type::Uint8,
        ExprNode::Uint32(_)  => Type::Uint32,
        ExprNode::Uint64(_)  => Type::Uint64,
        ExprNode::Float32(_) => Type::Float32,
        ExprNode::Float64(_) => Type::Float64,
        ExprNode::Str(_)     => Type::Str,

        ExprNode::Var(name) => {
            // an `Enum::Variant` reference. A unit variant is a value: a field-less
            // enum's scalar discriminant, or (for a data enum) an aggregate with no
            // payload. A tuple/struct variant used bare is a missing constructor
            // call - `Msg::Note` needs `Msg::Note(...)`.
            if let Some((ename, _val, repr)) = enum_variant(cx, name) {
                let variant = name.split_once("::").unwrap().1;
                if cx.enums[ename].payloads.get(variant).is_some_and(|p| !p.is_empty()) {
                    return Err(Error {
                        msg: format!("variant '{}' carries a payload; construct it with `{}(...)`", name, name),
                        span,
                    });
                }
                // a generic enum's variant can never be used bare - a bare `Var`
                // has no syntax to attach a turbofish, so even a unit variant needs
                // the call form: `Option::None::<i32>()`.
                if cx.generic_enums.contains_key(ename) {
                    return Err(Error {
                        msg: format!(
                            "enum '{}' is generic; construct '{}' with explicit type arguments, e.g. `{}::<...>()`",
                            ename, name, name,
                        ),
                        span,
                    });
                }
                Type::Enum { name: ename, repr: Box::new(repr), has_payload: cx.enums[ename].has_payload, args: Vec::new() }
            } else {
                let Some((binding, ty)) = cx.lookup(name) else {
                    let msg = format!("Undefined variable '{}'", name);
                    return Err(Error {
                        msg,
                        span,
                    });
                };
                let binding = *binding;
                let ty = ty.clone();
                // record which param/local this use resolves to (globals -> None)
                if let Some(b) = binding {
                    cx.resolved.insert(metadata.id, b);
                }
                ty
            }
        },

        ExprNode::Slice(inner) if inner.len() == 0 => {
            return Err(Error {
                msg: "Cannot infer type of empty slice literal".to_string(),
                span,
            });
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
                let msg = format!(
                    "Expected index of type i32, got {}",
                    index_ty
                );
                return Err(Error {
                    msg,
                    span,
                });
            }
            match slice_ty {
                Type::Slice(inner)
                | Type::Array(inner, _)
                | Type::Pointer(inner) => *inner,
                _ => {
                    let msg = format!(
                        "Expected a slice or pointer type for indexing, got {}",
                        slice_ty
                    );
                    return Err(Error {
                        msg,
                        span,
                    });
                }
            }
        },

        ExprNode::Unary { op, operand } => {
            let operand_ty = infer(cx, operand)?;
            match op {
                UnaryOp::AddrOf => {
                    // `&` needs a place to point at. Taking the address of a
                    // temporary (a call result, literal, arithmetic, ...) has no
                    // storage; reject it here rather than panicking in lowering,
                    // which only knows how to address real places.
                    if !is_place(operand) {
                        return Err(Error {
                            msg: format!(
                                "cannot take the address of a temporary value `{}`; \
                                 bind it to a `let` first, then take a pointer to that",
                                operand.value),
                            span,
                        });
                    }
                    Type::Pointer(Box::new(operand_ty))
                },
                UnaryOp::Deref => match operand_ty {
                    Type::Pointer(inner) => *inner,
                    _ => {
                        let msg = format!(
                            "Expected a pointer type for dereference, got {}",
                            operand_ty
                        );
                        return Err(Error {
                            msg,
                            span,
                        });
                    }
                },
                UnaryOp::Neg => if operand_ty.is_numeric() {
                    operand_ty
                } else {
                    let msg = format!(
                        "Expected a numeric type for negation, got {}",
                        operand_ty
                    );
                    return Err(Error {
                        msg,
                        span,
                    });
                },
                UnaryOp::Not => Type::Bool,
            }
        },

        ExprNode::Binary { op, left, right } => {
            use haven_common::ast::BinaryOp::*;
            match op {
                Eq | Ne => {
                    let left_ty = infer(cx, left)?;
                    check_expr(cx, &left_ty, right)?;
                    Type::Bool
                },
                Add | Sub | Mul | Div | Mod
                | Lt | Gt | Le | Ge => {
                    let left_ty = infer(cx, left)?;
                    check_expr(cx, &left_ty, right)?;

                    // check if both are numeric (scalar or SIMD)
                    if !left_ty.is_numeric_or_numeric_simd() {
                        let msg = format!(
                            "Expected a numeric type for binary operator, got {}",
                            left_ty
                        );
                        return Err(Error { msg, span });
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
                    let left_ty = infer(cx, left)?;
                    check_expr(cx, &left_ty, right)?;
                    if !left_ty.is_integer() {
                        let msg = format!(
                            "Expected an integer type for bitwise operator '{}', got {}",
                            op, left_ty
                        );
                        return Err(Error { msg, span });
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
                // a generic enum's struct-style variant needs turbofish (no context
                // inference yet, matching plain generic-struct construction); a
                // non-generic one must NOT have any.
                let (type_subst, const_subst, resolved_args) = match cx.generic_enums.get(ename).cloned() {
                    Some(params) => {
                        if type_args.is_empty() {
                            return Err(Error {
                                msg: format!(
                                    "enum '{}' is generic; construct '{}' with type arguments, e.g. `{}::<...> {{ ... }}`",
                                    ename, name, name,
                                ),
                                span,
                            });
                        }
                        bind_struct_generics(cx, ename, &params, type_args, &span)?
                    }
                    None => {
                        if !type_args.is_empty() {
                            return Err(Error {
                                msg: format!("enum constructor '{}' takes no type arguments", name),
                                span,
                            });
                        }
                        (HashMap::new(), HashMap::new(), Vec::new())
                    }
                };
                let pstruct = enum_payload_struct_name(ename, variant);
                let pdef = match cx.structs.get(pstruct) {
                    Some(d) if !d.is_empty() => d.clone(),
                    _ => return Err(Error {
                        msg: format!("variant '{}' has no fields; construct it as `{}`", name, name),
                        span,
                    }),
                };
                // a tuple variant's fields are named "0", "1", ...; those can't be
                // written in a `{ }` literal, so point the user at the `( )` form.
                if pdef.first().is_some_and(|(n, _)| n.bytes().all(|b| b.is_ascii_digit())) {
                    return Err(Error {
                        msg: format!("variant '{}' is a tuple variant; construct it with `{}(...)`", name, name),
                        span,
                    });
                }
                if fields.len() != pdef.len() {
                    return Err(Error {
                        msg: format!("variant '{}' expects {} field(s), got {}",
                            name, pdef.len(), fields.len()),
                        span,
                    });
                }
                // field order must match the declaration (same rule as a struct
                // literal); each field checks against its (param-substituted)
                // declared payload type.
                for ((def_name, def_ty), (lit_name, lit_value)) in pdef.iter().zip(fields.iter()) {
                    if def_name != lit_name {
                        return Err(Error {
                            msg: format!("In variant '{}': expected field '{}', got '{}'",
                                name, def_name, lit_name),
                            span: lit_value.span.clone(),
                        });
                    }
                    let expected = subst_param_type(&type_subst, &const_subst, def_ty);
                    check_expr(cx, &expected, lit_value)?;
                }
                let repr = cx.enums[ename].repr.clone();
                let ty = Type::Enum { name: ename, repr: Box::new(repr), has_payload: true, args: resolved_args };
                cx.node_types.insert(expr.id, ty.clone());
                return Ok(ty);
            }

            let def = match cx.structs.get(name) {
                Some(d) => d.clone(),
                None => return Err(Error {
                    msg: format!("Unknown struct '{}'", name),
                    span,
                }),
            };

            // Bind the struct's params to the turbofish args so `Param` field types
            // check against a concrete type (and `[T; N]` sizes against a concrete
            // count). Non-generic structs take no args; a generic struct requires
            // them (no context inference yet). `struct_args` are the resolved args
            // in declaration order.
            let (type_subst, const_subst, struct_args) = match cx.generic_structs.get(name).cloned() {
                Some(params) => {
                    if type_args.is_empty() {
                        return Err(Error {
                            msg: format!(
                                "constructing generic struct '{}' needs type arguments, e.g. `{}::<...> {{ ... }}`",
                                name, name,
                            ),
                            span,
                        });
                    }
                    bind_struct_generics(cx, name, &params, type_args, &span)?
                }
                None => {
                    if !type_args.is_empty() {
                        return Err(Error {
                            msg: format!(
                                "struct '{}' is not generic; no type arguments expected",
                                name,
                            ),
                            span,
                        });
                    }
                    (HashMap::new(), HashMap::new(), Vec::new())
                }
            };

            if fields.len() != def.len() {
                return Err(Error {
                    msg: format!(
                        "Struct '{}' expects {} fields, got {}",
                        name, def.len(), fields.len()
                    ),
                    span,
                });
            }

            // field order must match definition; each field checks against its
            // (param-substituted) declared type.
            for ((def_name, def_ty), (lit_name, lit_value)) in def.iter().zip(fields.iter()) {
                if def_name != lit_name {
                    return Err(Error {
                        msg: format!(
                            "In struct '{}': expected field '{}', got '{}'",
                            name, def_name, lit_name
                        ),
                        span: lit_value.span.clone(),
                    });
                }
                let expected = subst_param_type(&type_subst, &const_subst, def_ty);
                check_expr(cx, &expected, lit_value)?;
            }

            // The literal's type carries its concrete args (`Option<i32>`,
            // `Buf<i32, 8>`); monomorphization rewrites both this literal and the
            // type to the flat instance before codegen. Non-generic structs get the
            // usual no-args `Struct`.
            Type::Struct { name, args: struct_args }
        },

        ExprNode::Access { base, field } => {
            let base_ty = infer(cx, base)?;
            let (struct_name, struct_args) = match &base_ty {
                Type::Struct { name, args } => (*name, args.as_slice()),
                // auto-deref one level of pointer to a struct (like C's `->`)
                Type::Pointer(inner) => match inner.as_ref() {
                    Type::Struct { name, args } => (*name, args.as_slice()),
                    _ => return Err(Error {
                        msg: format!("Cannot access field '{}' on type {}", field, base_ty),
                        span,
                    }),
                },
                _ => return Err(Error {
                    msg: format!("Cannot access field '{}' on type {}", field, base_ty),
                    span,
                }),
            };

            let def = match cx.structs.get(struct_name) {
                Some(d) => d,
                None => return Err(Error {
                    msg: format!("Unknown struct '{}'", struct_name),
                    span,
                }),
            };

            let field_ty = match def.iter().find(|(n, _)| n == field) {
                Some((_, ty)) => ty.clone(),
                None => return Err(Error {
                    msg: format!("Struct '{}' has no field '{}'", struct_name, field),
                    span,
                }),
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
                if let Some(params) = cx.generic_structs.get(struct_name) {
                    for (gp, ga) in params.iter().zip(struct_args.iter()) {
                        match (gp, ga) {
                            (GenericParam::Type { name: n, .. }, GenericArg::Type(t)) => { type_subst.insert(*n, t.clone()); }
                            (GenericParam::Const(n, _), GenericArg::Const(ConstVal::Lit(v))) => { const_subst.insert(*n, ConstVal::Lit(*v)); }
                            _ => {}
                        }
                    }
                }
                subst_param_type(&type_subst, &const_subst, &field_ty)
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
            if type_args.is_empty() {
                if let ExprNode::Access { base, field } = &func.value {
                    let base_ty = infer(cx, base)?;
                    // a bounded type-param receiver: `x.m(...)` where `x: T` (or
                    // `*T`) and `T: SomeTrait`. Resolve through the bound and yield
                    // the trait method's result type. The generic body itself is
                    // never lowered - monomorphization substitutes `T` and the
                    // concrete call re-resolves via the path below - so we only
                    // typecheck here and record nothing in `method_calls`.
                    if let Some(ret) = resolve_bounded_method(cx, &base_ty, field, args, &span)? {
                        cx.node_types.insert(metadata.id, ret.clone());
                        return Ok(ret);
                    }
                    let recv_name = match &base_ty {
                        Type::Struct { name, .. } | Type::Enum { name, .. } => Some(*name),
                        Type::Pointer(inner) => match inner.as_ref() {
                            Type::Struct { name, .. } | Type::Enum { name, .. } => Some(*name),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(recv_name) = recv_name {
                        // an associated fn (no `self`) is not callable as
                        // `value.assoc()`, so it falls through rather than
                        // misbinding. The member record says which it is outright;
                        // this used to be inferred by checking whether the first
                        // parameter looked like a `self` of the right type.
                        let resolved = match cx.members.get(&(recv_name, field)) {
                            Some(m) if m.receiver != Receiver::Associated => {
                                match cx.lookup(m.name) {
                                    Some((_, Type::Function { params, return_type })) =>
                                        Some((m.name, params.clone(), (**return_type).clone())),
                                    _ => None,
                                }
                            }
                            _ => None,
                        };
                        if let Some((target, params, return_type)) = resolved {
                            let base_is_ptr = matches!(base_ty, Type::Pointer(_));

                            // params[0] is the receiver `self`; args match the rest.
                            let arg_params = &params[1..];
                            if args.len() != arg_params.len() {
                                return Err(Error {
                                    msg: format!("method '{}' expects {} argument(s), got {}",
                                        field, arg_params.len(), args.len()),
                                    span,
                                });
                            }
                            for (param_ty, arg) in arg_params.iter().zip(args.iter()) {
                                check_expr(cx, param_ty, arg)?;
                            }

                            // a `*self` method called on a value takes its address;
                            // otherwise the base passes straight through (a `*self`
                            // on a `*T`, or a value receiver whose aggregate is
                            // already handled by pointer).
                            let self_is_ptr = matches!(params[0], Type::Pointer(_));
                            let adjust = if self_is_ptr && !base_is_ptr {
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
                }
            }

            // a data-enum constructor `E::V(args...)` looks like a call but names
            // no function; check arity + each arg against the payload field types
            // and yield the aggregate enum type. Guarded before ordinary dispatch.
            if let ExprNode::Var(cname) = &func.value {
                if let Some((ename, payload_tys)) = enum_variant_ctor(cx, cname) {
                    // a generic enum's tuple/unit variant needs turbofish (no
                    // context inference yet, matching plain generic-struct
                    // construction); a non-generic one must NOT have any.
                    let (type_subst, const_subst, resolved_args) = match cx.generic_enums.get(ename).cloned() {
                        Some(params) => {
                            if type_args.is_empty() {
                                return Err(Error {
                                    msg: format!(
                                        "enum '{}' is generic; construct '{}' with type arguments, e.g. `{}::<...>(...)`",
                                        ename, cname, cname,
                                    ),
                                    span,
                                });
                            }
                            bind_struct_generics(cx, ename, &params, type_args, &span)?
                        }
                        None => {
                            if !type_args.is_empty() {
                                return Err(Error {
                                    msg: format!("enum constructor '{}' takes no type arguments", cname),
                                    span,
                                });
                            }
                            (HashMap::new(), HashMap::new(), Vec::new())
                        }
                    };
                    if args.len() != payload_tys.len() {
                        return Err(Error {
                            msg: format!("variant '{}' expects {} field(s), got {}",
                                cname, payload_tys.len(), args.len()),
                            span,
                        });
                    }
                    for (pty, arg) in payload_tys.iter().zip(args.iter()) {
                        let expected = subst_param_type(&type_subst, &const_subst, pty);
                        check_expr(cx, &expected, arg)?;
                    }
                    let repr = cx.enums[ename].repr.clone();
                    let ty = Type::Enum { name: ename, repr: Box::new(repr), has_payload: cx.enums[ename].has_payload, args: resolved_args };
                    cx.node_types.insert(expr.id, ty.clone());
                    return Ok(ty);
                }
            }
            // user generic call (`foo::<T>(...)`) check against the generic sig
            // mono emits the instance later.
            // TODO: `.cloned()` copies the whole sig on every generic call site
            // (twice, since we typecheck again after mono) just to dodge the
            // borrow of cx
            if let ExprNode::Var(name) = &func.value {
                if let Some(sig) = cx.generic_fns.get(*name).cloned() {
                    let name = *name;
                    let ty = check_generic_call(cx, name, &sig, type_args, args, &span)?;
                    cx.node_types.insert(expr.id, ty.clone());
                    return Ok(ty);
                }
            }
            if !type_args.is_empty() {
                // a turbofished name that reached here is neither a generic fn (the
                // `generic_fns` check above) nor an intrinsic (its own arm). Before
                // blaming the turbofish syntax, check whether the name resolves at
                // all: an unresolved one is almost always an undefined or, more
                // often, un-imported symbol (imports aren't re-exported), which the
                // syntax message hides
                if let ExprNode::Var(name) = &func.value {
                    if cx.lookup(name).is_none() {
                        return Err(Error {
                            msg: format!(
                                "unknown function '{}', is it defined and imported into this module?",
                                name,
                            ),
                            span,
                        });
                    }
                }
                return Err(Error {
                    msg: "type arguments (`::<...>`) are only valid on generic or intrinsic calls".into(),
                    span,
                });
            }
            let callee_ty = infer(cx, func)?;
            match callee_ty {
                Type::Function { params, return_type } => {
                    if params.len() != args.len() {
                        let msg = format!(
                            "Expected {} arguments, got {}",
                            params.len(),
                            args.len()
                        );
                        return Err(Error {
                            msg,
                            span,
                        });
                    }

                    for (param_ty, arg_expr) in params.iter().zip(args.iter()) {
                        check_expr(cx, param_ty, arg_expr)?;
                    }

                    *return_type
                }
                _ => {
                    let msg = format!("Expected a function, got {}", callee_ty);
                    return Err(Error {
                        msg,
                        span,
                    });
                }
            }
        },
    };

    cx.node_types.insert(expr.id, ty.clone());
    Ok(ty)
}

/// Whether executing `stmt` guarantees control flow never falls through to the
/// statement after it - i.e. it returns (or diverges) on every path. Used to
/// verify that a non-void function cannot reach the end of its body without
/// returning a value.
///
/// Conservative on loops: a `while` is assumed to possibly run zero times, so it
/// never guarantees a return (not even `while (true)`, since there's no
/// break analysis), and `break`/`continue` count as fall-through.
pub(crate) fn always_returns(stmt: &Stmt) -> bool {
    match &stmt.value {
        StmtNode::Return(_) => true,
        // a block returns if any statement in it returns (anything after the
        // first returning statement is dead, which is fine for this check)
        StmtNode::Block(stmts) => stmts.iter().any(always_returns),
        // an `if` guarantees a return only with an `else` where BOTH branches
        // return; a bare `if` falls through when the condition is false.
        StmtNode::If { then_branch, else_branch, .. } => match else_branch {
            Some(else_branch) => always_returns(then_branch) && always_returns(else_branch),
            None => false,
        },
        // a match is exhaustive (typecheck guarantees it), so it returns on every
        // path iff every arm body does.
        StmtNode::Match { arms, .. } => !arms.is_empty() && arms.iter().all(|(_, body)| always_returns(body)),
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
            if let Err(msg) = check_const_scope(&cx.const_generics, ty) {
                return Err(Error { msg: format!("in declaration of '{}': {}", name, msg), span: stmt.span.clone() });
            }
            let ty = resolve_type(&cx.generics, &cx.enums, ty);
            check_expr(cx, &ty, value)?;
            // the local's binding identity is this Declare stmt's node id, which
            // is globally unique - so shadowed same-named locals stay distinct.
            cx.insert(name, Some(Binding::Local(stmt.id)), ty);
        },

        StmtNode::Assign { left, value } => {
            // a `const` global is read-only: reject a direct `GLOBAL = ...`.
            // (mutating through a pointer/field is still the pointee's business.)
            if let ExprNode::Var(name) = &left.value {
                if cx.global_consts.contains(name) {
                    return Err(Error {
                        msg: format!("cannot assign to constant global '{}'", name),
                        span: left.span.clone(),
                    });
                }
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
            let enum_name: Option<&'a str> = match &scrut_ty {
                Type::Enum { name, .. } => Some(name),
                t if t.is_integer() => None,
                other => return Err(Error {
                    msg: format!("match scrutinee must be an enum or integer type, got {}", other),
                    span: scrutinee.span.clone(),
                }),
            };
            // `cx.enums[en].payloads` holds the enum's OWN declared payload types
            // (`Type::Param("T")` for a generic enum's field) - if the scrutinee is
            // a concrete instance of a generic enum (`Option<i32>`, args non-empty),
            // substitute those params with the scrutinee's actual args before using
            // any payload type below, mirroring how struct field access substitutes
            // a generic struct's `Param` fields via its own `args`.
            let (type_subst, const_subst): (HashMap<&'a str, Type<'a>>, HashMap<&'a str, ConstVal<'a>>) =
                match &scrut_ty {
                    Type::Enum { name, args, .. } if !args.is_empty() => {
                        let mut ts = HashMap::new();
                        let mut cs = HashMap::new();
                        if let Some(params) = cx.generic_enums.get(name) {
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
                    return Err(Error { msg: "unreachable match arm after `_`".into(), span: pat.span.clone() });
                }
                // payload bindings this arm introduces into its own scope.
                let mut arm_bindings: Vec<(&'a str, Binding<'a>, Type<'a>)> = Vec::new();
                match &pat.value {
                    PatternNode::Wildcard => has_wildcard = true,
                    PatternNode::Int(n) => {
                        if let Some(en) = enum_name {
                            return Err(Error { msg: format!("integer pattern in a match on enum '{}'", en), span: pat.span.clone() });
                        }
                        if !covered_ints.insert(*n) {
                            return Err(Error { msg: format!("duplicate match arm for `{}`", n), span: pat.span.clone() });
                        }
                    }
                    PatternNode::Path(p) => {
                        let Some(en) = enum_name else {
                            return Err(Error { msg: format!("enum-variant pattern `{}` in a match on integer type", p), span: pat.span.clone() });
                        };
                        let variant = check_variant_pattern(cx, en, *p, &pat.span)?;
                        // a bare `E::V` on a data variant would leave the payload
                        // unbound - require the destructuring form `E::V(..)`.
                        let arity = cx.enums[en].payloads.get(variant).map_or(0, |p| p.len());
                        if arity != 0 {
                            return Err(Error {
                                msg: format!("variant `{}` carries {} field(s); destructure it as `{}(..)`", p, arity, p),
                                span: pat.span.clone(),
                            });
                        }
                        if !covered_variants.insert(variant) {
                            return Err(Error { msg: format!("duplicate match arm for `{}`", p), span: pat.span.clone() });
                        }
                    }
                    PatternNode::Variant { path, fields } => {
                        let Some(en) = enum_name else {
                            return Err(Error { msg: format!("enum-variant pattern `{}` in a match on integer type", path), span: pat.span.clone() });
                        };
                        let variant = check_variant_pattern(cx, en, *path, &pat.span)?;
                        let payload = cx.enums[en].payloads.get(variant).cloned().unwrap_or_default();
                        if fields.len() != payload.len() {
                            return Err(Error {
                                msg: format!("variant `{}` has {} field(s) but the pattern binds {}",
                                    path, payload.len(), fields.len()),
                                span: pat.span.clone(),
                            });
                        }
                        if !covered_variants.insert(variant) {
                            return Err(Error { msg: format!("duplicate match arm for `{}`", path), span: pat.span.clone() });
                        }
                        for (fpat, (_, fty)) in fields.iter().zip(payload.iter()) {
                            match &fpat.value {
                                PatternNode::Wildcard => {}
                                // each `Bind` is keyed by its own node id (its
                                // binding identity), so shadowing/reuse is distinct.
                                PatternNode::Bind(bname) => arm_bindings.push((
                                    *bname, Binding::Local(fpat.id),
                                    subst_param_type(&type_subst, &const_subst, fty),
                                )),
                                other => return Err(Error {
                                    msg: format!("unsupported payload sub-pattern `{}`", other),
                                    span: fpat.span.clone(),
                                }),
                            }
                        }
                    }
                    PatternNode::StructVariant { path, fields } => {
                        let Some(en) = enum_name else {
                            return Err(Error { msg: format!("enum-variant pattern `{}` in a match on integer type", path), span: pat.span.clone() });
                        };
                        let variant = check_variant_pattern(cx, en, *path, &pat.span)?;
                        let payload = cx.enums[en].payloads.get(variant).cloned().unwrap_or_default();
                        if payload.is_empty() {
                            return Err(Error {
                                msg: format!("variant `{}` has no fields to destructure with `{{ }}`", path),
                                span: pat.span.clone(),
                            });
                        }
                        if fields.len() != payload.len() {
                            return Err(Error {
                                msg: format!("variant `{}` has {} field(s) but the pattern binds {}",
                                    path, payload.len(), fields.len()),
                                span: pat.span.clone(),
                            });
                        }
                        if !covered_variants.insert(variant) {
                            return Err(Error { msg: format!("duplicate match arm for `{}`", path), span: pat.span.clone() });
                        }
                        // by-name: each named field must exist on the payload struct;
                        // a `Bind` view is keyed by its node id, a `_` ignores it.
                        let mut seen: HashSet<&str> = HashSet::new();
                        for (fname, fpat) in fields {
                            let Some((_, fty)) = payload.iter().find(|(n, _)| n == fname) else {
                                return Err(Error {
                                    msg: format!("variant `{}` has no field `{}`", path, fname),
                                    span: fpat.span.clone(),
                                });
                            };
                            if !seen.insert(*fname) {
                                return Err(Error {
                                    msg: format!("field `{}` bound more than once in `{}`", fname, path),
                                    span: fpat.span.clone(),
                                });
                            }
                            match &fpat.value {
                                PatternNode::Wildcard => {}
                                PatternNode::Bind(bname) => arm_bindings.push((
                                    *bname, Binding::Local(fpat.id),
                                    subst_param_type(&type_subst, &const_subst, fty),
                                )),
                                other => return Err(Error {
                                    msg: format!("unsupported payload sub-pattern `{}`", other),
                                    span: fpat.span.clone(),
                                }),
                            }
                        }
                    }
                    PatternNode::Bind(name) => return Err(Error {
                        msg: format!("bare binding `{}` is not a valid match pattern; bindings appear inside a variant destructure", name),
                        span: pat.span.clone(),
                    }),
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
                        let mut missing: Vec<&str> = cx.enums[en].variants.keys()
                            .filter(|v| !covered_variants.contains(**v)).cloned().collect();
                        if !missing.is_empty() {
                            missing.sort();
                            return Err(Error {
                                msg: format!("non-exhaustive match on enum '{}': missing {}; cover them or add a `_` arm",
                                    en, missing.join(", ")),
                                span: stmt.span.clone(),
                            });
                        }
                    }
                    None => return Err(Error {
                        msg: "non-exhaustive match on an integer type: add a `_` arm".into(),
                        span: stmt.span.clone(),
                    }),
                }
            }
        },

        StmtNode::Continue | StmtNode::Break => {
            // nothing to check
        },

        StmtNode::Return(expr) => {
            check_expr(cx, return_ty, expr)?;
        },
    }

    Ok(())
}
