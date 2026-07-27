use haven_common::ast::*;
use crate::intrinsics::Intrinsic;
use crate::typecheck::RecvAdjust;
use super::ir::*;
use super::ctx::{LowerCtx, coerce, aggregate_struct_name, enum_const, ta_type, ta_const};

fn lower_intrinsic<'a>(
    cx: &mut LowerCtx<'a>,
    intrinsic: Intrinsic,
    type_args: &[GenericArg<'a>],
    args: &[Expr<'a>],
) -> Value {
    match intrinsic {
        Intrinsic::Null => Value::Const(Const::Null),
        // Intrinsic::Len => {
        //     let arg_val = lower_expr(cx, &args[0]);
        //     if matches!(cx.node_types[&args[0].id], Type::Str) {
        //         // `str` is a raw NUL-terminated C string with no carried length,
        //         // so recover it at runtime with libc `strlen` (returns i64) and
        //         // narrow to the i32 that `len()` is typed as.
        //         let raw = cx.fresh_reg();
        //         cx.emit(Inst::Call {
        //             dst: Some(raw),
        //             callee: Callee::Direct("strlen"),
        //             args: vec![(arg_val, Type::Pointer(Box::new(Type::Uint8)))],
        //             return_type: Type::Uint64,
        //             sret: None,
        //         });
        //         let dst = cx.fresh_reg();
        //         cx.emit(Inst::Extend {
        //             dst,
        //             val: Value::Reg(raw),
        //             from_ty: Type::Uint64,
        //             to_ty: Type::Int32,
        //         });
        //         Value::Reg(dst)
        //     } else {
        //         // slices/arrays carry an explicit i32 length in field 1.
        //         let dst = cx.fresh_reg();
        //         cx.emit(Inst::ExtractValue { dst, val: arg_val, index: 1 });
        //         Value::Reg(dst)
        //     }
        // }
        Intrinsic::NumericalCast => {
            // numerical_cast::<T>(value). An enum on either side casts as its
            // integer discriminant repr, so unwrap it before selecting the cast.
            let unwrap_enum = |t: Type<'a>| match t {
                Type::Enum { repr, .. } => *repr,
                other => other,
            };
            let val = lower_expr(cx, &args[0]);
            let from_ty = unwrap_enum(cx.node_types[&args[0].id].clone());
            let to_ty = unwrap_enum(ta_type(type_args, 0));
            let dst = cx.fresh_reg();
            cx.emit(Inst::Extend { dst, val, from_ty, to_ty });
            Value::Reg(dst)
        }
        Intrinsic::Sizeof => {
            // sizeof::<T>(); the type is taken directly from the turbofish.
            let ty = ta_type(type_args, 0);
            let dst = cx.fresh_reg();
            cx.emit(Inst::Sizeof { dst, ty });
            Value::Reg(dst)
        }
        Intrinsic::PtrCast => {
            // ptr_cast::<*T>(p) is a no-op under opaque pointers: the value is
            // already a `ptr`, only its static pointee type changes. Pass it
            // through; the result's type is tracked in node_types by typecheck.
            lower_expr(cx, &args[0])
        }
        Intrinsic::SimdSplat => {
            let ty = ta_type(type_args, 0);
            let size = ta_const(type_args, 1);
            let value_val = lower_expr(cx, &args[0]);

            let v0 = cx.fresh_reg();
            cx.emit(Inst::Splat { dst: v0, val: value_val, ty: ty.clone(), size });
            let dst = cx.fresh_reg();
            cx.emit(Inst::Shuffle {
                dst,
                value_size: size,
                v0: Value::Reg(v0),
                v1: Value::Const(Const::Undef),
                ty, size,
                mask: vec![0; size],
            });
            Value::Reg(dst)
        }
        Intrinsic::SimdLoad => {
            // simd_load::<T, N>(slice, offset) -> simd[T, N]
            let ty = ta_type(type_args, 0);
            let size = ta_const(type_args, 1);
            let slice_val = lower_expr(cx, &args[0]);
            let offset_val = lower_expr(cx, &args[1]);

            cx.emit(Inst::Comment(format!("simd_load")));
            // extract the data pointer from the fat pointer struct, or use directly if it's already a pointer
            let data_ptr = match cx.node_types[&args[0].id] {
                Type::Slice(_) => {
                    let extracted = cx.fresh_reg();
                    cx.emit(Inst::ExtractValue { dst: extracted, val: slice_val.clone(), index: 0 });
                    extracted
                }
                Type::Pointer(_) | Type::Array(_, _) => {
                    // already a raw pointer, use it directly
                    match slice_val {
                        Value::Reg(r) => r,
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };
            // get the element pointer with the offset
            let elem_ptr = cx.fresh_reg();
            cx.emit(Inst::Index { dst: elem_ptr, slice: data_ptr, index: offset_val, index_ty: cx.node_types[&args[1].id].clone(), element_ty: ty.clone() });
            // load the SIMD vector from the element pointer
            let dst = cx.fresh_reg();
            cx.emit(Inst::Load { dst, ptr: elem_ptr, ty: Type::Simd(Box::new(ty), ConstVal::Lit(size)), align: None });
            Value::Reg(dst)
        }
        Intrinsic::SimdStore => {
            // simd_store::<T, N>(slice, offset, value) -> ()
            let ty = ta_type(type_args, 0);
            let size = ta_const(type_args, 1);
            let slice_val = lower_expr(cx, &args[0]);
            let offset_val = lower_expr(cx, &args[1]);
            let value_val = lower_expr(cx, &args[2]);

            // like simd_load
            let data_ptr = match cx.node_types[&args[0].id] {
                Type::Slice(_) => {
                    let extracted = cx.fresh_reg();
                    cx.emit(Inst::ExtractValue { dst: extracted, val: slice_val.clone(), index: 0 });
                    extracted
                }
                Type::Pointer(_) | Type::Array(_, _) => {
                    match slice_val {
                        Value::Reg(r) => r,
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };
            // get the element pointer with the offset
            let elem_ptr = cx.fresh_reg();
            cx.emit(Inst::Comment(format!("simd_store")));
            cx.emit(Inst::Index { dst: elem_ptr, slice: data_ptr, index: offset_val, index_ty: cx.node_types[&args[1].id].clone(), element_ty: ty.clone() });
            // store the SIMD vector to the element pointer
            cx.emit(Inst::Store { ptr: elem_ptr, val: value_val, ty: Type::Simd(Box::new(ty), ConstVal::Lit(size)), align: None });
            Value::Const(Const::Undef) // placeholder since void return
        }
        Intrinsic::SimdConcat => {
            let ty = ta_type(type_args, 0);
            let size = ta_const(type_args, 1);
            let value1_val = lower_expr(cx, &args[0]);
            let value2_val = lower_expr(cx, &args[1]);
            let dst = cx.fresh_reg();
            cx.emit(Inst::Comment(format!("simd_concat({}, {}, ..., ...)", ty, size)));
            cx.emit(Inst::Shuffle {
                dst,
                value_size: match &cx.node_types[&args[0].id] {
                    Type::Simd(_, s) => s.expect_lit(),
                    _ => unreachable!(),
                },
                v0: value1_val,
                v1: value2_val,
                ty,
                size,
                mask: (0..size).collect(),
            });
            Value::Reg(dst)
        }
        Intrinsic::SimdLow
        | Intrinsic::SimdHigh => {
            // simd_low::<T, N>(value) -> simd[T, N]
            let ty = ta_type(type_args, 0);
            let size = ta_const(type_args, 1);
            let value_val = lower_expr(cx, &args[0]);

            let dst = cx.fresh_reg();
            cx.emit(Inst::Comment(format!("{}({}, {}, ...)", intrinsic, ty, size)));
            cx.emit(Inst::Shuffle {
                dst,
                value_size: match &cx.node_types[&args[0].id] {
                    Type::Simd(_, s) => s.expect_lit(),
                    _ => unreachable!(),
                },
                v0: value_val,
                v1: Value::Const(Const::Undef),
                ty,
                size,
                mask: match intrinsic {
                    Intrinsic::SimdLow => (0..size).collect(),
                    Intrinsic::SimdHigh => (size..size*2).collect(),
                    _ => unreachable!(),
                },
            });
            Value::Reg(dst)
        }
    }
}

pub(crate) fn lower_expr<'a>(cx: &mut LowerCtx<'a>, expr: &Expr<'a>) -> Value {
    match &expr.value {
        ExprNode::Bool(b)    => Value::Const(Const::Bool(*b)),
        ExprNode::Int8(n)    => Value::Const(Const::Int8(*n)),
        ExprNode::Int32(n)   => Value::Const(Const::Int32(*n)),
        ExprNode::Int64(n)   => Value::Const(Const::Int64(*n)),
        ExprNode::Uint8(n)   => Value::Const(Const::Uint8(*n)),
        ExprNode::Uint32(n)  => Value::Const(Const::Uint32(*n)),
        ExprNode::Uint64(n)  => Value::Const(Const::Uint64(*n)),
        ExprNode::Float32(n) => Value::Const(Const::Float32(*n)),
        ExprNode::Float64(n) => Value::Const(Const::Float64(*n)),

        // a string literal is a raw `*const u8`: the bare address of a
        // read-only, NUL-terminated global blob. No length is carried; `len()`
        // recovers it with `strlen` (see lower_intrinsic).
        ExprNode::Str(s) => {
            let idx = cx.intern_string(s);
            Value::Const(Const::GlobalStr(idx))
        }

        ExprNode::Var(name) => {
            // locals/params (env) shadow module-level globals. resolution picked
            // the exact binding for this use, so shadowing is already decided.
            if let Some((reg, ty)) = cx.resolved.get(&expr.id).and_then(|b| cx.env.get(b)).cloned() {
                match ty {
                    // fixed arrays are stored in env as the alloca register itself, not a pointer
                    // to one, so loading would yield the array value - we want the pointer
                    Type::Array(_, _) => Value::Reg(reg),
                    // a struct or data-enum aggregate is held by pointer: hand it back.
                    _ if aggregate_struct_name(&ty).is_some() => Value::Reg(reg),
                    _ => {
                        let dst = cx.fresh_reg();
                        cx.emit(Inst::Load { dst, ptr: reg, ty, align: None });
                        Value::Reg(dst)
                    }
                }
            } else if let Some(ty) = cx.globals.get(name).cloned() {
                // reference to a module-level global: its symbol @name is already
                // a pointer to the constant. Materialize that address, then treat
                // it like an env entry - hand back the pointer for aggregates,
                // load the value for scalars.
                let addr = cx.fresh_reg();
                cx.emit(Inst::GlobalPtr { dst: addr, name });
                match ty {
                    Type::Array(_, _) => Value::Reg(addr),
                    _ if aggregate_struct_name(&ty).is_some() => Value::Reg(addr),
                    _ => {
                        let dst = cx.fresh_reg();
                        cx.emit(Inst::Load { dst, ptr: addr, ty, align: None });
                        Value::Reg(dst)
                    }
                }
            } else if matches!(cx.node_types.get(&expr.id), Some(Type::Function { .. })) {
                // a bare top-level function used as a value: its symbol @name is a
                // function pointer. Materialize the address (reusing GlobalPtr).
                let dst = cx.fresh_reg();
                cx.emit(Inst::GlobalPtr { dst, name });
                Value::Reg(dst)
            } else {
                panic!("unknown variable '{name}' in MIL lowering");
            }
        }

        // a unit enum variant used as a value. For a *data* enum it is still an
        // aggregate (alloca + tag store); a field-less enum stays a bare const.
        ExprNode::Path(path) => {
            let c = enum_const(&cx.enums, path)
                .unwrap_or_else(|| panic!("unknown variant '{path}' in MIL lowering"));
            let agg_enum = match cx.node_types.get(&expr.id) {
                Some(Type::Enum { has_payload: true, name: ename, .. }) => Some(*ename),
                _ => None,
            };
            match agg_enum {
                Some(ename) => construct_data_variant(cx, ename, path, c, &[]),
                None => Value::Const(c),
            }
        }

        ExprNode::Struct { name, fields, .. } => {
            // a struct-style data-enum constructor `Msg::Cc { id, val }` reuses the
            // struct-literal syntax but builds the aggregate in place. Detected by
            // the node's inferred aggregate-enum type; construction requires
            // declaration order (typecheck enforced), so the field exprs are already
            // in payload-struct order and pass positionally to the shared builder.
            if let Some(Type::Enum { has_payload: true, name: ename, .. }) =
                cx.node_types.get(&expr.id).cloned().as_ref()
            {
                let ename = *ename;
                let tag = enum_const(&cx.enums, name).expect("variant const validated in typecheck");
                let args: Vec<Expr<'a>> = fields.iter().map(|(_, e)| e.clone()).collect();
                return construct_data_variant(cx, ename, name, tag, &args);
            }

            // an ordinary struct literal: resolution left it a single segment.
            let name = name.as_single().expect("struct literal name validated in typecheck");

            // reuse a hoisted entry-block slot if the caller provided one;
            // otherwise this literal owns a fresh slot. take() so nested field
            // literals don't inherit the target.
            let dst = match cx.store_target.take() {
                Some(slot) => slot,
                None => {
                    let dst = cx.fresh_reg();
                    cx.emit(Inst::AllocaStruct { dst, name, align: None });
                    dst
                }
            };

            for (i, (_field_name, field_expr)) in fields.iter().enumerate() {
                let field_ty = cx.structs[name][i].1.clone();
                let field_val = lower_expr(cx, field_expr);
                let field_ptr = cx.fresh_reg();

                cx.emit(Inst::FieldPtr {
                    dst: field_ptr,
                    struct_name: name,
                    base: dst,
                    field_index: i,
                });
                match field_ty {
                    // a nested struct field is inlined storage, so copy the
                    // source struct's contents into it rather than storing a
                    // pointer (field_val is the source struct's address)
                    Type::Struct { name: inner, .. } => {
                        let src = match field_val {
                            Value::Reg(r) => r,
                            _ => unreachable!(),
                        };
                        copy_struct(cx, inner, src, field_ptr);
                    }
                    // an array field is likewise inlined aggregate storage:
                    // field_val is the source array's address, so copy the whole
                    // aggregate in (load+store) rather than storing the pointer as
                    // if it were the array value.
                    Type::Array(..) => {
                        let src = match field_val {
                            Value::Reg(r) => r,
                            _ => unreachable!(),
                        };
                        let loaded = cx.fresh_reg();
                        cx.emit(Inst::Load { dst: loaded, ptr: src, ty: field_ty.clone(), align: None });
                        cx.emit(Inst::Store { ptr: field_ptr, val: Value::Reg(loaded), ty: field_ty, align: None });
                    }
                    _ => {
                        cx.emit(Inst::Store {
                            ptr: field_ptr,
                            val: field_val,
                            ty: field_ty,
                            align: None,
                        });
                    }
                }
            }
            Value::Reg(dst)
        }
        ExprNode::Access { base, field } => {
            let base_val = lower_expr(cx, base);
            let base_ty = cx.node_types[&base.id].clone();
            // matches the typechecker's one-level auto-deref for `ptr.field`
            let struct_name = match base_ty {
                Type::Struct { name, .. } => name,
                Type::Pointer(inner) => match *inner {
                    Type::Struct { name, .. } => name,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let field_index = cx.structs[&struct_name]
                .iter()
                .position(|(fname, _)| fname == field)
                .unwrap();
            let field_ty = cx.structs[&struct_name][field_index].1.clone();

            let field_ptr = cx.fresh_reg();
            cx.emit(Inst::FieldPtr {
                dst: field_ptr,
                struct_name: &struct_name,
                base: match base_val {
                    Value::Reg(r) => r,
                    _ => unreachable!(),
                },
                field_index,
            });
            match field_ty {
                // a struct- or array-typed field is inlined aggregate storage: its
                // value is its address (indexing/copying use the pointer), so hand
                // back the field pointer instead of loading it
                Type::Struct { .. } | Type::Array(..) => Value::Reg(field_ptr),
                _ => {
                    let dst = cx.fresh_reg();
                    cx.emit(Inst::Load { dst, ptr: field_ptr, ty: field_ty, align: None });
                    Value::Reg(dst)
                }
            }
        }

        // use fat pointer struct for slices
        ExprNode::Slice(elements) if elements.len() == 0 => {
            todo!()
        },
        ExprNode::Slice(elements) => {
            let (ty, is_fixed) = match cx.node_types[&expr.id].clone() {
                Type::Slice(inner) => (*inner, false),
                Type::Array(inner, _) => (*inner, true),
                _ => unreachable!(),
            };

            // %arr_reg = alloca [N x T]
            // a fixed array bound to a local may reuse a hoisted entry-block slot
            // (see `collect_locals`); the fat-pointer case always allocs backing
            // storage fresh and never carries a store_target.
            let arr_reg = match (is_fixed, cx.store_target.take()) {
                (true, Some(slot)) => slot,
                _ => {
                    let arr_reg = cx.fresh_reg();
                    cx.emit(Inst::AllocaArray {
                        dst: arr_reg,
                        ty: ty.clone(),
                        length: elements.len(),
                    });
                    arr_reg
                }
            };

            // %ptr = gep
            for (index, element) in elements.iter().enumerate() {
                let ptr = cx.fresh_reg();
                cx.emit(Inst::IndexArray {
                    dst: ptr,
                    ty: ty.clone(),
                    length: elements.len(),
                    array: arr_reg,
                    index,
                });

                let elem_val = lower_expr(cx, element);
                cx.emit(Inst::Store { ptr, val: elem_val, ty: ty.clone(), align: None });
            }

            if is_fixed {
                // the alloca register is the value
                Value::Reg(arr_reg)
            } else {
                // construct fat pointer struct { ptr, len }
                // %fat_ptr0 = insertvalue { ptr, i32 } undef, ptr %arr_reg, 0
                // %fat_ptr1 = insertvalue { ptr, i32 } %fat_ptr0, i32 N, 1
                let fat_ptr0 = cx.fresh_reg();
                cx.emit(Inst::InsertValue {
                    dst: fat_ptr0,
                    elem: Value::Const(Const::Undef),
                    ty: Type::Pointer(Box::new(ty.clone())),
                    val: Value::Reg(arr_reg),
                    index: 0,
                });
                let fat_ptr1 = cx.fresh_reg();
                cx.emit(Inst::InsertValue {
                    dst: fat_ptr1,
                    elem: Value::Reg(fat_ptr0),
                    ty: Type::Int32,
                    val: Value::Const(Const::Int32(elements.len() as i32)),
                    index: 1,
                });

                Value::Reg(fat_ptr1)
            }
        }

        // *ptr (where ptr is *T or T[]) = load from ptr, so dst = load ptr
        ExprNode::Unary { op: UnaryOp::Deref, operand } => {
            let ptr_val = lower_expr(cx, operand);
            let ptr_reg = match ptr_val {
                Value::Reg(r) => r,
                _ => unreachable!(),
            };
            let ty = match cx.node_types[&operand.id].clone() {
                Type::Pointer(inner) | Type::Slice(inner) => *inner,
                _ => unreachable!(),
            };
            // dereferencing to a struct keeps it in memory
            // the pointer already points at the struct's storage, so it is the
            // struct value
            if matches!(ty, Type::Struct { .. }) {
                return Value::Reg(ptr_reg);
            }
            let dst = cx.fresh_reg();
            cx.emit(Inst::Load { dst, ptr: ptr_reg, ty, align: None });
            Value::Reg(dst)
        }

        ExprNode::Unary { op: UnaryOp::AddrOf, operand } => {
            let ptr = lower_lvalue(cx, operand); // get the address of the operand
            Value::Reg(ptr)
        }

        ExprNode::Unary { op, operand } => {
            let val = lower_expr(cx, operand);
            let ty = cx.node_types[&operand.id].clone();
            let dst = cx.fresh_reg();
            cx.emit(Inst::Unary { dst, op: *op, val, ty });
            Value::Reg(dst)
        }

        // short-circuiting && and ||
        ExprNode::Binary { op: op @ (BinaryOp::And | BinaryOp::Or), left, right } => {
            let lhs = lower_expr(cx, left);
            // branch needs a Register, but lhs might be a bare Const(Bool) (e.g. `true && foo()`)
            let lhs_reg = match lhs {
                Value::Reg(r) => r,
                v => {
                    let r = cx.fresh_reg();
                    cx.emit(Inst::Alloca { dst: r, ty: Type::Bool, align: None });
                    cx.emit(Inst::Store { ptr: r, val: v, ty: Type::Bool, align: None });
                    let s = cx.fresh_reg();
                    cx.emit(Inst::Load { dst: s, ptr: r, ty: Type::Bool, align: None });
                    s
                }
            };

            let result_ptr = cx.fresh_reg();
            cx.emit(Inst::Alloca { dst: result_ptr, ty: Type::Bool, align: None });

            let rhs_block           = cx.fresh_block();
            let short_circuit_block = cx.fresh_block();
            let merge_block         = cx.fresh_block();

            // && : lhs false -> skip rhs, result = false
            // || : lhs true  -> skip rhs, result = true
            cx.terminate(match op {
                BinaryOp::And => Terminator::Branch { cond: lhs_reg, then_block: rhs_block, else_block: short_circuit_block },
                BinaryOp::Or  => Terminator::Branch { cond: lhs_reg, then_block: short_circuit_block, else_block: rhs_block },
                _ => unreachable!(),
            });

            cx.current_block = short_circuit_block;
            cx.emit(Inst::Comment(format!("{} short-circuit", op)));
            cx.emit(Inst::Store {
                ptr: result_ptr,
                val: Value::Const(Const::Bool(matches!(op, BinaryOp::Or))),
                ty: Type::Bool, align: None,
            });
            cx.terminate(Terminator::Jump(merge_block));

            cx.current_block = rhs_block;
            let rhs = lower_expr(cx, right);
            cx.emit(Inst::Store { ptr: result_ptr, val: rhs, ty: Type::Bool, align: None });
            cx.terminate(Terminator::Jump(merge_block));

            cx.current_block = merge_block;
            let dst = cx.fresh_reg();
            cx.emit(Inst::Load { dst, ptr: result_ptr, ty: Type::Bool, align: None });
            Value::Reg(dst)
        }

        ExprNode::Binary { op, left, right } => {
            // And and Or doesn't reach here but we keep it just in case
            // (for non short-circuiting versions)
            let lhs = lower_expr(cx, left);
            let rhs = lower_expr(cx, right);
            let ty = cx.node_types[&left.id].clone();
            let dst = cx.fresh_reg();
            cx.emit(Inst::Binary { dst, op: *op, lhs, rhs, ty });
            Value::Reg(dst)
        }

        ExprNode::Call { func, type_args, args }
            if matches!(&func.value, ExprNode::Var(name)
                if Intrinsic::lookup(name).is_some()) => {
            let ExprNode::Var(name) = &func.value else { unreachable!() };
            let intrinsic = Intrinsic::lookup(name).unwrap();
            lower_intrinsic(cx, intrinsic, type_args, args)
        }

        // a receiver method call `recv.method(args)`, resolved by typecheck into
        // `method_calls`. Lower to a direct call to the desugared function, passing
        // the adjusted receiver (address-of, or as-is) as the leading `self` arg.
        ExprNode::Call { func, args, .. } if cx.method_calls.contains_key(&expr.id) => {
            let mc = cx.method_calls[&expr.id].clone();
            let base = match &func.value {
                ExprNode::Access { base, .. } => base,
                _ => unreachable!("method call callee is always a field access"),
            };
            cx.emit(Inst::Comment(format!("method call {}(...)", mc.target)));

            let recv_val = match mc.adjust {
                RecvAdjust::AddrOf => Value::Reg(lower_lvalue(cx, base)),
                RecvAdjust::AsIs => lower_expr(cx, base),
            };
            let mut lowered_args: Vec<(Value, Type<'a>)> = Vec::with_capacity(args.len() + 1);
            lowered_args.push((recv_val, mc.param_tys[0].clone()));
            for (i, arg) in args.iter().enumerate() {
                let arg_ty = cx.node_types[&arg.id].clone();
                let val = lower_expr(cx, arg);
                let param_ty = mc.param_tys[i + 1].clone();
                lowered_args.push((coerce(cx, val, &arg_ty, &param_ty), param_ty));
            }

            let callee = Callee::Direct(mc.target);
            let return_type = mc.return_type.clone();
            if let Some(sname) = aggregate_struct_name(&return_type) {
                let slot = cx.fresh_reg();
                cx.emit(Inst::AllocaStruct { dst: slot, name: sname, align: None });
                cx.emit(Inst::Call {
                    dst: None, callee, args: lowered_args,
                    return_type: Type::Void, sret: Some((slot, sname)),
                });
                Value::Reg(slot)
            } else if return_type == Type::Void {
                cx.emit(Inst::Call { dst: None, callee, args: lowered_args, return_type, sret: None });
                Value::Const(Const::Bool(false)) // placeholder
            } else {
                let dst = cx.fresh_reg();
                cx.emit(Inst::Call { dst: Some(dst), callee, args: lowered_args, return_type, sret: None });
                Value::Reg(dst)
            }
        }

        ExprNode::Call { func, args, .. } => {
            // a data-enum constructor `Msg::Note(a, b)` is not a real call: build
            // the aggregate in place. (A field-less enum "call" is impossible -
            // typecheck yields a scalar-typed variant, never `has_payload: true`.)
            if let ExprNode::Path(path) = &func.value {
                if let Some(c) = enum_const(&cx.enums, path) {
                    if let Some(Type::Enum { has_payload: true, name: ename, .. }) =
                        cx.node_types.get(&expr.id).cloned().as_ref()
                    {
                        let ename = *ename;
                        return construct_data_variant(cx, ename, path, c, args);
                    }
                }
            }
            cx.emit(Inst::Comment(format!("call {}(...)", func.value)));
            // a bare name that isn't a local/param/global is a top-level function
            // -> direct call. Anything else (a local holding a fn pointer, a struct
            // field, etc.) is lowered to a `ptr` value and called indirectly.
            let callee = match &func.value {
                ExprNode::Var(name)
                    if !cx.resolved.contains_key(&func.id) && !cx.globals.contains_key(name) =>
                    Callee::Direct(*name),
                _ => Callee::Indirect(lower_expr(cx, func)),
            };
            // grab the callee's parameter types so we can apply array->slice coercion
            let param_tys: Vec<Type<'a>> = match cx.node_types.get(&func.id).cloned() {
                Some(Type::Function { params, .. }) => params,
                _ => vec![],
            };
            let lowered_args = args.iter().enumerate().map(|(i, arg)| {
                let arg_ty = cx.node_types[&arg.id].clone();
                let val = lower_expr(cx, arg);
                let param_ty = param_tys.get(i).cloned()
                    .unwrap_or_else(|| arg_ty.clone());
                let coerced_val = coerce(cx, val, &arg_ty, &param_ty);
                (coerced_val, param_ty)
            }).collect();

            let return_type = cx.node_types[&expr.id].clone();
            let struct_ret = aggregate_struct_name(&return_type);
            if let Some(sname) = struct_ret {
                // if sret, allocate the result slot here and hand the callee a
                // pointer to it. The call returns void & the slot is the value
                let slot = cx.fresh_reg();
                cx.emit(Inst::AllocaStruct { dst: slot, name: sname, align: None });
                cx.emit(Inst::Call {
                    dst: None,
                    callee,
                    args: lowered_args,
                    return_type: Type::Void,
                    sret: Some((slot, sname)),
                });
                Value::Reg(slot)
            } else if return_type == Type::Void {
                cx.emit(Inst::Call { dst: None, callee, args: lowered_args, return_type, sret: None });
                Value::Const(Const::Bool(false)) // placeholder
            } else {
                let dst = cx.fresh_reg();
                cx.emit(Inst::Call { dst: Some(dst), callee, args: lowered_args, return_type, sret: None });
                Value::Reg(dst)
            }
        }

        ExprNode::Index { slice, index } => {
            let slice_val = lower_expr(cx, slice);
            let index_val = lower_expr(cx, index);
            let element_ty = match &cx.node_types[&slice.id] {
                Type::Slice(inner) => *inner.clone(),
                Type::Pointer(inner) => *inner.clone(),
                Type::Array(inner, _) => *inner.clone(),
                _ => unreachable!(),
            };

            let data_ptr = match cx.node_types[&slice.id] {
                Type::Slice(_) => {
                    let extracted = cx.fresh_reg();
                    cx.emit(Inst::ExtractValue { dst: extracted, val: slice_val.clone(), index: 0 });
                    extracted
                }
                Type::Pointer(_) | Type::Array(_, _) => {
                    // for fixed arrays the Var lowering already gave us the alloca pointer
                    match slice_val {
                        Value::Reg(r) => r,
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };
            let elem_ptr = cx.fresh_reg();
            cx.emit(Inst::Index { dst: elem_ptr, slice: data_ptr, index: index_val, index_ty: cx.node_types[&index.id].clone(), element_ty: element_ty.clone() });

            let dst = cx.fresh_reg();
            cx.emit(Inst::Load { dst, ptr: elem_ptr, ty: element_ty, align: None });

            Value::Reg(dst)
        }
    }
}

pub(crate) fn lower_lvalue<'a>(cx: &mut LowerCtx<'a>, expr: &Expr<'a>) -> Register {
    match &expr.value {
        ExprNode::Var(_) => cx.env[&cx.resolved[&expr.id]].0,

        ExprNode::Access { base, field } => {
            let base_val = lower_expr(cx, base);
            let base_ty = cx.node_types[&base.id].clone();
            // matches the typechecker's one-level auto-deref for `ptr.field`
            let struct_name = match base_ty {
                Type::Struct { name, .. } => name,
                Type::Pointer(inner) => match *inner {
                    Type::Struct { name, .. } => name,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let field_index = cx.structs[&struct_name]
                .iter()
                .position(|(fname, _)| fname == field)
                .unwrap();

            let field_ptr = cx.fresh_reg();
            cx.emit(Inst::FieldPtr {
                dst: field_ptr,
                struct_name: &struct_name,
                base: match base_val {
                    Value::Reg(r) => r,
                    _ => unreachable!(),
                },
                field_index,
            });
            field_ptr
        }

        // evaluate ptr as a value, register is the address
        ExprNode::Unary { op: UnaryOp::Deref, operand } => {
            match lower_expr(cx, operand) {
                Value::Reg(r) => r,
                _ => unreachable!(),
            }
        }

        ExprNode::Index { slice, index } => {
            let slice_val = lower_expr(cx, slice);
            let index_val = lower_expr(cx, index);
            let element_ty = match cx.node_types[&slice.id].clone() {
                Type::Slice(inner) => *inner,
                Type::Pointer(inner) => *inner,
                Type::Array(inner, _) => *inner,
                _ => unreachable!(),
            };

            // same extraction as rvalue
            let data_ptr = match cx.node_types[&slice.id] {
                Type::Slice(_) => {
                    let extracted = cx.fresh_reg();
                    cx.emit(Inst::ExtractValue { dst: extracted, val: slice_val.clone(), index: 0 });
                    extracted
                }
                Type::Pointer(_) | Type::Array(_, _) => {
                    match slice_val {
                        Value::Reg(r) => r,
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };
            // return the address of the element instead of loading
            let dst = cx.fresh_reg();
            cx.emit(Inst::Index { dst, slice: data_ptr, index: index_val, index_ty: cx.node_types[&index.id].clone(), element_ty });
            dst
        }

        e => panic!("invalid lvalue: {}", e),
    }
}

// Field-by-field copy from one struct pointer to another. Both `src` and `dst`
// must be pointers to a struct of the given name. Recurses on nested struct
// fields so the whole tree is deep-copied (value semantics).
// TODO: switch to an `llvm.memcpy` intrinsic for large structs instead of
// emitting a load/store per scalar field.
pub(crate) fn copy_struct<'a>(cx: &mut LowerCtx<'a>, struct_name: &'a str, src: Register, dst: Register) {
    let fields = cx.structs[struct_name].clone();
    for (i, (_fname, fty)) in fields.iter().enumerate() {
        let src_field = cx.fresh_reg();
        let dst_field = cx.fresh_reg();
        cx.emit(Inst::FieldPtr { dst: src_field, struct_name, base: src, field_index: i });
        cx.emit(Inst::FieldPtr { dst: dst_field, struct_name, base: dst, field_index: i });
        match fty {
            // for nested struct, both field pointers point at inlined sub-struct
            // storage, so recurse to deep-copy it
            Type::Struct { name: inner_name, .. } => {
                copy_struct(cx, inner_name, src_field, dst_field);
            }
            // everything else (scalars, arrays, slices, simd) is a single
            // value/aggregate that an LLVM load/store copies directly
            _ => {
                let loaded = cx.fresh_reg();
                cx.emit(Inst::Load { dst: loaded, ptr: src_field, ty: fty.clone(), align: None });
                cx.emit(Inst::Store { ptr: dst_field, val: Value::Reg(loaded), ty: fty.clone(), align: None });
            }
        }
    }
}

/// Construct a data-enum aggregate in place: alloca `%Enum` (or reuse a hoisted
/// slot), store the variant's discriminant into the tag (field 0), then store
/// each payload argument into the variant's payload struct at field 1. `tag` is
/// the variant's discriminant const; `variant_path` is the `Enum::Variant` path.
/// Returns a pointer to the aggregate. Mirrors `ExprNode::Struct` lowering, so a
/// unit variant (`args` empty) is just alloca + tag store.
fn construct_data_variant<'a>(
    cx: &mut LowerCtx<'a>,
    enum_name: &'a str,
    variant_path: &Path<'a>,
    tag: Const,
    args: &[Expr<'a>],
) -> Value {
    let base = match cx.store_target.take() {
        Some(slot) => slot,
        None => {
            let dst = cx.fresh_reg();
            cx.emit(Inst::AllocaStruct { dst, name: enum_name, align: None });
            dst
        }
    };
    // tag -> field 0
    let tag_ptr = cx.fresh_reg();
    cx.emit(Inst::FieldPtr { dst: tag_ptr, struct_name: enum_name, base, field_index: 0 });
    let repr = cx.enums[enum_name].repr.clone();
    cx.emit(Inst::Store { ptr: tag_ptr, val: Value::Const(tag), ty: repr, align: None });

    if !args.is_empty() {
        let variant = variant_path.as_variant().unwrap().1;
        let pstruct = crate::typecheck::enum_payload_struct_name(enum_name, variant);
        let payload_base = cx.fresh_reg();
        cx.emit(Inst::FieldPtr { dst: payload_base, struct_name: enum_name, base, field_index: 1 });
        for (i, arg) in args.iter().enumerate() {
            let field_ty = cx.structs[pstruct][i].1.clone();
            let arg_ty = cx.node_types[&arg.id].clone();
            let field_val = lower_expr(cx, arg);
            let field_ptr = cx.fresh_reg();
            cx.emit(Inst::FieldPtr { dst: field_ptr, struct_name: pstruct, base: payload_base, field_index: i });
            match aggregate_struct_name(&field_ty) {
                // a struct- or data-enum-typed payload field is inlined storage:
                // deep-copy the source aggregate into it (field_val is its address).
                Some(inner) => {
                    let src = match field_val { Value::Reg(r) => r, _ => unreachable!() };
                    copy_struct(cx, inner, src, field_ptr);
                }
                None => match field_ty {
                    // an array field is inlined aggregate storage: copy it whole.
                    Type::Array(..) => {
                        let src = match field_val { Value::Reg(r) => r, _ => unreachable!() };
                        let loaded = cx.fresh_reg();
                        cx.emit(Inst::Load { dst: loaded, ptr: src, ty: field_ty.clone(), align: None });
                        cx.emit(Inst::Store { ptr: field_ptr, val: Value::Reg(loaded), ty: field_ty, align: None });
                    }
                    _ => {
                        let cv = coerce(cx, field_val, &arg_ty, &field_ty);
                        cx.emit(Inst::Store { ptr: field_ptr, val: cv, ty: field_ty, align: None });
                    }
                },
            }
        }
    }
    Value::Reg(base)
}
