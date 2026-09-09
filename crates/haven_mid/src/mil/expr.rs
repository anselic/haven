use haven_common::ast::*;
use crate::intrinsics::Intrinsic;
use crate::typecheck::RecvAdjust;
use super::ir::*;
use haven_common::defs::DefId;
use super::ctx::{LowerCtx, coerce, aggregate_def, is_aggregate_ty, enum_const, lit_const, ta_type, ta_const};

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
            let enums = cx.enums.clone();
            let unwrap_enum = move |t: Type<'a>| match t.def().and_then(|d| enums.get(&d)) {
                Some(e) => e.repr.clone(),
                None => t,
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
        Intrinsic::PtrWrite => {
            // ptr_write::<T>(dst, value): the store half of `*dst = value`,
            // without the destroy-the-old-value half. Identical to how `Assign`
            // lowers, except the destination is a pointer *value* rather than a
            // place, so it is lowered rather than addressed.
            let ty = ta_type(type_args, 0);
            let dst = lower_expr(cx, &args[0]);
            let val = lower_expr(cx, &args[1]);
            let ptr = as_register(cx, dst, &Type::Pointer(Box::new(ty.clone())));
            // an aggregate is held by pointer on both sides, so this is the same
            // copy `Assign` does; a plain `Store` would write the source pointer
            // into the first field. An array element type is an aggregate here
            // too - `Vec<[Res; 2]>` reaches this with `T = [Res; 2]`.
            if is_aggregate_ty(&ty, &cx.enums) {
                let Value::Reg(src) = val else {
                    unreachable!("an aggregate value is always a pointer register")
                };
                copy_aggregate(cx, &ty, src, ptr);
                return Value::Const(Const::Undef);
            }
            let value_ty = cx.node_types[&args[1].id].clone();
            let val = coerce(cx, val, &value_ty, &ty);
            cx.emit(Inst::Store { ptr, val, ty, align: None });
            Value::Const(Const::Undef)
        }
        Intrinsic::DropInPlace => {
            // reaching lowering means the ownership pass did *not* rewrite this
            // into a destructor loop, i.e. `T` owns nothing (or the program has
            // no `Delete` impl at all), so there is nothing to destroy. The
            // arguments are still lowered: they are ordinary expressions and may
            // have side effects.
            lower_expr(cx, &args[0]);
            lower_expr(cx, &args[1]);
            Value::Const(Const::Undef)
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

            cx.emit(Inst::Comment("simd_load".to_string()));
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
            cx.emit(Inst::Comment("simd_store".to_string()));
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
        ExprNode::Int16(n)   => Value::Const(Const::Int16(*n)),
        ExprNode::Int32(n)   => Value::Const(Const::Int32(*n)),
        ExprNode::Int64(n)   => Value::Const(Const::Int64(*n)),
        ExprNode::Uint8(n)   => Value::Const(Const::Uint8(*n)),
        ExprNode::Uint16(n)  => Value::Const(Const::Uint16(*n)),
        ExprNode::Uint32(n)  => Value::Const(Const::Uint32(*n)),
        ExprNode::Uint64(n)  => Value::Const(Const::Uint64(*n)),
        ExprNode::Float32(n) => Value::Const(Const::Float32(*n)),
        ExprNode::Float64(n) => Value::Const(Const::Float64(*n)),

        // a literal written without a width suffix carries no type in the node -
        // the checker put the one it was given in `node_types`.
        ExprNode::IntLit(_) | ExprNode::FloatLit(_) =>
            Value::Const(lit_const(&cx.node_types[&expr.id], &expr.value)),

        // a string literal is a raw `*const u8`: the bare address of a
        // read-only, NUL-terminated global blob. No length is carried; `len()`
        // recovers it with `strlen` (see lower_intrinsic).
        ExprNode::Str(s) => {
            let idx = cx.intern_string(s);
            Value::Const(Const::GlobalStr(idx))
        }

        // a generic fn reference `foo::<T>` is rewritten to a bare `Var` of the
        // mangled instance by monomorphization, so none survives to MIL.
        ExprNode::FnRef { .. } => unreachable!("FnRef eliminated by monomorphization"),

        ExprNode::Var(name) => {
            // locals/params (env) shadow module-level globals. resolution picked
            // the exact binding for this use, so shadowing is already decided.
            if let Some((reg, ty)) = cx.resolved.get(&expr.id).and_then(|b| cx.env.get(b)).cloned() {
                match ty {
                    // an aggregate - struct, data enum or fixed array - is held by
                    // pointer, and that pointer *is* the value. Loading would
                    // yield the record itself where a handle is expected.
                    _ if is_aggregate_ty(&ty, &cx.enums) => Value::Reg(reg),
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
                    _ if is_aggregate_ty(&ty, &cx.enums) => Value::Reg(addr),
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
            let agg_enum = cx.node_types.get(&expr.id).cloned()
                .and_then(|t| aggregate_def(&t, &cx.enums));
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
            let node_ty = cx.node_types.get(&expr.id).cloned();
            if let Some(ename) = node_ty.as_ref()
                .filter(|t| t.def().is_some_and(|d| cx.enums.contains_key(&d)))
                .and_then(|t| aggregate_def(t, &cx.enums))
            {
                let tag = enum_const(&cx.enums, name).expect("variant const validated in typecheck");
                let args: Vec<Expr<'a>> = fields.iter().map(|(_, e)| e.clone()).collect();
                return construct_data_variant(cx, ename, name, tag, &args);
            }

            // an ordinary struct literal: resolution recorded which struct.
            let name = name.def;

            // reuse a hoisted entry-block slot if the caller provided one;
            // otherwise this literal owns a fresh slot. take() so nested field
            // literals don't inherit the target.
            let dst = match cx.store_target.take() {
                Some(slot) => slot,
                None => {
                    let dst = cx.fresh_reg();
                    cx.emit(Inst::AllocaStruct { dst, def: name, align: None });
                    dst
                }
            };

            for (i, (_field_name, field_expr)) in fields.iter().enumerate() {
                let field_ty = cx.types[&name].fields[i].1.clone();
                let field_val = lower_expr(cx, field_expr);
                let field_ptr = cx.fresh_reg();

                cx.emit(Inst::FieldPtr {
                    dst: field_ptr,
                    struct_def: name,
                    base: dst,
                    field_index: i,
                });
                match aggregate_def(&field_ty, &cx.enums) {
                    // a nested aggregate field is inlined storage, so copy the
                    // source's contents into it rather than storing a pointer
                    // (field_val is the source aggregate's address)
                    Some(inner) => {
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
                    None if matches!(field_ty, Type::Array(..)) => {
                        let src = match field_val {
                            Value::Reg(r) => r,
                            _ => unreachable!(),
                        };
                        copy_aggregate(cx, &field_ty, src, field_ptr);
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
            let struct_name = match &base_ty {
                Type::Pointer(inner) => inner.def(),
                t => t.def(),
            }.expect("field access on a non-aggregate rejected in typecheck");
            let field_index = cx.types[&struct_name].fields
                .iter()
                .position(|(fname, _)| fname == field)
                .unwrap();
            let field_ty = cx.types[&struct_name].fields[field_index].1.clone();

            let field_ptr = cx.fresh_reg();
            cx.emit(Inst::FieldPtr {
                dst: field_ptr,
                struct_def: struct_name,
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
                _ if is_aggregate_ty(&field_ty, &cx.enums) => Value::Reg(field_ptr),
                _ => {
                    let dst = cx.fresh_reg();
                    cx.emit(Inst::Load { dst, ptr: field_ptr, ty: field_ty, align: None });
                    Value::Reg(dst)
                }
            }
        }

        // use fat pointer struct for slices
        // `[value; N]`: one alloca, the element evaluated *once*, then copied
        // into each of the N slots. Evaluating once is the contract - `[f(); 8]`
        // calls `f` a single time - and it is also what keeps a large repeat from
        // re-running arbitrary work N times.
        //
        // Structurally this is the `Slice` arm with the element list replaced by
        // a count, so it takes the same `store_target` (letting a fixed array
        // bound to a local write straight into its hoisted entry-block slot) and
        // the same aggregate-vs-scalar store decision.
        ExprNode::Repeat { value, count } => {
            let n = count.expect_lit();
            let (ty, is_fixed) = match cx.node_types[&expr.id].clone() {
                Type::Slice(inner) => (*inner, false),
                Type::Array(inner, _) => (*inner, true),
                _ => unreachable!(),
            };

            // taken before the element is lowered, so the element's own lowering
            // cannot claim the slot meant for this array.
            let arr_reg = match (is_fixed, cx.store_target.take()) {
                (true, Some(slot)) => slot,
                _ => {
                    let arr_reg = cx.fresh_reg();
                    cx.emit(Inst::AllocaArray { dst: arr_reg, ty: ty.clone(), length: n });
                    arr_reg
                }
            };

            let elem_val = lower_expr(cx, value);
            let aggregate = is_aggregate_ty(&ty, &cx.enums);
            for index in 0..n {
                let ptr = cx.fresh_reg();
                cx.emit(Inst::IndexArray {
                    dst: ptr, ty: ty.clone(), length: n, array: arr_reg, index,
                });
                if aggregate {
                    let Value::Reg(src) = elem_val else {
                        unreachable!("an aggregate value is always a pointer register")
                    };
                    // every slot gets its own copy of the one evaluated element.
                    copy_aggregate(cx, &ty, src, ptr);
                } else {
                    cx.emit(Inst::Store {
                        ptr, val: elem_val.clone(), ty: ty.clone(), align: None });
                }
            }

            if is_fixed {
                Value::Reg(arr_reg)
            } else {
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
                    val: Value::Const(Const::Int32(n as i32)),
                    index: 1,
                });
                Value::Reg(fat_ptr1)
            }
        }

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
                // an aggregate element lands in inline storage: the slot *is* the
                // element, so copy the produced aggregate into it rather than
                // storing its pointer as one machine word (the old boxed
                // `[N x ptr]` layout). A nested array is an aggregate too, which
                // is why this asks about the type and not about a definition.
                // Scalars store directly.
                if is_aggregate_ty(&ty, &cx.enums) {
                    let Value::Reg(src) = elem_val else {
                        unreachable!("an aggregate value is always a pointer register")
                    };
                    copy_aggregate(cx, &ty, src, ptr);
                } else {
                    cx.emit(Inst::Store { ptr, val: elem_val, ty: ty.clone(), align: None });
                }
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
            // dereferencing to an aggregate keeps it in memory: the pointer
            // already points at its storage, so it *is* the value. This used to
            // ask `aggregate_def`, which does not speak for `[T; N]`, so a
            // `*p` on a `*[T; N]` tried to load the whole array as a scalar.
            if is_aggregate_ty(&ty, &cx.enums) {
                return Value::Reg(ptr_reg);
            }
            let dst = cx.fresh_reg();
            cx.emit(Inst::Load { dst, ptr: ptr_reg, ty, align: None });
            Value::Reg(dst)
        }

        ExprNode::Unary { op: UnaryOp::AddrOf, operand } => {
            // `&place` addresses the place directly - a local or global var, a
            // field, an index, a deref, everything `lower_lvalue` handles (a
            // global yields its real address, not a copy). `&<temporary>` has no
            // such storage, so its value is spilled into a fresh slot addressed in
            // its place.
            let ptr = if matches!(&operand.value,
                ExprNode::Var(_)
                | ExprNode::Access { .. }
                | ExprNode::Index { .. }
                | ExprNode::Unary { op: UnaryOp::Deref, .. })
            {
                lower_lvalue(cx, operand)
            } else {
                spill_temporary(cx, operand)
            };
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

            let result_ptr = cx.fresh_reg();
            cx.emit(Inst::Alloca { dst: result_ptr, ty: Type::Bool, align: None });

            let rhs_block           = cx.fresh_block();
            let short_circuit_block = cx.fresh_block();
            let merge_block         = cx.fresh_block();

            // && : lhs false -> skip rhs, result = false
            // || : lhs true  -> skip rhs, result = true
            cx.terminate(match op {
                BinaryOp::And => Terminator::Branch { cond: lhs, then_block: rhs_block, else_block: short_circuit_block },
                BinaryOp::Or  => Terminator::Branch { cond: lhs, then_block: short_circuit_block, else_block: rhs_block },
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
                RecvAdjust::AddrOf => Value::Reg(lower_receiver(cx, base)),
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

            // a diverging receiver or argument already terminated the block.
            if cx.is_terminated() {
                return Value::Const(Const::Undef);
            }

            let callee = Callee::Direct(mc.target);
            let return_type = mc.return_type.clone();
            // a `!`-returning method never comes back - void call + unreachable.
            if return_type == Type::Never {
                cx.emit(Inst::Call { dst: None, callee, args: lowered_args, return_type: Type::Void, sret: None });
                cx.terminate(Terminator::Unreachable);
                return Value::Const(Const::Undef);
            }
            if is_aggregate_ty(&return_type, &cx.enums) {
                let slot = alloca_aggregate(cx, &return_type);
                cx.emit(Inst::Call {
                    dst: None, callee, args: lowered_args,
                    return_type: Type::Void, sret: Some((slot, return_type.clone())),
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
            if let ExprNode::Path(path) = &func.value
                && let Some(c) = enum_const(&cx.enums, path) {
                    let agg = cx.node_types.get(&expr.id).cloned()
                        .and_then(|t| aggregate_def(&t, &cx.enums));
                    if let Some(ename) = agg {
                        return construct_data_variant(cx, ename, path, c, args);
                    }
                }
            cx.emit(Inst::Comment(format!("call {}(...)", func.value)));
            // a bare name that isn't a local/param/global is a top-level function
            // -> direct call. Anything else (a local holding a fn pointer, a struct
            // field, etc.) is lowered to a `ptr` value and called indirectly.
            let callee = match &func.value {
                ExprNode::Var(name)
                    if !cx.resolved.contains_key(&func.id) && !cx.globals.contains_key(name) =>
                    Callee::Direct(name),
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

            // a diverging argument (`f(abort())`) already terminated the block;
            // the call itself is unreachable, so drop it.
            if cx.is_terminated() {
                return Value::Const(Const::Undef);
            }

            let return_type = cx.node_types[&expr.id].clone();
            // a call to a `!`-returning proc never comes back: emit it as a void
            // call and mark the block unreachable, exactly like `abort`.
            if return_type == Type::Never {
                cx.emit(Inst::Call { dst: None, callee, args: lowered_args, return_type: Type::Void, sret: None });
                cx.terminate(Terminator::Unreachable);
                return Value::Const(Const::Undef);
            }
            if is_aggregate_ty(&return_type, &cx.enums) {
                // if sret, allocate the result slot here and hand the callee a
                // pointer to it. The call returns void & the slot is the value
                let slot = alloca_aggregate(cx, &return_type);
                cx.emit(Inst::Call {
                    dst: None,
                    callee,
                    args: lowered_args,
                    return_type: Type::Void,
                    sret: Some((slot, return_type.clone())),
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

            // an aggregate element is inline storage, and an aggregate value *is*
            // its address (the same convention `Access` and `Deref` follow), so
            // the element pointer is the value - loading it would read the first
            // word of the struct as if it were a handle. Only a scalar is loaded.
            if is_aggregate_ty(&element_ty, &cx.enums) {
                return Value::Reg(elem_ptr);
            }

            let dst = cx.fresh_reg();
            cx.emit(Inst::Load { dst, ptr: elem_ptr, ty: element_ty, align: None });

            Value::Reg(dst)
        }
    }
}

/// Whether `expr` names storage that already exists, and so has an address
/// `lower_lvalue` can hand back without making one.
///
/// This is deliberately the *lowering's* notion of a place rather than the
/// ownership pass's: a field of a temporary (`make().inner`) is a place here,
/// because the temporary's own lowering already put it in a slot.
fn is_place<'a>(cx: &LowerCtx<'a>, expr: &Expr<'a>) -> bool {
    match &expr.value {
        // a local or parameter is an alloca. A module-level global deliberately
        // is *not* a place: every one is emitted as an LLVM `constant`, and
        // `*self` carries no distinction between reading and writing, so
        // borrowing one directly would hand a writable pointer into read-only
        // memory - undefined behaviour for a `bump()` that the reader of
        // `G.display()` never asked to be different. Copying is well defined and
        // is what a constant *is*: a value, not storage. `&G` is another matter
        // and still yields the real address; the user wrote that one.
        ExprNode::Var(_) =>
            cx.resolved.get(&expr.id).is_some_and(|b| cx.env.contains_key(b)),
        ExprNode::Access { .. }
        | ExprNode::Index { .. }
        | ExprNode::Unary { op: UnaryOp::Deref, .. } => true,
        _ => false,
    }
}

/// The address to pass as `self` for a method whose receiver is `*self`.
///
/// A place has one already. Anything else is a *temporary* - `7.display()`,
/// `(a + b).show()`, `make().len()`, and a `const` global, which is a value
/// rather than storage - and `self` still has to point somewhere, so the value
/// is spilled into a slot of its own. That slot is exactly as long-lived as the
/// call, which is all a borrowing `*self` needs; a receiver that *owns*
/// something is rejected earlier, in the ownership pass, because nothing would
/// ever destroy it.
///
/// Before this existed the whole family reached `lower_lvalue` and panicked
/// there, so a method on a builtin - the very thing `extend i32` is for - could
/// not be called on anything but a variable.
fn lower_receiver<'a>(cx: &mut LowerCtx<'a>, base: &Expr<'a>) -> Register {
    if is_place(cx, base) {
        return lower_lvalue(cx, base);
    }
    spill_temporary(cx, base)
}

/// Give a temporary storage and return its address: evaluate the value and
/// `Store` it into a fresh slot as long-lived as the enclosing function. Used by
/// a `*self` receiver on a temporary and by an explicit `&<temporary>`, which
/// have the same need - a value that must be pointed at but owns no place.
///
/// A struct, a data enum and a fixed array are produced *in* storage already, and
/// the register lowering hands back is that storage, so no copy is made.
fn spill_temporary<'a>(cx: &mut LowerCtx<'a>, expr: &Expr<'a>) -> Register {
    let ty = cx.node_types[&expr.id].clone();
    let val = lower_expr(cx, expr);
    if is_aggregate_ty(&ty, &cx.enums) {
        let Value::Reg(r) = val else {
            unreachable!("an aggregate lowers to the register holding its storage")
        };
        return r;
    }
    let slot = cx.fresh_reg();
    cx.emit(Inst::Alloca { dst: slot, ty: ty.clone(), align: None });
    cx.emit(Inst::Store { ptr: slot, val, ty, align: None });
    slot
}

/// A value forced into a register, for the instructions that can only address
/// memory through one (`Store`, `Index`, `FieldPtr`).
///
/// Nearly every pointer-typed expression already lowers to a register; the
/// exception is a constant, `null` being the only one that exists. Round-tripping
/// it through a slot keeps those instructions total instead of panicking on
/// `ptr_write::<i32>(null::<*i32>(), 1)` - which is a segfault waiting to happen
/// either way, but should be the program's, not the compiler's. LLVM folds the
/// pair away immediately.
fn as_register<'a>(cx: &mut LowerCtx<'a>, val: Value, ty: &Type<'a>) -> Register {
    if let Value::Reg(r) = val { return r; }
    let slot = cx.fresh_reg();
    cx.emit(Inst::Alloca { dst: slot, ty: ty.clone(), align: None });
    cx.emit(Inst::Store { ptr: slot, val, ty: ty.clone(), align: None });
    let dst = cx.fresh_reg();
    cx.emit(Inst::Load { dst, ptr: slot, ty: ty.clone(), align: None });
    dst
}

pub(crate) fn lower_lvalue<'a>(cx: &mut LowerCtx<'a>, expr: &Expr<'a>) -> Register {
    match &expr.value {
        ExprNode::Var(name) => match cx.resolved.get(&expr.id).and_then(|b| cx.env.get(b)) {
            Some((reg, _)) => *reg,
            // a module-level global: its symbol is already a pointer to the
            // storage, so materializing the address is the whole job.
            None => {
                let dst = cx.fresh_reg();
                cx.emit(Inst::GlobalPtr { dst, name });
                dst
            }
        },

        ExprNode::Access { base, field } => {
            let base_val = lower_expr(cx, base);
            let base_ty = cx.node_types[&base.id].clone();
            // matches the typechecker's one-level auto-deref for `ptr.field`
            let struct_name = match &base_ty {
                Type::Pointer(inner) => inner.def(),
                t => t.def(),
            }.expect("field access on a non-aggregate rejected in typecheck");
            let field_index = cx.types[&struct_name].fields
                .iter()
                .position(|(fname, _)| fname == field)
                .unwrap();

            let field_ptr = cx.fresh_reg();
            cx.emit(Inst::FieldPtr {
                dst: field_ptr,
                struct_def: struct_name,
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

/// Stack storage for one aggregate of type `ty`, yielding a pointer to it.
///
/// `AllocaStruct` names a definition, so it cannot serve a `[T; N]`; the generic
/// `Alloca` can, now that an array renders as `[N x <element storage>]`.
pub(crate) fn alloca_aggregate<'a>(cx: &mut LowerCtx<'a>, ty: &Type<'a>) -> Register {
    let dst = cx.fresh_reg();
    match aggregate_def(ty, &cx.enums) {
        Some(def) => cx.emit(Inst::AllocaStruct { dst, def, align: None }),
        None => cx.emit(Inst::Alloca { dst, ty: ty.clone(), align: None }),
    }
    dst
}

/// Copy an aggregate of type `ty` from `src` into `dst`, both pointers to
/// storage of that type.
///
/// A struct is copied field by field (`copy_struct`); a fixed-size array is one
/// contiguous value, so a whole-array load/store does it. Anything else is a
/// scalar and stores directly. Callers that know they hold an aggregate get the
/// right copy without asking which kind it is.
pub(crate) fn copy_aggregate<'a>(cx: &mut LowerCtx<'a>, ty: &Type<'a>, src: Register, dst: Register) {
    match aggregate_def(ty, &cx.enums) {
        Some(def) => copy_struct(cx, def, src, dst),
        None => {
            let loaded = cx.fresh_reg();
            cx.emit(Inst::Load { dst: loaded, ptr: src, ty: ty.clone(), align: None });
            cx.emit(Inst::Store { ptr: dst, val: Value::Reg(loaded), ty: ty.clone(), align: None });
        }
    }
}

/// Deep-copy a struct from `src` to `dst`, both pointers to a struct of
/// `struct_def`. Recurses into nested struct fields (value semantics).
// TODO: use `llvm.memcpy` for large structs instead of a load/store per field.
pub(crate) fn copy_struct<'a>(cx: &mut LowerCtx<'a>, struct_def: DefId, src: Register, dst: Register) {
    let fields = cx.types[&struct_def].fields.clone();
    for (i, (_fname, fty)) in fields.iter().enumerate() {
        let src_field = cx.fresh_reg();
        let dst_field = cx.fresh_reg();
        cx.emit(Inst::FieldPtr { dst: src_field, struct_def, base: src, field_index: i });
        cx.emit(Inst::FieldPtr { dst: dst_field, struct_def, base: dst, field_index: i });
        match aggregate_def(fty, &cx.enums) {
            // for a nested aggregate, both field pointers point at inlined
            // sub-struct storage, so recurse to deep-copy it
            Some(inner) => {
                copy_struct(cx, inner, src_field, dst_field);
            }
            // everything else (scalars, arrays, slices, simd) is a single
            // value/aggregate that an LLVM load/store copies directly
            None => {
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
    enum_name: DefId,
    variant_path: &NameRef<'a>,
    tag: Const,
    args: &[Expr<'a>],
) -> Value {
    let base = match cx.store_target.take() {
        Some(slot) => slot,
        None => {
            let dst = cx.fresh_reg();
            cx.emit(Inst::AllocaStruct { dst, def: enum_name, align: None });
            dst
        }
    };
    // tag -> field 0
    let tag_ptr = cx.fresh_reg();
    cx.emit(Inst::FieldPtr { dst: tag_ptr, struct_def: enum_name, base, field_index: 0 });
    let repr = cx.enums[&enum_name].repr.clone();
    cx.emit(Inst::Store { ptr: tag_ptr, val: Value::Const(tag), ty: repr, align: None });

    if !args.is_empty() {
        let pstruct = cx.payloads[&(enum_name, variant_path.variant())];
        let payload_base = cx.fresh_reg();
        cx.emit(Inst::FieldPtr { dst: payload_base, struct_def: enum_name, base, field_index: 1 });
        for (i, arg) in args.iter().enumerate() {
            let field_ty = cx.types[&pstruct].fields[i].1.clone();
            let arg_ty = cx.node_types[&arg.id].clone();
            let field_val = lower_expr(cx, arg);
            let field_ptr = cx.fresh_reg();
            cx.emit(Inst::FieldPtr { dst: field_ptr, struct_def: pstruct, base: payload_base, field_index: i });
            match aggregate_def(&field_ty, &cx.enums) {
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
