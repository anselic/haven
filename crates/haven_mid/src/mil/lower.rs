use std::collections::HashMap;
use haven_common::ast::*;
use crate::typecheck::EnumDef;
use super::ir::*;
use super::ctx::{LowerCtx, LoopTargets, int_const, pattern_variant_const, resolve_enum_ty, aggregate_struct_name, coerce, enum_const};
use super::expr::{lower_expr, lower_lvalue, copy_struct};

fn lower_stmt<'a>(cx: &mut LowerCtx<'a>, stmt: &Stmt<'a>) {
    match &stmt.value {
        StmtNode::Expr(expr) => {
            lower_expr(cx, expr); // side effects only, discard result
        }

        StmtNode::Block(stmts) => {
            for stmt in stmts {
                lower_stmt(cx, stmt);
            }
        }

        StmtNode::Declare { ty, value, .. } => {
            // resolve enum names so an enum local takes the scalar path below.
            let ty = &resolve_enum_ty(&cx.enums, ty);
            // this local's binding identity is the Declare stmt's node id, matching
            // what name resolution recorded for every use of it.
            let binding = Binding::Local(stmt.id);
            // A struct/array literal local has a slot hoisted to the entry block
            // (collect_locals pre-allocated it into env); point the literal at that
            // slot so it fills it in place instead of alloca'ing at the literal
            // site - which, inside a loop, would grow the stack every iteration.
            if matches!(ty, Type::Array(_, _) | Type::Struct { .. })
                && matches!(value.value, ExprNode::Struct { .. } | ExprNode::Slice(_))
            {
                let (slot, _) = cx.env[&binding]; // pre-allocated in the entry block
                cx.store_target = Some(slot);
                lower_expr(cx, value); // fills `slot`, consuming store_target
                return;
            }
            let val = lower_expr(cx, value);
            let value_ty = cx.node_types[&value.id].clone();
            match ty {
                Type::Array(_, _) => {
                    // a non-literal array value (e.g. a var) is already backed by a
                    // slot; adopt its register directly.
                    let arr_reg = match val {
                        Value::Reg(r) => r,
                        _ => unreachable!(),
                    };
                    cx.env.insert(binding, (arr_reg, ty.clone()));
                }
                // a struct or data-enum aggregate: same value-semantics rules.
                _ if aggregate_struct_name(ty).is_some() => {
                    let struct_name = aggregate_struct_name(ty).unwrap();
                    let src_reg = match val {
                        Value::Reg(r) => r,
                        _ => unreachable!(),
                    };
                    // a call (incl. a data-enum constructor) writes into a fresh
                    // slot nobody else aliases, so adopt it. anything else (Var,
                    // field access, ...) references someone else's storage and
                    // needs a copy for value semantics. (Literals handled above.)
                    if matches!(value.value, ExprNode::Call { .. }) {
                        cx.env.insert(binding, (src_reg, ty.clone()));
                    } else {
                        let dst_reg = cx.fresh_reg();
                        cx.emit(Inst::AllocaStruct { dst: dst_reg, name: struct_name, align: None });
                        copy_struct(cx, struct_name, src_reg, dst_reg);
                        cx.env.insert(binding, (dst_reg, ty.clone()));
                    }
                }
                _ => {
                    let val = coerce(cx, val, &value_ty, ty);
                    let (ptr, _) = cx.env[&binding]; // already alloca'd
                    cx.emit(Inst::Store { ptr, val, ty: ty.clone(), align: None });
                }
            }
        }

        StmtNode::Assign { left, value } => {
            let ptr = lower_lvalue(cx, left);  // see below
            let val = lower_expr(cx, value);
            let value_ty = cx.node_types[&value.id].clone();
            let ty = cx.node_types[&left.id].clone();
            let val = coerce(cx, val, &value_ty, &ty);
            cx.emit(Inst::Store { ptr, val, ty, align: None });
        }

        StmtNode::If { condition, then_branch, else_branch } => {
            cx.emit(Inst::Comment(format!("if {}", condition.value)));

            let cond_val = lower_expr(cx, condition);
            let cond_reg = match cond_val {
                Value::Reg(r) => r,
                _ => unreachable!(),
            };

            let then_block = cx.fresh_block();
            let else_block = cx.fresh_block();
            let merge_block = cx.fresh_block();

            cx.terminate(Terminator::Branch { cond: cond_reg, then_block, else_block });

            // then branch
            cx.current_block = then_block;
            cx.emit(Inst::Comment("if-then".to_string()));
            lower_stmt(cx, then_branch);
            // Check the *current* block, not `then_block`: if the branch body
            // contained nested control flow (a `while`/`if`), `current_block`
            // has advanced to that construct's merge block, and it - not the
            // already-terminated `then_block` - is what still needs a jump.
            if cx.blocks.iter().find(|b| b.id == cx.current_block).unwrap().terminator.is_none() {
                cx.terminate(Terminator::Jump(merge_block));
            }

            // else branch
            cx.current_block = else_block;
            if let Some(else_branch) = else_branch {
                cx.emit(Inst::Comment("if-else".to_string()));
                lower_stmt(cx, else_branch);
            }
            if cx.blocks.iter().find(|b| b.id == cx.current_block).unwrap().terminator.is_none() {
                cx.terminate(Terminator::Jump(merge_block));
            }

            cx.current_block = merge_block;
            cx.emit(Inst::Comment("if-merge".to_string()));
        }

        StmtNode::While { condition, body } => {
            let cond_block  = cx.fresh_block();
            let body_block  = cx.fresh_block();
            let merge_block = cx.fresh_block();

            cx.terminate(Terminator::Jump(cond_block));

            cx.current_block = cond_block;
            cx.emit(Inst::Comment(format!("while {}", condition.value)));
            let cond_val = lower_expr(cx, condition);
            let cond_reg = match cond_val {
                Value::Reg(r) => r,
                // if simple expression (e.g. while(true)) then we have to get
                // the value into a register first to branch on it
                v => {
                    let r = cx.fresh_reg();
                    cx.emit(Inst::Alloca { dst: r, ty: Type::Bool, align: None });
                    cx.emit(Inst::Store { ptr: r, val: v, ty: Type::Bool, align: None });
                    let s = cx.fresh_reg();
                    cx.emit(Inst::Load { dst: s, ptr: r, ty: Type::Bool, align: None });
                    s
                }
            };
            cx.terminate(Terminator::Branch {
                cond: cond_reg,
                then_block: body_block,
                else_block: merge_block,
            });

            cx.current_block = body_block;
            cx.emit(Inst::Comment("while body".to_string()));

            cx.loop_stack.push(LoopTargets {
                continue_block: cond_block,
                break_block: merge_block,
            });
            lower_stmt(cx, body);
            cx.loop_stack.pop();

            // add a jump to the merge block if the body doesn't already terminate
            // (e.g. via break/continue/return inside the last statement of the body)
            if cx.blocks.iter().find(|b| b.id == cx.current_block).unwrap().terminator.is_none() {
                cx.terminate(Terminator::Jump(cond_block));
            }

            cx.current_block = merge_block;
        }

        StmtNode::Match { scrutinee, arms } => {
            cx.emit(Inst::Comment(format!("match {}", scrutinee.value)));
            let scrut_ty = cx.node_types[&scrutinee.id].clone();
            let scrut_val = lower_expr(cx, scrutinee);
            // A data enum lowers to a pointer to its aggregate: the switch runs on
            // the tag (field 0), and payload bindings read off field 1. A field-less
            // enum / integer lowers to the scalar value directly.
            // the scrutinee's OWN (possibly monomorphized, e.g. "Option$i32") enum
            // name - patterns are matched against this, not their own textual
            // qualifier, since mono never rewrites match-pattern text (see
            // `pattern_variant_const`). Named distinctly from the `ename` bound
            // below by destructuring `agg` (same value, different scope/purpose).
            let switch_ename: Option<&'a str> = match &scrut_ty { Type::Enum { name, .. } => Some(*name), _ => None };
            let (switch_val, value_ty, agg): (Value, Type<'a>, Option<(&'a str, Register)>) =
                match &scrut_ty {
                    Type::Enum { repr, has_payload: true, name, .. } => {
                        let ptr = match scrut_val { Value::Reg(r) => r, _ => unreachable!() };
                        let tag_ptr = cx.fresh_reg();
                        cx.emit(Inst::FieldPtr { dst: tag_ptr, struct_name: name, base: ptr, field_index: 0 });
                        let tag = cx.fresh_reg();
                        cx.emit(Inst::Load { dst: tag, ptr: tag_ptr, ty: (**repr).clone(), align: None });
                        (Value::Reg(tag), (**repr).clone(), Some((*name, ptr)))
                    }
                    Type::Enum { repr, .. } => (scrut_val, (**repr).clone(), None),
                    t => (scrut_val, t.clone(), None),
                };

            let merge_block = cx.fresh_block();
            let mut cases: Vec<(Const, BlockId)> = Vec::new();
            let mut wildcard_block: Option<BlockId> = None;
            // one block per arm; bodies (and any payload bindings) are lowered
            // after the switch is wired. Carry the pattern to bind inside the arm.
            let mut arm_blocks: Vec<(BlockId, &Stmt<'a>, &Pattern<'a>)> = Vec::new();
            for (pat, body) in arms {
                let block = cx.fresh_block();
                match &pat.value {
                    PatternNode::Wildcard => wildcard_block = Some(block),
                    PatternNode::Int(n) => cases.push((int_const(&value_ty, *n), block)),
                    PatternNode::Path(p) => {
                        let c = pattern_variant_const(&cx.enums, switch_ename.expect("enum pattern validated in typecheck"), p);
                        cases.push((c, block));
                    }
                    PatternNode::Variant { path, .. } | PatternNode::StructVariant { path, .. } => {
                        let c = pattern_variant_const(&cx.enums, switch_ename.expect("enum pattern validated in typecheck"), path);
                        cases.push((c, block));
                    }
                    PatternNode::Bind(_) => unreachable!("bare binding rejected in typecheck"),
                }
                arm_blocks.push((block, body, pat));
            }

            // the default target is the wildcard arm, or - for an exhaustive enum
            // with no `_` - a synthesized unreachable block.
            let synth_default = wildcard_block.is_none();
            let default = wildcard_block.unwrap_or_else(|| cx.fresh_block());

            cx.terminate(Terminator::Switch { value: switch_val, value_ty, default, cases });

            for (block, body, pat) in arm_blocks {
                cx.current_block = block;
                // bind a data-variant's payload fields as views into the aggregate
                // (a FieldPtr per bound field, no fresh storage), keyed by each
                // binding's node id so uses in the body resolve to it.
                if let (PatternNode::Variant { path, fields }, Some((ename, ptr))) = (&pat.value, agg) {
                    let variant = path.split_once("::").unwrap().1;
                    let pstruct = crate::typecheck::enum_payload_struct_name(ename, variant);
                    let payload_base = cx.fresh_reg();
                    cx.emit(Inst::FieldPtr { dst: payload_base, struct_name: ename, base: ptr, field_index: 1 });
                    for (i, fpat) in fields.iter().enumerate() {
                        if let PatternNode::Bind(_) = &fpat.value {
                            let fty = cx.structs[pstruct][i].1.clone();
                            let fp = cx.fresh_reg();
                            cx.emit(Inst::FieldPtr { dst: fp, struct_name: pstruct, base: payload_base, field_index: i });
                            cx.env.insert(Binding::Local(fpat.id), (fp, fty));
                        }
                    }
                }
                // struct-style destructure: bind by field name, so the FieldPtr uses
                // the payload struct's index for that name (order in the pattern is
                // irrelevant). A `_` field is skipped, a `Bind` becomes a view.
                if let (PatternNode::StructVariant { path, fields }, Some((ename, ptr))) = (&pat.value, agg) {
                    let variant = path.split_once("::").unwrap().1;
                    let pstruct = crate::typecheck::enum_payload_struct_name(ename, variant);
                    let payload_base = cx.fresh_reg();
                    cx.emit(Inst::FieldPtr { dst: payload_base, struct_name: ename, base: ptr, field_index: 1 });
                    for (fname, fpat) in fields {
                        if let PatternNode::Bind(_) = &fpat.value {
                            let idx = cx.structs[pstruct].iter().position(|(n, _)| n == fname).unwrap();
                            let fty = cx.structs[pstruct][idx].1.clone();
                            let fp = cx.fresh_reg();
                            cx.emit(Inst::FieldPtr { dst: fp, struct_name: pstruct, base: payload_base, field_index: idx });
                            cx.env.insert(Binding::Local(fpat.id), (fp, fty));
                        }
                    }
                }
                lower_stmt(cx, body);
                if cx.blocks.iter().find(|b| b.id == cx.current_block).unwrap().terminator.is_none() {
                    cx.terminate(Terminator::Jump(merge_block));
                }
            }
            if synth_default {
                cx.current_block = default;
                cx.terminate(Terminator::Unreachable);
            }

            cx.current_block = merge_block;
        }

        StmtNode::Continue => {
            cx.emit(Inst::Comment("continue".to_string()));
            cx.terminate(Terminator::Jump(cx.loop_stack.last().unwrap().continue_block));
        }
        StmtNode::Break => {
            cx.emit(Inst::Comment("break".to_string()));
            cx.terminate(Terminator::Jump(cx.loop_stack.last().unwrap().break_block));
        }

        StmtNode::Return(expr) => {
            let val = lower_expr(cx, expr);
            let value_ty = cx.node_types[&expr.id].clone();
            let ret_ty = cx.current_return_type.clone();
            let struct_ret = aggregate_struct_name(&ret_ty);
            if let Some(name) = struct_ret {
                // copy the aggregate into the caller-provided sret slot, ret void
                let src = match val {
                    Value::Reg(r) => r,
                    _ => unreachable!(),
                };
                let dst = cx.sret_param.expect("struct-returning function has no sret slot");
                copy_struct(cx, name, src, dst);
                cx.terminate(Terminator::Return(None));
            } else {
                let val = coerce(cx, val, &value_ty, &ret_ty);
                cx.terminate(Terminator::Return(Some((val, ret_ty))));
            }
        }
    }
}

/// Recursively collect all local variable declarations in the function body,
/// including nested ones in blocks and branches.
/// This is for emitting all Alloca instructions upfront in the entry block
fn collect_locals<'a>(stmt: &Stmt<'a>, enums: &HashMap<&'a str, EnumDef<'a>>, out: &mut Vec<(usize, &'a str, Type<'a>)>) {
    match &stmt.value {
        StmtNode::Declare { name, ty, value } => {
            // resolve enum names so an enum local is pre-allocated as its scalar
            // repr (not skipped as a struct), matching the scalar Declare path.
            let ty = resolve_enum_ty(enums, ty);
            // key by the Declare stmt's node id (its binding identity)
            match ty {
                // A struct/array/data-enum *literal* used to alloca at its own
                // site, which sits inside any enclosing loop body and grows the
                // stack every iteration. Hoist one slot to the entry block; the
                // Declare arm fills it in place via store_target. Non-literal
                // aggregate declares (a call, incl. a data-enum constructor, or a
                // var copy) still adopt or copy in lower_stmt and are not
                // pre-allocated here.
                Type::Array(_, _) => {
                    if matches!(value.value, ExprNode::Struct { .. } | ExprNode::Slice(_)) {
                        out.push((stmt.id, name, ty));
                    }
                }
                _ if aggregate_struct_name(&ty).is_some() => {
                    if matches!(value.value, ExprNode::Struct { .. } | ExprNode::Slice(_)) {
                        out.push((stmt.id, name, ty));
                    }
                }
                _ => out.push((stmt.id, name, ty)),
            }
        }
        StmtNode::Block(stmts) => {
            for s in stmts { collect_locals(s, enums, out); }
        }
        StmtNode::If { then_branch, else_branch, .. } => {
            collect_locals(then_branch, enums, out);
            if let Some(e) = else_branch { collect_locals(e, enums, out); }
        }
        StmtNode::While { body, .. } => {
            collect_locals(body, enums, out);
        }
        StmtNode::Match { arms, .. } => {
            for (_pat, body) in arms { collect_locals(body, enums, out); }
        }
        // rest should have no declarations
        _ => {}
    }
}

/// Lower a global's initializer expression to a constant. Scalar literals
/// (optionally negated), struct/array literals of constants, and function names
/// are supported; the typechecker has already rejected anything else, so the
/// unreachable arms indicate a compiler bug. Struct/array field types are read
/// from `cx.structs` / `cx.node_types`.
fn lower_const_init<'a>(
    cx: &mut LowerCtx<'a>,
    expr: &Expr<'a>,
) -> ConstInit<'a> {
    match &expr.value {
        ExprNode::Bool(b)    => ConstInit::Scalar(Const::Bool(*b)),
        ExprNode::Int8(n)    => ConstInit::Scalar(Const::Int8(*n)),
        ExprNode::Int32(n)   => ConstInit::Scalar(Const::Int32(*n)),
        ExprNode::Int64(n)   => ConstInit::Scalar(Const::Int64(*n)),
        ExprNode::Uint8(n)   => ConstInit::Scalar(Const::Uint8(*n)),
        ExprNode::Uint32(n)  => ConstInit::Scalar(Const::Uint32(*n)),
        ExprNode::Uint64(n)  => ConstInit::Scalar(Const::Uint64(*n)),
        ExprNode::Float32(f) => ConstInit::Scalar(Const::Float32(*f)),
        ExprNode::Float64(f) => ConstInit::Scalar(Const::Float64(*f)),
        ExprNode::Unary { op: UnaryOp::Neg, operand } => match &operand.value {
            ExprNode::Int8(n)    => ConstInit::Scalar(Const::Int8(-*n)),
            ExprNode::Int32(n)   => ConstInit::Scalar(Const::Int32(-*n)),
            ExprNode::Int64(n)   => ConstInit::Scalar(Const::Int64(-*n)),
            ExprNode::Uint8(n)   => ConstInit::Scalar(Const::Uint8(n.wrapping_neg())),
            ExprNode::Uint32(n)  => ConstInit::Scalar(Const::Uint32(n.wrapping_neg())),
            ExprNode::Uint64(n)  => ConstInit::Scalar(Const::Uint64(n.wrapping_neg())),
            ExprNode::Float32(f) => ConstInit::Scalar(Const::Float32(-*f)),
            ExprNode::Float64(f) => ConstInit::Scalar(Const::Float64(-*f)),
            other => unreachable!("non-constant global initializer reached lowering: -{}", other),
        },
        // struct literal: pair each field's declared type with its constant.
        // typecheck guarantees the fields match the definition in order.
        ExprNode::Struct { name, fields, .. } => {
            let defs = cx.structs[name].clone();
            let mut inits = Vec::with_capacity(fields.len());
            for ((_, fty), (_, fexpr)) in defs.iter().zip(fields.iter()) {
                inits.push((fty.clone(), lower_const_init(cx, fexpr)));
            }
            ConstInit::Struct(inits)
        }
        // array literal: the element type comes from the inferred array type.
        ExprNode::Slice(elements) => {
            let elem_ty = match cx.node_types.get(&expr.id) {
                Some(Type::Array(inner, _)) => (**inner).clone(),
                other => unreachable!("array const initializer has non-array type: {other:?}"),
            };
            let mut inits = Vec::with_capacity(elements.len());
            for elem in elements {
                inits.push(lower_const_init(cx, elem));
            }
            ConstInit::Array(elem_ty, inits)
        }
        // an `Enum::Variant` in a const initializer is its discriminant constant.
        ExprNode::Var(name) if enum_const(&cx.enums, name).is_some() =>
            ConstInit::Scalar(enum_const(&cx.enums, name).unwrap()),
        // a bare name in a const initializer is a function (typecheck ensured it);
        // its address @name is the constant.
        ExprNode::Var(name) => ConstInit::FnAddr(name),
        other => unreachable!("non-constant global initializer reached lowering: {}", other),
    }
}

fn lower_function<'a>(cx: &mut LowerCtx<'a>, func: &TopLevel<'a>)
-> (Vec<(Register, Type<'a>)>, Option<(Register, &'a str)>) {
    match &func.value {
        TopLevelNode::Function { attributes, params, return_type, body, .. } => {
            let entry = cx.fresh_block();
            cx.current_block = entry;
            // resolve enum names in the declared return type (raw parser type).
            let return_type = &resolve_enum_ty(&cx.enums, return_type);
            cx.current_return_type = return_type.clone();

            // struct / data-enum returns are lowered with an sret out-pointer;
            // allocate its register up front so `return` can copy into it.
            let sret = if let Some(name) = aggregate_struct_name(return_type) {
                let r = cx.fresh_reg();
                cx.sret_param = Some(r);
                Some((r, name))
            } else {
                cx.sret_param = None;
                None
            };

            let mut param_regs = Vec::new();
            for (name, ty) in params {
                // resolve enum names in the declared param type (raw parser type)
                // so an enum param takes the scalar path, not the by-value struct one.
                let ty = &resolve_enum_ty(&cx.enums, ty);
                if attributes.iter().any(|a| a.value.name == "export")
                && matches!(ty, Type::Slice(_)) {
                    // split the fat pointer into 2 parameters for the data pointer and length
                    // e.g. foo(buf: T[]) -> @foo(*T %buf_ptr, i32 %buf_len)
                    let ptr_reg = cx.fresh_reg();
                    let len_reg = cx.fresh_reg();

                    let inner_ty = match ty {
                        Type::Slice(inner) => *inner.clone(),
                        _ => unreachable!(),
                    };

                    let fat0 = cx.fresh_reg();
                    cx.emit(Inst::Comment(format!("params {}: {} (splitted)", name, ty)));
                    cx.emit(Inst::InsertValue {
                        dst: fat0,
                        elem: Value::Const(Const::Undef),
                        ty: Type::Pointer(Box::new(inner_ty.clone())),
                        val: Value::Reg(ptr_reg),
                        index: 0,
                    });
                    let fat1 = cx.fresh_reg();
                    cx.emit(Inst::InsertValue {
                        dst: fat1,
                        elem: Value::Reg(fat0),
                        ty: Type::Int32,
                        val: Value::Reg(len_reg),
                        index: 1,
                    });

                    // store the reconstructed fat pointer into the alloca'd local
                    let param_ptr = cx.fresh_reg();
                    cx.emit(Inst::Alloca { dst: param_ptr, ty: ty.clone(), align: None });
                    cx.emit(Inst::Store { ptr: param_ptr, val: Value::Reg(fat1), ty: ty.clone(), align: None });
                    cx.env.insert(Binding::Param(name), (param_ptr, ty.clone()));

                    // push BOTH as incoming params
                    param_regs.push((ptr_reg, Type::Pointer(Box::new(inner_ty.clone()))));
                    param_regs.push((len_reg, Type::Int32));
                } else if let Some(struct_name) = aggregate_struct_name(ty) {
                    // pass-by-value (struct or data enum): caller still hands us a
                    // `ptr` to its aggregate, but we copy into our own local slot on
                    // entry so mutations don't leak back. For pass-by-reference the
                    // user writes `*Stereo` explicitly - that falls through to the
                    // `_` arm and stays a plain pointer.
                    let param_reg = cx.fresh_reg();
                    cx.emit(Inst::Comment(format!("params {}: {} (by value, copied on entry)", name, ty)));
                    param_regs.push((param_reg, ty.clone()));

                    let local_reg = cx.fresh_reg();
                    cx.emit(Inst::AllocaStruct { dst: local_reg, name: struct_name, align: None });
                    copy_struct(cx, struct_name, param_reg, local_reg);
                    cx.env.insert(Binding::Param(name), (local_reg, ty.clone()));
                } else {
                    // actual register for the parameter value
                    // e.g. @foo(T %param_regN, ...)
                    // and then use param_ptr so we can treat it like normal mutable locals
                    let param_reg = cx.fresh_reg();
                    let param_ptr = cx.fresh_reg();
                    cx.emit(Inst::Comment(format!("params {}: {}", name, ty)));
                    cx.emit(Inst::Alloca { dst: param_ptr, ty: ty.clone(), align: None });
                    cx.emit(Inst::Store { ptr: param_ptr, val: Value::Reg(param_reg), ty: ty.clone(), align: None });

                    param_regs.push((param_reg, ty.clone()));
                    cx.env.insert(Binding::Param(name), (param_ptr, ty.clone()));
                }
            }

            // collect all local variable declarations in the function body and emit Allocas for them.
            // each local is keyed by its Declare's node id, so shadowed same-named
            // locals get distinct slots (uses are routed by name resolution).
            let mut locals = Vec::new();
            for stmt in body {
                collect_locals(stmt, &cx.enums, &mut locals);
            }
            for (decl_id, name, ty) in locals {
                let local_reg = cx.fresh_reg();
                cx.emit(Inst::Comment(format!("local {}: {}", name, ty)));
                match &ty {
                    // struct/array/data-enum literal locals are hoisted here
                    // (collect_locals) so the slot is allocated once, not per loop
                    // iteration.
                    Type::Array(inner, length) => {
                        cx.emit(Inst::AllocaArray { dst: local_reg, ty: (**inner).clone(), length: length.expect_lit() });
                    }
                    _ if aggregate_struct_name(&ty).is_some() => {
                        let struct_name = aggregate_struct_name(&ty).unwrap();
                        cx.emit(Inst::AllocaStruct { dst: local_reg, name: struct_name, align: None });
                    }
                    _ => {
                        cx.emit(Inst::Alloca { dst: local_reg, ty: ty.clone(), align: None });
                    }
                }
                cx.env.insert(Binding::Local(decl_id), (local_reg, ty.clone()));
            }

            cx.emit(Inst::Comment("function body".to_string()));

            for stmt in body {
                lower_stmt(cx, stmt);
            }

            if cx.blocks.iter().find(|b| b.id == cx.current_block).unwrap().terminator.is_none() {
                if *return_type == Type::Void {
                    cx.terminate(Terminator::Return(None));
                } else {
                    // typecheck's all-paths-return analysis guarantees every path
                    // of a non-void function returns, so a block still open at the
                    // end of the body is provably dead (e.g. the merge block after
                    // an if/else whose branches both return). mark it unreachable.
                    cx.terminate(Terminator::Unreachable);
                }
            }

            (param_regs, sret)
        }
        TopLevelNode::Extern { .. } => (vec![], None),
        TopLevelNode::Struct { .. } => (vec![], None),
        TopLevelNode::Global { .. } => (vec![], None),
        TopLevelNode::Enum { .. } => (vec![], None),
        TopLevelNode::Extend { .. } => unreachable!("extend desugared before lowering"),
    }
}

/// Whether `ty` is a bare reference to one of `type_params` (an unresolved
/// generic slot like `T`, which parses as `Type::Struct { name: "T", args: [] }`).
/// Used to erase a generic extern's type-param params from the declaration.
fn is_generic_type(type_params: &[&str], ty: &Type<'_>) -> bool {
    matches!(ty, Type::Struct { name, args } if args.is_empty() && type_params.contains(name))
}

pub fn lower<'a>(
    program: &[TopLevel<'a>],
    typecheck_context: &crate::typecheck::Context<'a>,
) -> Module<'a> {
    let mut cx = LowerCtx {
        reg_counter: 0,
        block_counter: 0,
        current_block: BlockId(0),
        blocks: vec![],
        loop_stack: vec![],
        env: HashMap::new(),
        globals: HashMap::new(),
        structs: typecheck_context.structs.clone(),
        enums: typecheck_context.enums.clone(),
        node_types: typecheck_context.node_types.clone(),
        resolved: typecheck_context.resolved.clone(),
        method_calls: typecheck_context.method_calls.clone(),
        current_return_type: Type::Void,
        sret_param: None,
        store_target: None,
        strings: Vec::new(),
    };

    let mut functions = Vec::new();
    let mut externs = Vec::new();
    let mut structs = Vec::new();
    let mut globals = Vec::new();
    let mut enum_unions = HashMap::new();

    // register every global up front so a function can reference one declared
    // later in the file.
    for toplevel in program {
        if let TopLevelNode::Global { name, ty, .. } = &toplevel.value {
            cx.globals.insert(name, ty.clone());
        }
    }

    for toplevel in program {
        match &toplevel.value {
            TopLevelNode::Function { name, attributes, return_type, generics, .. } => {
                // generic functions are templates; without monomorphization there
                // is nothing to lower until they're instantiated at a call site
                if !generics.is_empty() {
                    continue;
                }
                let (param_regs, sret) = lower_function(&mut cx, toplevel);
                functions.push(Function {
                    name,
                    attributes: attributes.clone(),
                    params: param_regs,
                    return_type: resolve_enum_ty(&cx.enums, return_type),
                    blocks: cx.blocks.clone(),
                    sret,
                });
                // reset for next function. env holds only this function's
                // params/locals; clearing keeps stale names from a prior function
                // from masquerading as in-scope (the call dispatch consults it).
                cx.reg_counter = 0;
                cx.block_counter = 0;
                cx.blocks.clear();
                cx.env.clear();
            }
            TopLevelNode::Extern { name, attributes, generics, params, return_type, .. } => {
                // a generic extern's type-param slots (`arg: T`) have no concrete
                // layout, so drop them from the declaration and keep only the
                // concrete leading params. the call site still passes the concrete
                // args; the underlying C symbol (e.g. printf) accepts them.
                let type_params: Vec<&'a str> = generics.iter().filter_map(|g| match g {
                    GenericParam::Type(n) => Some(*n),
                    GenericParam::Const(_, _) => None,
                }).collect();
                let concrete_params: Vec<Type<'a>> = params.iter()
                    .filter(|(_, ty)| !is_generic_type(&type_params, ty))
                    .map(|(_, ty)| resolve_enum_ty(&cx.enums, ty))
                    .collect();
                externs.push(ExternDecl {
                    name,
                    attributes: attributes.clone(),
                    params: concrete_params,
                    return_type: resolve_enum_ty(&cx.enums, return_type),
                });
            }
            // generic struct templates carry `Param` fields and are never laid out
            // directly; monomorphization emits concrete instances. skip them here,
            // mirroring how generic functions are skipped above.
            TopLevelNode::Struct { generics, .. } if !generics.is_empty() => {}
            TopLevelNode::Struct { name, fields, .. } => {
                // resolve enum-typed fields to their integer repr; a bare enum name
                // otherwise emits an undefined `%EnumName` in the struct type.
                let fields = fields.iter()
                    .map(|(fname, fty)| (*fname, resolve_enum_ty(&cx.enums, fty)))
                    .collect();
                structs.push((*name, fields));
            }
            TopLevelNode::Global { name, attributes, ty, value, .. } => {
                let init = lower_const_init(&mut cx, value);
                globals.push(Global {
                    name,
                    export: attributes.iter().any(|a| a.value.name == "export"),
                    ty: ty.clone(),
                    init,
                });
            }
            // a field-less enum is erased to its integer repr and emits nothing. A
            // data enum emits its synthetic aggregate: the per-variant payload
            // structs (`Msg$Note`) followed by the tagged aggregate (`Msg`), both
            // built in typecheck's forward-decl pass and looked up here so the
            // backend renders them as ordinary `%Name = type { .. }` structs. Keys
            // carry `'a`, so fetch them via `get_key_value`.
            TopLevelNode::Enum { name, .. } => {
                if cx.enums[name].has_payload {
                    let variants: Vec<&'a str> = cx.enums[name].payloads.keys().copied().collect();
                    let mut pstruct_names = Vec::with_capacity(variants.len());
                    for v in variants {
                        let sname = crate::typecheck::enum_payload_struct_name(name, v);
                        if let Some((k, f)) = cx.structs.get_key_value(sname) {
                            structs.push((*k, f.clone()));
                            pstruct_names.push(*k);
                        }
                    }
                    if let Some((k, f)) = cx.structs.get_key_value(*name) {
                        structs.push((*k, f.clone()));
                    }
                    enum_unions.insert(*name, pstruct_names);
                }
            }
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before lowering"),
        }
    }

    Module { functions, externs, structs, globals, strings: cx.strings, enum_unions }
}