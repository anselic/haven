//! A fixed-size temporary constructed in a loop must not allocate another
//! stack slot on every iteration. Reuse its slot only when no pointer to that
//! storage can survive the iteration.

use std::collections::HashSet;

use haven_common::ast::Type;

use crate::mil::{BlockId, Function, Inst, Module, Register, Terminator, Value};

fn contains(value: &Value, aliases: &HashSet<Register>) -> bool {
    matches!(value, Value::Reg(reg) if aliases.contains(reg))
}

fn escapes(function: &Function<'_>, root: Register) -> bool {
    let mut aliases = HashSet::from([root]);

    // GEPs point into the same allocation. Collect their descendants before
    // checking uses, regardless of the order of the basic blocks.
    loop {
        let mut changed = false;
        for inst in function.blocks.iter().flat_map(|b| &b.instructions) {
            let derived = match inst {
                Inst::IndexArray { dst, array, .. } if aliases.contains(array) => Some(*dst),
                Inst::FieldPtr { dst, base, .. } if aliases.contains(base) => Some(*dst),
                Inst::TupleFieldPtr { dst, base, .. } if aliases.contains(base) => Some(*dst),
                _ => None,
            };
            if let Some(dst) = derived {
                changed |= aliases.insert(dst);
            }
        }
        if !changed {
            break;
        }
    }

    for block in &function.blocks {
        for inst in &block.instructions {
            let leaked = match inst {
                Inst::Store { val, .. } => contains(val, &aliases),
                Inst::Call { callee, args, .. } => {
                    let indirect =
                        matches!(callee, crate::mil::Callee::Indirect(v) if contains(v, &aliases));
                    indirect
                        || args.iter().any(|(val, ty)| {
                            contains(val, &aliases)
                                && !matches!(ty, Type::Array(..) | Type::Tuple(..))
                        })
                    // Array/tuple parameters cross the ABI by value. The
                    // callee receives registers or its own byval copy.
                }
                Inst::Unary { val, .. } => contains(val, &aliases),
                Inst::Binary { lhs, rhs, .. } => contains(lhs, &aliases) || contains(rhs, &aliases),
                Inst::Index { slice, index, .. } => {
                    aliases.contains(slice) || contains(index, &aliases)
                }
                Inst::InsertValue { elem, val, .. } => {
                    contains(elem, &aliases) || contains(val, &aliases)
                }
                Inst::ExtractValue { val, .. } => contains(val, &aliases),
                Inst::PtrToInt { ptr, .. } => contains(ptr, &aliases),
                Inst::Extend { val, .. } | Inst::Splat { val, .. } => contains(val, &aliases),
                Inst::Shuffle { v0, v1, .. } => contains(v0, &aliases) || contains(v1, &aliases),
                Inst::Load { .. }
                | Inst::IndexArray { .. }
                | Inst::FieldPtr { .. }
                | Inst::TupleFieldPtr { .. }
                | Inst::Comment(_)
                | Inst::GlobalPtr { .. }
                | Inst::Alloca { .. }
                | Inst::AllocaArray { .. }
                | Inst::AllocaStruct { .. }
                | Inst::Sizeof { .. } => false,
            };
            if leaked {
                return true;
            }
        }
        let leaked = match &block.terminator {
            Some(Terminator::Return(Some((val, _)))) => contains(val, &aliases),
            Some(Terminator::Branch { cond, .. }) => contains(cond, &aliases),
            Some(Terminator::Switch { value, .. }) => contains(value, &aliases),
            _ => false,
        };
        if leaked {
            return true;
        }
    }
    false
}

fn successors(term: &Terminator<'_>) -> Vec<BlockId> {
    match term {
        Terminator::Jump(next) => vec![*next],
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        Terminator::Switch { default, cases, .. } => {
            let mut next = vec![*default];
            next.extend(cases.iter().map(|(_, block)| *block));
            next
        }
        Terminator::Return(_) | Terminator::Unreachable => vec![],
    }
}

fn is_cyclic(function: &Function<'_>, start: BlockId) -> bool {
    let mut seen = HashSet::new();
    let mut pending = function
        .blocks
        .iter()
        .find(|b| b.id == start)
        .and_then(|b| b.terminator.as_ref())
        .map(successors)
        .unwrap_or_default();
    while let Some(block) = pending.pop() {
        if block == start {
            return true;
        }
        if !seen.insert(block) {
            continue;
        }
        if let Some(term) = function
            .blocks
            .iter()
            .find(|b| b.id == block)
            .and_then(|b| b.terminator.as_ref())
        {
            pending.extend(successors(term));
        }
    }
    false
}

pub(super) fn hoist_nonescaping_allocas(module: &mut Module<'_>) {
    for function in &mut module.functions {
        if function.blocks.is_empty() {
            continue;
        }
        let mut hoist = HashSet::new();
        for block in function.blocks.iter().skip(1) {
            if !is_cyclic(function, block.id) {
                continue;
            }
            for inst in &block.instructions {
                let root = match inst {
                    Inst::Alloca { dst, .. }
                    | Inst::AllocaArray { dst, .. }
                    | Inst::AllocaStruct { dst, .. } => *dst,
                    _ => continue,
                };
                if !escapes(function, root) {
                    hoist.insert(root);
                }
            }
        }

        let mut slots = Vec::new();
        for block in function.blocks.iter_mut().skip(1) {
            block.instructions.retain(|inst| {
                let root = match inst {
                    Inst::Alloca { dst, .. }
                    | Inst::AllocaArray { dst, .. }
                    | Inst::AllocaStruct { dst, .. } => *dst,
                    _ => return true,
                };
                if hoist.contains(&root) {
                    slots.push(inst.clone());
                    false
                } else {
                    true
                }
            });
        }
        function.blocks[0].instructions.splice(0..0, slots);
    }
}
