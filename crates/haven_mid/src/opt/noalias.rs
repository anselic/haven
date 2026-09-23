//! Infer proven aliasing facts and lower explicit `@noalias` contracts into MIL.
//!
//! Haven's ordinary `*T` permits overlapping pointers. A function marked
//! `@noalias` promises that memory accessed through each `*T` parameter is not
//! accessed through an unrelated pointer for the duration of the call. The
//! backend exposes that promise to LLVM, enabling loop-invariant field loads,
//! scalar replacement, state promotion, and vectorization.
//!
//! Inference is per parameter and body-local: a leaf function whose accesses all
//! use one parameter or fresh stack storage needs no caller-side disjointness
//! assumption. Everything else still requires an explicit contract. In particular,
//! this pass does not infer facts from the current set of call sites.

use std::collections::{HashMap, HashSet};

use haven_common::ast::Type;

use crate::mil::{Function, Inst, Module, Register, Value};

pub(super) fn annotate_pointer_params(module: &mut Module<'_>) {
    for function in &mut module.functions {
        if !function
            .attributes
            .iter()
            .any(|a| a.value.name == "noalias")
        {
            infer_pointer_params(function);
            continue;
        }

        function.noalias_params.extend(
            function
                .params
                .iter()
                .filter_map(|(register, ty)| matches!(ty, Type::Pointer(_)).then_some(*register)),
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    Local,
    Param(Register),
}

/// Prove the contract from the body, independently of the callers. All accesses
/// must be through one parameter (including its derived pointers) or fresh stack
/// storage. Multiple accessed parameters may overlap, even if their types differ.
/// Calls are deliberately excluded: even a call with no pointer arguments may
/// access the parameter through a global or callback. Unknown addresses, including
/// pointers loaded from pointees, also prevent inference.
///
/// This does not infer ownership, nonnull, or return-value noalias. It only adds
/// parameter facts, and only for parameters actually accessed by the body.
fn infer_pointer_params(function: &mut Function<'_>) {
    let instructions: Vec<_> = function.blocks.iter()
        .flat_map(|block| &block.instructions).collect();
    if instructions.iter().any(|inst| matches!(inst, Inst::Call { .. })) {
        return;
    }

    let mut origins = HashMap::new();
    for (reg, ty) in &function.params {
        if matches!(ty, Type::Pointer(_)) {
            origins.insert(*reg, Origin::Param(*reg));
        }
    }
    if origins.is_empty() {
        return;
    }

    // Lowering spills parameters to mutable stack slots. Recover provenance only
    // for single-store slots whose address is used exclusively by direct loads
    // and stores. Taking the slot's address (even for a GEP or cast) invalidates
    // this shortcut, since an indirect store could change its pointer value.
    let mut slots = HashSet::new();
    let mut stores: HashMap<Register, Vec<&Value>> = HashMap::new();
    let mut exposed = HashSet::new();
    for inst in &instructions {
        match inst {
            Inst::Alloca { dst, ty, .. } => {
                origins.insert(*dst, Origin::Local);
                if matches!(ty, Type::Pointer(_)) {
                    slots.insert(*dst);
                }
            }
            Inst::AllocaArray { dst, .. } | Inst::AllocaStruct { dst, .. } => {
                origins.insert(*dst, Origin::Local);
            }
            Inst::Store { ptr, val, .. } => {
                stores.entry(*ptr).or_default().push(val);
                expose_value(&mut exposed, val);
            }
            Inst::Load { .. } | Inst::Comment(_) | Inst::Sizeof { .. }
            | Inst::GlobalPtr { .. } => {}
            Inst::FieldPtr { base, .. } | Inst::TupleFieldPtr { base, .. } => { exposed.insert(*base); }
            Inst::IndexArray { array, .. } => { exposed.insert(*array); }
            Inst::Index { slice, index, .. } => {
                exposed.insert(*slice);
                expose_value(&mut exposed, index);
            }
            Inst::Unary { val, .. } | Inst::Extend { val, .. }
            | Inst::ExtractValue { val, .. } | Inst::Splat { val, .. } => {
                expose_value(&mut exposed, val);
            }
            Inst::PtrToInt { ptr, .. } => { expose_value(&mut exposed, ptr); }
            Inst::Binary { lhs, rhs, .. } => {
                expose_value(&mut exposed, lhs);
                expose_value(&mut exposed, rhs);
            }
            Inst::InsertValue { elem, val, .. } => {
                expose_value(&mut exposed, elem);
                expose_value(&mut exposed, val);
            }
            Inst::Shuffle { v0, v1, .. } => {
                expose_value(&mut exposed, v0);
                expose_value(&mut exposed, v1);
            }
            Inst::Call { .. } => unreachable!("calls excluded above"),
        }
    }
    slots.retain(|slot| !exposed.contains(slot)
        && stores.get(slot).is_some_and(|values| values.len() == 1));

    // Registers are SSA, but block order need not be definition order. Iterate
    // until no further known origins can be propagated. Missing facts always
    // mean unknown; in particular a cyclic spill cannot bootstrap its own proof.
    loop {
        let before = origins.len();
        for inst in &instructions {
            let derived = match inst {
                Inst::FieldPtr { dst, base, .. } => Some((*dst, *base)),
                Inst::Index { dst, slice, .. } => Some((*dst, *slice)),
                Inst::IndexArray { dst, array, .. } => Some((*dst, *array)),
                Inst::Extend { dst, val: Value::Reg(src), from_ty: Type::Pointer(_),
                    to_ty: Type::Pointer(_), .. } => Some((*dst, *src)),
                Inst::Load { dst, ptr, ty: Type::Pointer(_), .. }
                    if slots.contains(ptr) => match stores[ptr][0] {
                        Value::Reg(src) => Some((*dst, *src)),
                        Value::Const(_) => None,
                    },
                _ => None,
            };
            if let Some((dst, src)) = derived {
                if let Some(origin) = origins.get(&src).copied() {
                    origins.insert(dst, origin);
                }
            }
        }
        if origins.len() == before { break; }
    }

    let mut accessed = None;
    for inst in instructions {
        let ptr = match inst {
            Inst::Load { ptr, .. } | Inst::Store { ptr, .. } => ptr,
            _ => continue,
        };
        match origins.get(ptr) {
            Some(Origin::Local) => {}
            Some(Origin::Param(reg)) => {
                if accessed.is_some_and(|previous| previous != *reg) { return; }
                accessed = Some(*reg);
            }
            None => return,
        }
    }
    if let Some(reg) = accessed {
        function.noalias_params.insert(reg);
    }
}

fn expose_value(exposed: &mut HashSet<Register>, value: &Value) {
    if let Value::Reg(reg) = value { exposed.insert(*reg); }
}
