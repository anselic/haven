//! Lower the explicit `@noalias` contract into MIL parameter facts.
//!
//! Haven's ordinary `*T` permits overlapping pointers. A function marked
//! `@noalias` promises that memory accessed through each `*T` parameter is not
//! accessed through an unrelated pointer for the duration of the call. The
//! backend exposes that promise to LLVM, enabling loop-invariant field loads,
//! scalar replacement, state promotion, and vectorization.

use haven_common::ast::Type;

use crate::mil::Module;

pub(super) fn annotate_pointer_params(module: &mut Module<'_>) {
    for function in &mut module.functions {
        if !function
            .attributes
            .iter()
            .any(|a| a.value.name == "noalias")
        {
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
