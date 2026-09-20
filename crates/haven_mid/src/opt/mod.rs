//! Middle-end optimization passes.

mod noalias;
pub mod reach;

use crate::{
    mil::Module,
    typecheck::Context
};
use haven_common::ast::TopLevel;

/// Run every AST optimization in pipeline order.
pub fn optimize_ast<'a>(program: &mut Vec<TopLevel<'a>>, cx: &Context<'a>) {
    reach::prune_unreachable(program, &cx);
}

/// Run every MIL optimization in pipeline order.
pub fn optimize_mil(module: &mut Module<'_>) {
    noalias::annotate_pointer_params(module);
}
