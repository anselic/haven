//! MIL: the mid-level IR the AST lowers into before LLVM codegen.
//!
//! - `ir`:    the output data model (instructions, blocks, functions, module)
//! - `ctx`:   the lowering context and shared type/const helpers
//! - `expr`:  expression and lvalue lowering
//! - `lower`: statement/function lowering and the public `lower` entry point

mod ir;
mod ctx;
mod expr;
mod lower;

// Public surface: `use haven_mid::mil::*;` in the backend pulls these in.
pub use ir::*;
pub use ctx::{LowerCtx, LoopTargets, aggregate_def};
pub use lower::lower;
