//! Middle end: typecheck, check effects, monomorphize, and lower the AST to MIL.
// moved to haven_common (name resolution needs it); re-exported so the mid end's
// `crate::intrinsics::...` paths keep working.
pub use haven_common::intrinsics;
pub mod typecheck;
pub mod own;
pub mod effects;
pub mod mono;
pub mod opt;
pub mod mil;
