//! Shared foundations for the haven compiler: the AST (with spans) and the
//! diagnostic reporter. Everything downstream (front/mid/back) builds on these.
pub mod ast;
pub mod diag;
// lives here rather than in the mid end because name resolution has to recognize
// intrinsic names too (they're in no module's symbol table, so an unresolved-name
// error would otherwise fire on every `sizeof`/`null`/`__simd_*` call).
pub mod intrinsics;
pub mod layout;
