//! When `havenc` crashes under `haven build`, the build must say so - "the
//! compiler crashed", not "compilation failed" - because the two send the user
//! to different places: their own code, or a bug report.
//!
//! `HAVENC_INTERNAL_PANIC` makes the child `havenc` panic at startup; it is the
//! environment twin of the hidden `--internal-panic` flag, for exactly this
//! caller, which cannot add to the compiler's argv.

use std::process::Command;

mod common;
use common::{err, scaffold, HAVEN};

#[test]
fn a_compiler_crash_is_reported_as_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("haven.toml", "[project]\nname = \"app\"\nkind = [\"bin\"]\n"),
        ("src/main.hv", "proc main() i32 { return 0; }\n"),
    ]);

    let o = Command::new(HAVEN)
        .current_dir(dir.path())
        .env("HAVENC_INTERNAL_PANIC", "1")
        .env_remove("RUST_BACKTRACE")
        .arg("build")
        .output()
        .expect("failed to spawn haven");
    assert!(!o.status.success(), "a crashed compiler must fail the build");

    let e = err(&o);
    assert!(e.contains("internal compiler error"),
        "the compiler's own report reaches the user:\n{e}");
    assert!(e.contains("the compiler crashed"),
        "and haven names it as a crash, not a build failure:\n{e}");
    assert!(!e.contains("compilation failed"), "got:\n{e}");
}
