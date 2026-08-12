//! Shared test helpers. The compiler embeds no std: it discovers `std.hvmeta` on
//! disk (via `$HAVEN_STD` or a sysroot-relative path). Integration tests spawn
//! the compiler directly, so they build the standalone `std` package into a temp
//! artifact once per binary and point `$HAVEN_STD` at it.

#![allow(dead_code)] // each test binary uses a different subset of these

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Path to the freshly-built compiler. Cargo sets this per integration test.
pub const HAVENC: &str = env!("CARGO_BIN_EXE_havenc");

static STD_META: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();

/// Build the standalone `std` package into a discoverable `std.hvmeta`, once per test
/// binary. The `TempDir` is parked in the `OnceLock` so the artifact outlives the
/// whole run. Building std needs no discovered std itself (it is `--package-name
/// std --prelude std`, so the compiler's self-dependency guard skips discovery).
pub fn std_meta() -> &'static Path {
    &STD_META.get_or_init(|| {
        // `CARGO_MANIFEST_DIR` is `bins/havenc`; the repo root is two up.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().and_then(|p| p.parent())
            .expect("repo root above bins/havenc");
        let std_dir = repo.join("std");
        let tmp = tempfile::tempdir().expect("temp dir for std.hvmeta");
        let meta = tmp.path().join("std.hvmeta");
        let out = Command::new(HAVENC)
            .arg(std_dir.join("src/lib.hv"))
            .args(["--lib", "--package-name", "std", "--prelude", "std"])
            .arg("--c-file").arg(std_dir.join("c/rt.c"))
            .arg("--c-file").arg(std_dir.join("c/env.c"))
            .arg("--c-file").arg(std_dir.join("c/fs.c"))
            .arg("--c-file").arg(std_dir.join("c/process.c"))
            .args(["--link-lib", "m"])
            .arg("-o").arg(&meta)
            .output()
            .expect("failed to spawn havenc to build std.hvmeta");
        assert!(out.status.success(),
            "building std.hvmeta for the tests failed:\n{}",
            String::from_utf8_lossy(&out.stderr));
        (tmp, meta)
    }).1
}

/// A `havenc` command with `$HAVEN_STD` pointed at the tests' built std, so a
/// flagless build discovers it. Tests that bind their own `--dep std=` are
/// unaffected: the compiler skips discovery when `std` is already bound, and the
/// discovered std is a fallback-priority prelude provider, so a test's own
/// prelude-bearing dep still wins.
pub fn havenc_cmd() -> Command {
    let mut cmd = Command::new(HAVENC);
    cmd.env("HAVEN_STD", std_meta());
    cmd
}
