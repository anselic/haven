//! Shared harness for the `vestry` integration suites.
//!
//! These drive the real `vestry` binary, which locates its sibling `havenc` in the
//! same target directory - so a passing test exercises manifest parsing and the
//! handoff to the compiler end to end.
//!
//! Included by each test binary with `mod common;`, so every suite gets its own
//! copy; the `dead_code` allowance is because no single suite uses all of it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

pub const VESTRY: &str = env!("CARGO_BIN_EXE_vestry");

/// Write `(relative path, contents)` files under `root`, creating parent dirs.
pub fn scaffold(root: &Path, files: &[(&str, &str)]) {
    for (rel, contents) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
    }
}

/// The `havenc` beside the `vestry` under test.
pub fn havenc() -> PathBuf {
    Path::new(VESTRY).with_file_name(if cfg!(windows) { "havenc.exe" } else { "havenc" })
}

static STD_META: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();

/// Build the standalone `std` package into a discoverable `std.hvmeta`, once. The
/// compiler embeds no std; the `vestry` tool locates its sibling `havenc`, and
/// `$HAVEN_STD` set on the `vestry` process (below) propagates to that `havenc`.
/// The build uses the same sibling `havenc`, found next to the `vestry` binary.
pub fn std_meta() -> &'static Path {
    &STD_META.get_or_init(|| {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().and_then(|p| p.parent()).expect("repo root above bins/vestry");
        let std_dir = repo.join("stdlib/std");
        let tmp = tempfile::tempdir().expect("temp dir for std.hvmeta");
        let meta = tmp.path().join("std.hvmeta");
        let o = Command::new(havenc())
            .arg(std_dir.join("src/lib.hv"))
            .args(["--lib", "--package-name", "std", "--prelude", "std"])
            .arg("--c-file").arg(std_dir.join("c/rt.c"))
            .arg("--c-file").arg(std_dir.join("c/env.c"))
            .arg("--c-file").arg(std_dir.join("c/fs.c"))
            .arg("--c-file").arg(std_dir.join("c/process.c"))
            .args(["--link-lib", "m"])
            .arg("-o").arg(&meta)
            .output().expect("failed to spawn havenc to build std.hvmeta");
        assert!(o.status.success(), "building std.hvmeta failed:\n{}",
            String::from_utf8_lossy(&o.stderr));
        (tmp, meta)
    }).1
}

pub fn vestry(cwd: &Path, args: &[&str]) -> std::process::Output {
    vestry_env(cwd, args, &[])
}

/// Like [`vestry`], but with extra environment variables set on the process -
/// used by the git-dependency suite to point `VESTRY_HOME` at a disposable cache.
pub fn vestry_env(cwd: &Path, args: &[&str], envs: &[(&str, &Path)]) -> std::process::Output {
    let mut cmd = Command::new(VESTRY);
    cmd.current_dir(cwd).env("HAVEN_STD", std_meta());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.args(args).output().expect("failed to spawn vestry")
}

pub fn out(o: &std::process::Output) -> String {
    String::from_utf8_lossy(&o.stdout).replace("\r\n", "\n")
}

pub fn err(o: &std::process::Output) -> String {
    String::from_utf8_lossy(&o.stderr).replace("\r\n", "\n")
}
