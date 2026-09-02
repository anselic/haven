//! End-to-end test harness for the compiler.
//!
//! Each fixture is a single `.hv` file; the whole compiler pipeline (parse ->
//! typecheck -> safecheck -> mono -> mil -> llvm -> clang link) is exercised by
//! actually compiling and running it. Tests are discovered from disk and run in
//! parallel via `libtest-mimic`, so each one shows up as its own named case
//! under `cargo test`.
//!
//! Two kinds of fixtures, by directory:
//!
//! * `tests/cases/run/`  - must compile, link, and run. If a sibling `<name>.out`
//!   file exists, the program's stdout must match it exactly. The process must
//!   exit 0 unless the file opts out with `//@ exit: any` (used for `proc main`
//!   with no return type, whose exit code is whatever's left in the register).
//!   `//@ exit: N` asserts a specific code.
//!
//! * `tests/cases/fail/` - must FAIL to compile (non-zero exit). Every
//!   `//@ error: <substr>` line in the fixture must appear somewhere in the
//!   compiler's stderr. Match short, stable phrases, not whole diagnostics.
//!
//! Directives are `//@ key: value` lines anywhere in the file. To re-bless the
//! `.out` goldens after an intentional output change, run
//! `tests/cases/bless.sh`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use libtest_mimic::{Arguments, Failed, Trial};

/// Path to the freshly-built compiler binary. Cargo sets this for integration
/// tests so we always test the current build, no matter the target dir.
const COMPILER_BIN: &str = env!("CARGO_BIN_EXE_havenc");

/// The compiler embeds no std: it discovers `std.hvmeta` on disk. The fixtures
/// invoke `havenc foo.hv -o out` with no std flags, so the harness builds the
/// standalone `std` package into an artifact once and points every fixture at it via
/// `$HAVEN_STD` (set per-invocation, so it never leaks into the other test
/// binaries that bring their own std/mini). The `TempDir` is parked in the
/// `OnceLock` so it outlives the whole run rather than being cleaned up when the
/// builder returns.
static STD_META: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();

/// Build the standalone `std` package into a `std.hvmeta` the fixtures can discover,
/// once per process. Uses the same `havenc` under test; the std build itself
/// needs no discovered std (it is `--package-name std --prelude std`, so the
/// compiler's self-dependency guard skips discovery).
fn std_meta() -> &'static Path {
    &STD_META.get_or_init(|| {
        // `CARGO_MANIFEST_DIR` is `bins/havenc`; the repo root is two up.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().and_then(|p| p.parent())
            .expect("repo root above bins/havenc");
        let std_dir = repo.join("stdlib/std");
        let tmp = tempfile::tempdir().expect("temp dir for std.hvmeta");
        let meta = tmp.path().join("std.hvmeta");
        let status = Command::new(COMPILER_BIN)
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
        assert!(status.status.success(),
            "building std.hvmeta for the harness failed:\n{}",
            String::from_utf8_lossy(&status.stderr));
        (tmp, meta)
    }).1
}

/// Built dependency artifacts, keyed by package dir relative to the repo root. A
/// `//@ dep: name=stdlib/foo` fixture binds `foo.hvmeta`; several fixtures share
/// one, so each is built once. The `TempDir`s are parked so the artifacts outlive
/// the run.
static DEP_METAS: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();
static DEP_TMPS: OnceLock<Mutex<Vec<tempfile::TempDir>>> = OnceLock::new();

/// Build the stdlib package at `<repo>/<pkgdir>` (e.g. `stdlib/plug`) into a
/// `<name>.hvmeta` a fixture can bind with `--dep`, once per `pkgdir`. These
/// packages' own modules do `import std/...`, so the build runs with `$HAVEN_STD`
/// pointed at the harness's std artifact - the same discovery a real consumer
/// build gets from the installed std.
fn dep_meta(pkgdir: &str, name: &str, prior: &[(String, PathBuf)]) -> PathBuf {
    let cache = DEP_METAS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = cache.lock().unwrap().get(pkgdir) {
        return p.clone();
    }
    // `CARGO_MANIFEST_DIR` is `bins/havenc`; the repo root is two up.
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(|p| p.parent())
        .expect("repo root above bins/havenc");
    let dir = repo.join(pkgdir);
    let tmp = tempfile::tempdir().expect("temp dir for dep hvmeta");
    let meta = tmp.path().join(format!("{name}.hvmeta"));
    let mut cmd = Command::new(COMPILER_BIN);
    cmd.arg(dir.join("src/lib.hv"))
        .args(["--lib", "--package-name", name])
        .arg("-o").arg(&meta)
        .env("HAVEN_STD", std_meta());
    // A package built here may itself depend on an earlier `//@ dep:` entry (e.g.
    // `plug` on `dsp`), so bind everything declared before it. The cache key is
    // `pkgdir`, which assumes a package is always built with the same prior set -
    // true for these fixtures.
    for (n, p) in prior {
        cmd.arg("--dep").arg(format!("{n}={}", p.display()));
    }
    // A package's `[[c]]` native sources ride into its `.hvmeta` for a consumer's
    // leaf to link (dsp ships `c/denormal.c`, backing `rt_denormals_*`). The
    // harness doesn't parse `haven.toml`, so it globs `<pkg>/c/*.c` - enough for
    // the stdlib packages, whose C all lives there.
    if let Ok(entries) = std::fs::read_dir(dir.join("c")) {
        let mut cfiles: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "c"))
            .collect();
        cfiles.sort();
        for c in cfiles {
            cmd.arg("--c-file").arg(c);
        }
    }
    let out = cmd.output().expect("failed to spawn havenc to build a dependency hvmeta");
    assert!(out.status.success(),
        "building {name}.hvmeta ({pkgdir}) for the harness failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
    DEP_TMPS.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap().push(tmp);
    cache.lock().unwrap().insert(pkgdir.to_string(), meta.clone());
    meta
}

fn main() {
    let args = Arguments::from_args();

    let cases = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases");
    let mut trials = Vec::new();
    collect(&cases.join("run"), Mode::Run, &mut trials);
    collect(&cases.join("fail"), Mode::Fail, &mut trials);

    libtest_mimic::run(&args, trials).exit();
}

#[derive(Clone, Copy)]
enum Mode {
    Run,
    Fail,
}

/// Turn every `.hv` file in `dir` into a `Trial`. The test name is
/// `run/<stem>` or `fail/<stem>` so failures point straight at the fixture.
fn collect(dir: &Path, mode: Mode, trials: &mut Vec<Trial>) {
    let kind = match mode {
        Mode::Run => "run",
        Mode::Fail => "fail",
    };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read fixtures in {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "hv"))
        .collect();
    entries.sort();

    for path in entries {
        let name = format!("{kind}/{}", path.file_stem().unwrap().to_string_lossy());
        trials.push(Trial::test(name, move || run_case(&path, mode)));
    }
}

fn run_case(path: &Path, mode: Mode) -> Result<(), Failed> {
    let src = std::fs::read_to_string(path)?;
    let directives = Directives::parse(&src);

    // Each case gets its own temp dir: the compiler writes `<out>.ll` next to the
    // output binary, and tests run in parallel, so a shared path would collide.
    let tmp = tempfile::tempdir()?;
    let out = tmp.path().join("out");

    let mut cmd = Command::new(COMPILER_BIN);
    cmd.arg(path)
        .arg("-o")
        .arg(&out)
        // point the flagless fixture at the on-disk std the harness built.
        .env("HAVEN_STD", std_meta());
    // `//@ dep: name=stdlib/foo` fixtures bind a standalone stdlib package (dsp,
    // plug) the same way a consumer does. Build each in declaration order,
    // threading the already-built ones in so a package that depends on an earlier
    // one (plug -> dsp) resolves; then bind the whole set on the fixture's compile
    // - the transitive closure a leaf needs, since a `.hvmeta` lists no deps.
    let mut built: Vec<(String, PathBuf)> = Vec::new();
    for (name, pkgdir) in &directives.deps {
        let meta = dep_meta(pkgdir, name, &built);
        built.push((name.clone(), meta));
    }
    for (name, meta) in &built {
        cmd.arg("--dep").arg(format!("{name}={}", meta.display()));
    }
    let compile = cmd
        .output()
        .map_err(|e| format!("failed to spawn {COMPILER_BIN}: {e}"))?;
    let stderr = strip_ansi(&String::from_utf8_lossy(&compile.stderr));

    match mode {
        Mode::Fail => {
            if compile.status.success() {
                return Err(format!("expected compilation to fail, but it succeeded\n{stderr}").into());
            }
            // a crash is not a failure to compile. the compiler must *report*
            // the error - exit 1 - never panic (exit 101, kept by the ICE hook)
            // or die some other way. without this, a fixture whose `//@ error:`
            // substring happened to appear in a panic message passed, and an
            // internal error hid behind a test that was green.
            if compile.status.code() != Some(1) {
                return Err(format!(
                    "expected a diagnostic exit (1) but the compiler crashed ({})\n{stderr}",
                    compile.status).into());
            }
            for want in &directives.errors {
                if !stderr.contains(want) {
                    return Err(format!(
                        "compiler stderr did not contain expected error.\n  want substring: {want:?}\n  --- stderr ---\n{stderr}"
                    )
                    .into());
                }
            }
            if directives.errors.is_empty() {
                return Err("fail-test has no `//@ error:` directive to check against".into());
            }
            Ok(())
        }
        Mode::Run => {
            if !compile.status.success() {
                return Err(format!("compilation failed\n{stderr}").into());
            }

            let run = Command::new(&out)
                .output()
                .map_err(|e| format!("failed to run compiled binary: {e}"))?;

            // exit code (default is 0), overridable via `//@ exit: N` or skipped
            // with `//@ exit: any`
            match directives.exit {
                ExitCheck::Any => {}
                ExitCheck::Code(want) => {
                    let got = run.status.code();
                    if got != Some(want) {
                        return Err(format!(
                            "wrong exit code: want {want}, got {got:?}\n  --- stdout ---\n{}",
                            String::from_utf8_lossy(&run.stdout)
                        )
                        .into());
                    }
                }
            }

            // stdout must match the `.out` golden byte-for-byte, if present.
            let golden_path = path.with_extension("out");
            if let Ok(want) = std::fs::read(&golden_path) {
                if run.stdout != want {
                    return Err(format!(
                        "stdout did not match {}\n  --- expected ---\n{}\n  --- actual ---\n{}",
                        golden_path.display(),
                        String::from_utf8_lossy(&want),
                        String::from_utf8_lossy(&run.stdout),
                    )
                    .into());
                }
            }
            Ok(())
        }
    }
}

/// `//@ key: value` directives extracted from a fixture.
struct Directives {
    exit: ExitCheck,
    errors: Vec<String>,
    /// `(package name, package dir relative to repo root)` from `//@ dep:` lines.
    deps: Vec<(String, String)>,
}

enum ExitCheck {
    Code(i32),
    Any,
}

impl Directives {
    fn parse(src: &str) -> Self {
        let mut exit = ExitCheck::Code(0);
        let mut errors = Vec::new();
        let mut deps = Vec::new();
        for line in src.lines() {
            let Some(rest) = line.trim_start().strip_prefix("//@") else {
                continue;
            };
            let rest = rest.trim();
            if let Some(v) = rest.strip_prefix("exit:") {
                let v = v.trim();
                exit = if v == "any" {
                    ExitCheck::Any
                } else {
                    ExitCheck::Code(v.parse().unwrap_or_else(|_| panic!("bad `//@ exit:` value: {v:?}")))
                };
            } else if let Some(v) = rest.strip_prefix("error:") {
                errors.push(v.trim().to_string());
            } else if let Some(v) = rest.strip_prefix("dep:") {
                let (name, dir) = v.trim().split_once('=').unwrap_or_else(||
                    panic!("`//@ dep:` wants `name=pkgdir`, got: {v:?}"));
                deps.push((name.trim().to_string(), dir.trim().to_string()));
            } else {
                panic!("unknown directive: {line:?}");
            }
        }
        Directives { exit, errors, deps }
    }
}

/// Strip ANSI SGR escapes so `//@ error:` substrings match against the plain
/// text of ariadne's colored diagnostics.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // skip until the terminating letter of the escape sequence
            for e in chars.by_ref() {
                if e.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
