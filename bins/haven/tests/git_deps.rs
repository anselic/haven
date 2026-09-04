//! Git `[dependencies]`: `haven build`/`run` on a project that depends on a Haven
//! library living in a git repository.
//!
//! No network: the "remote" is a git repo created in a tempdir and referenced by
//! its local path, which `git clone` handles like any other transport. Each test
//! points `HAVEN_HOME` at its own disposable directory so the global cache under
//! the developer's real `~/.haven` is never touched.

use std::path::Path;
use std::process::Command;

mod common;
use common::{err, haven_env, out, scaffold};

/// Run `git` in `dir`, asserting success. `args` are passed after an implicit
/// `-C <dir>`; commit-making commands carry an identity so the suite works on a
/// machine with no global git config (CI).
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C").arg(dir)
        .args(["-c", "user.email=test@haven", "-c", "user.name=haven test"])
        .args(args)
        .output()
        .expect("failed to run git (is it installed?)");
    assert!(out.status.success(),
        "git {:?} failed: {}", args, String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Create a git repository holding an `example_lib` Haven library, commit it, and
/// tag the commit `v1.0.0`. Returns `(repo path as a forward-slashed URL, commit
/// SHA)`. The library imports `std/math`, mirroring the path-dependency suite.
fn make_lib_repo(dir: &Path) -> (String, String) {
    scaffold(dir, &[
        ("haven.toml",
            "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n"),
        ("src/lib.hv",
            "import std/math { square }\n\
             pub proc sumsq(a: i32, b: i32) i32 {\n\
             \x20   return numerical_cast::<i32>(square(numerical_cast::<f64>(a)) \
             + square(numerical_cast::<f64>(b)));\n\
             }\n"),
    ]);
    git(dir, &["init", "--quiet"]);
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "--quiet", "-m", "initial"]);
    git(dir, &["tag", "v1.0.0"]);
    let sha = git(dir, &["rev-parse", "HEAD"]);
    // git clone accepts a local path; forward slashes keep it valid inside a TOML
    // basic string on Windows (`C:\...` would be an invalid escape).
    let url = dir.to_string_lossy().replace('\\', "/");
    (url, sha)
}

/// Scaffold an `app` binary whose manifest depends on `example_lib` via `dep_line`
/// (the `[dependencies]` entry, sans the leading key).
fn make_app(dir: &Path, dep_line: &str) {
    scaffold(dir, &[
        ("haven.toml", &format!(
            "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\n\
             [dependencies]\nexample_lib = {dep_line}\n")),
        ("src/main.hv",
            "import example_lib { sumsq }\n\
             proc main() i32 {\n\
             \x20   println(sumsq(5, 3));\n\
             \x20   return 0;\n\
             }\n"),
    ]);
}

#[test]
fn builds_and_runs_a_tagged_git_dependency() {
    let lib = tempfile::tempdir().unwrap();
    let (url, _sha) = make_lib_repo(lib.path());

    let work = tempfile::tempdir().unwrap();
    let app = work.path().join("app");
    make_app(&app, &format!("{{ git = \"{url}\", tag = \"v1.0.0\" }}"));
    let home = work.path().join("home");

    let res = haven_env(&app, &["run"], &[("HAVEN_HOME", home.as_path())]);
    assert!(res.status.success(), "run failed:\n{}", err(&res));
    assert!(out(&res).contains("Fetching"), "expected a Fetching line:\n{}", out(&res));
    assert!(out(&res).contains("34"), "program output missing:\n{}", out(&res));

    // the cache was populated under HAVEN_HOME, not the real home dir
    assert!(home.join("git").is_dir(), "git cache not created under HAVEN_HOME");
}

#[test]
fn pins_to_a_rev() {
    let lib = tempfile::tempdir().unwrap();
    let (url, sha) = make_lib_repo(lib.path());

    let work = tempfile::tempdir().unwrap();
    let app = work.path().join("app");
    make_app(&app, &format!("{{ git = \"{url}\", rev = \"{sha}\" }}"));
    let home = work.path().join("home");

    let res = haven_env(&app, &["run"], &[("HAVEN_HOME", home.as_path())]);
    assert!(res.status.success(), "run failed:\n{}", err(&res));
    assert!(out(&res).contains("34"), "program output missing:\n{}", out(&res));
    // the checkout directory is keyed by the resolved commit
    assert!(home.join("git").read_dir().unwrap()
        .filter_map(Result::ok)
        .any(|url_dir| url_dir.path().join(&sha).is_dir()),
        "no checkout keyed by the pinned commit {sha}");
}

#[test]
fn warm_cache_skips_the_network_for_a_pinned_ref() {
    let lib = tempfile::tempdir().unwrap();
    let (url, _sha) = make_lib_repo(lib.path());

    let work = tempfile::tempdir().unwrap();
    let app = work.path().join("app");
    make_app(&app, &format!("{{ git = \"{url}\", tag = \"v1.0.0\" }}"));
    let home = work.path().join("home");
    let env = [("HAVEN_HOME", home.as_path())];

    let first = haven_env(&app, &["build"], &env);
    assert!(first.status.success(), "first build failed:\n{}", err(&first));
    assert!(out(&first).contains("Fetching"), "first build should fetch:\n{}", out(&first));

    // clear the build output so the second run genuinely rebuilds and re-resolves
    // the dependency, then confirm it does not touch the network again.
    std::fs::remove_dir_all(app.join(".haven")).unwrap();
    let second = haven_env(&app, &["build"], &env);
    assert!(second.status.success(), "second build failed:\n{}", err(&second));
    assert!(!out(&second).contains("Fetching"),
        "a pinned ref on a warm cache should not re-fetch:\n{}", out(&second));
}

#[test]
fn a_branch_dependency_warns_that_it_is_not_reproducible() {
    let lib = tempfile::tempdir().unwrap();
    let (url, _sha) = make_lib_repo(lib.path());
    // put the library on a named branch so the manifest can track it
    git(lib.path(), &["branch", "dev"]);

    let work = tempfile::tempdir().unwrap();
    let app = work.path().join("app");
    make_app(&app, &format!("{{ git = \"{url}\", branch = \"dev\" }}"));
    let home = work.path().join("home");

    let res = haven_env(&app, &["build"], &[("HAVEN_HOME", home.as_path())]);
    assert!(res.status.success(), "build failed:\n{}", err(&res));
    assert!(err(&res).contains("not reproducible"),
        "expected a non-reproducibility warning:\n{}", err(&res));
}

#[test]
fn a_missing_ref_is_a_clean_error() {
    let lib = tempfile::tempdir().unwrap();
    let (url, _sha) = make_lib_repo(lib.path());

    let work = tempfile::tempdir().unwrap();
    let app = work.path().join("app");
    make_app(&app, &format!("{{ git = \"{url}\", tag = \"v9.9.9\" }}"));
    let home = work.path().join("home");

    let res = haven_env(&app, &["build"], &[("HAVEN_HOME", home.as_path())]);
    assert!(!res.status.success(), "build should have failed for a missing tag");
    let msg = err(&res);
    assert!(msg.contains("v9.9.9") && msg.contains("could not find"),
        "error should name the missing tag:\n{msg}");
}

#[test]
fn rejects_more_than_one_ref() {
    let lib = tempfile::tempdir().unwrap();
    let (url, sha) = make_lib_repo(lib.path());

    let work = tempfile::tempdir().unwrap();
    let app = work.path().join("app");
    make_app(&app, &format!("{{ git = \"{url}\", tag = \"v1.0.0\", rev = \"{sha}\" }}"));
    let home = work.path().join("home");

    let res = haven_env(&app, &["build"], &[("HAVEN_HOME", home.as_path())]);
    assert!(!res.status.success(), "build should reject two pinned refs");
    assert!(err(&res).contains("at most one"), "error should explain the rule:\n{}", err(&res));
}
