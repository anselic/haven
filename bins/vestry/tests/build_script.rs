//! Pre-build `build` scripts and artifact-time `post-build` scripts.
//!
//! The point of the hook is that `vestry` stays ignorant of what the script does -
//! bundling a `.clap`, stamping a version, signing - so what these tests pin down
//! is the *contract*, not any particular use of it: when the script runs, what it
//! is told, and what happens when it fails.

use std::path::Path;

mod common;
use common::{err, vestry, out, scaffold};

/// A script that reports its whole environment and proves the artifact is really
/// on disk by the time it runs. Also drops a file in the working directory, which
/// is how the tests check `vestry` runs it from the project root.
const POST_PROBE: &str = "\
import std/env
import std/fs
import std/result { Result }
import std/string { String }

// `NAME=<value>`, aborting when absent: these are contract, not configuration.
proc show(name: str) {
    let v: String = env::var(name).unwrap();
    print(name); print(\"=\"); println(v.as_str());
}

proc main() i32 {
    show(\"VESTRY_PKG_NAME\");
    show(\"VESTRY_PKG_VERSION\");
    show(\"VESTRY_OUTPUT_KIND\");
    show(\"VESTRY_TARGET_OS\");

    let artifact: String = env::var(\"VESTRY_ARTIFACT\").unwrap();
    if (!fs::exists(artifact.as_str())) {
        println(\"ARTIFACT-MISSING\");
        return 1;
    }
    println(\"ARTIFACT-EXISTS\");

    // a relative path, so where it lands says what the working directory was.
    let stamp: String = String::from_str(\"ran\\n\");
    match (fs::write(\"evidence.txt\", &stamp)) {
        Result::Ok(_)  -> { return 0; }
        Result::Err(e) -> { println(e.message()); return 1; }
    }
}
";

/// A binary project whose manifest declares `post-build = \"package.hv\"`.
fn app_with_post_script(root: &Path, script: &str) {
    scaffold(root, &[
        ("vestry.toml",
            "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\
             post-build = \"package.hv\"\n"),
        ("src/main.hv", "proc main() i32 {\n    println(1);\n    return 0;\n}\n"),
        ("package.hv", script),
    ]);
}

const PRE_PROBE: &str = "\
import std/env
import std/string { String }

proc main() i32 {
    let out: String = env::var(\"VESTRY_OUT_DIR\").unwrap();
    let arch: String = env::var(\"VESTRY_TARGET_ARCH\").unwrap();
    print(\"OUT=\"); println(out.as_str());
    print(\"ARCH=\"); println(arch.as_str());
    println(\"pre-ran\");
    println(\"vestry::warning=hello from build.hv\");
    return 0;
}
";

#[test]
fn build_script_runs_before_compilation() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("vestry.toml",
            "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\
             build = \"build.hv\"\n"),
        ("src/main.hv", "proc main() i32 { return 0; }\n"),
        ("build.hv", PRE_PROBE),
    ]);

    let res = vestry(dir.path(), &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));
    let stdout = out(&res);
    let script_at = stdout.find("pre-ran").expect("pre-build output missing");
    let compile_at = stdout.find("Compiling app v0.1.0 (executable)")
        .expect("package compile status missing");
    assert!(script_at < compile_at, "build.hv must run before compilation:\n{stdout}");
    assert!(stdout.contains(&format!("ARCH={}", std::env::consts::ARCH)), "got:\n{stdout}");
    assert!(stdout.contains("Warning") && stdout.contains("hello from build.hv"), "got:\n{stdout}");
    assert!(dir.path().join(".vestry/out").is_dir());
}

#[test]
fn build_directives_are_stored_in_library_metadata() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("vestry.toml",
            "[project]\nname = \"native_provider\"\nkind = [\"lib\"]\n\
             build = \"build.hv\"\n"),
        ("src/lib.hv", "pub proc answer() i32 { return 42; }\n"),
        ("build.hv",
            "proc main() i32 {\n\
             \x20   println(\"vestry::link-search=native=/native/lib\");\n\
             \x20   println(\"vestry::link-lib=SDL3\");\n\
             \x20   println(\"vestry::link-archive=native/helper.a\");\n\
             \x20   println(\"vestry::link-arg=-pthread\");\n\
             \x20   return 0;\n}\n"),
    ]);

    let res = vestry(dir.path(), &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));
    let meta = haven_meta::read(
        &dir.path().join(".vestry/target/native-provider.hvmeta")).unwrap();
    assert_eq!(meta.link_search, vec!["/native/lib"]);
    assert_eq!(meta.link_libs, vec!["SDL3"]);
    assert_eq!(meta.link_args, vec![
        dir.path().join("native/helper.a").to_string_lossy().into_owned(),
        "-pthread".to_string(),
    ]);
}

#[test]
fn rejects_unknown_build_directive() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("vestry.toml", "[project]\nname = \"app\"\nbuild = \"build.hv\"\n"),
        ("src/main.hv", "proc main() i32 { return 0; }\n"),
        ("build.hv", "proc main() i32 { println(\"vestry::mystery=value\"); return 0; }\n"),
    ]);
    let res = vestry(dir.path(), &["build"]);
    assert!(!res.status.success());
    assert!(err(&res).contains("unknown directive"), "got: {}", err(&res));
}

#[test]
fn runs_the_script_once_the_artifact_exists() {
    let dir = tempfile::tempdir().unwrap();
    app_with_post_script(dir.path(), POST_PROBE);

    let res = vestry(dir.path(), &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));
    let stdout = out(&res);

    // the ordering the hook exists for: a packaging step cannot stage an artifact
    // that has not been linked yet.
    assert!(stdout.contains("ARTIFACT-EXISTS"),
        "VESTRY_ARTIFACT should name a file that exists by the time the script \
         runs:\n{stdout}");

    // run from the project root, so a script can use relative paths.
    assert!(dir.path().join("evidence.txt").is_file(),
        "the script should run with the project root as its working directory");

    // the script's own executable is kept out of `target/`, where it would sit
    // beside - and be mistaken for - the project's artifact.
    assert!(dir.path().join(".vestry/build").is_dir(),
        "the compiled script belongs in .vestry/build/");
}

#[test]
fn the_environment_describes_the_build() {
    let dir = tempfile::tempdir().unwrap();
    app_with_post_script(dir.path(), POST_PROBE);

    let res = vestry(dir.path(), &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));
    let stdout = out(&res);

    for expected in [
        "VESTRY_PKG_NAME=app",
        "VESTRY_PKG_VERSION=0.1.0",
        "VESTRY_OUTPUT_KIND=bin",
        &format!("VESTRY_TARGET_OS={}", std::env::consts::OS),
    ] {
        assert!(stdout.contains(expected), "missing `{expected}`:\n{stdout}");
    }
}

/// `VESTRY_OUTPUT_KIND` speaks the manifest's vocabulary, not the prose of the
/// progress line: a script branching on the output kind matches against the same
/// word its author wrote under `kind`.
#[test]
fn output_kind_is_the_manifest_spelling() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("vestry.toml",
            "[project]\nname = \"plug\"\nversion = \"0.1.0\"\nkind = [\"cdylib\"]\n\
             post-build = \"package.hv\"\n"),
        ("src/lib.hv", "@export\nproc thing() i32 {\n    return 7;\n}\n"),
        ("package.hv", POST_PROBE),
    ]);

    let res = vestry(dir.path(), &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));
    let stdout = out(&res);
    assert!(stdout.contains("VESTRY_OUTPUT_KIND=cdylib"), "got:\n{stdout}");
    // and the artifact it is handed is the shared library, not some stale guess.
    assert!(stdout.contains("ARTIFACT-EXISTS"), "got:\n{stdout}");
}

/// A packaging step that fails has to fail the build - otherwise `vestry build`
/// reports success while the thing the user actually wanted is missing.
#[test]
fn a_failing_script_fails_the_build() {
    let dir = tempfile::tempdir().unwrap();
    app_with_post_script(dir.path(),
        "proc main() i32 {\n    println(\"packaging went wrong\");\n    return 3;\n}\n");

    let res = vestry(dir.path(), &["build"]);
    assert!(!res.status.success(), "a nonzero script exit must fail the build");
    let e = err(&res);
    assert!(e.contains("package.hv"), "the error should name the script; got: {e}");
    assert!(e.contains('3'), "the error should carry the script's status; got: {e}");
    // whatever it printed on the way out still reaches the user.
    assert!(out(&res).contains("packaging went wrong"), "got:\n{}", out(&res));
}

#[test]
fn a_missing_script_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("vestry.toml",
            "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\
             post-build = \"nowhere.hv\"\n"),
        ("src/main.hv", "proc main() i32 {\n    return 0;\n}\n"),
    ]);

    let res = vestry(dir.path(), &["build"]);
    assert!(!res.status.success(), "a missing build script must be reported");
    let e = err(&res);
    assert!(e.contains("nowhere.hv"), "should name the script; got: {e}");
    assert!(e.contains("vestry.toml"), "should point at where it was declared; got: {e}");
}

/// The script is compiled once and reused, so declaring one does not add a
/// compile to every build - but a change to it must be picked up.
#[test]
fn the_script_is_reused_until_it_changes() {
    let dir = tempfile::tempdir().unwrap();
    app_with_post_script(dir.path(), "proc main() i32 {\n    println(\"first\");\n    return 0;\n}\n");

    let first = vestry(dir.path(), &["build"]);
    assert!(first.status.success(), "build failed: {}", err(&first));
    assert_eq!(out(&first).matches("(build script)").count(), 1,
        "the first build must compile the script:\n{}", out(&first));

    let second = vestry(dir.path(), &["build"]);
    assert!(second.status.success(), "build failed: {}", err(&second));
    assert_eq!(out(&second).matches("(build script)").count(), 0,
        "an unchanged script must not be recompiled:\n{}", out(&second));
    assert!(out(&second).contains("first"), "it should still run:\n{}", out(&second));

    std::fs::write(dir.path().join("package.hv"),
        "proc main() i32 {\n    println(\"second\");\n    return 0;\n}\n").unwrap();

    let third = vestry(dir.path(), &["build"]);
    assert!(third.status.success(), "build failed: {}", err(&third));
    assert_eq!(out(&third).matches("(build script)").count(), 1,
        "an edited script must be recompiled:\n{}", out(&third));
    assert!(out(&third).contains("second"), "the new script should run:\n{}", out(&third));
}
