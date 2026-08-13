//! `[dependencies]` wiring: `haven build`/`haven run` on a project with a path
//! dependency on a Haven library.

use std::path::Path;

mod common;
use common::{err, haven, out, scaffold};

/// An `app` binary depending on an `example_lib` library by path — the layout the
/// `[dependencies]` table is for. The library imports `std/math`, so this also
/// covers std staying shared across the dependency boundary.
fn workspace(root: &Path) {
    scaffold(root, &[
        ("example_lib/haven.toml",
            "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n"),
        ("example_lib/src/lib.hv",
            "import std/math { square }\n\
             pub proc sumsq(a: i32, b: i32) i32 {\n\
             \x20   return numerical_cast::<i32>(square(numerical_cast::<f64>(a)) \
             + square(numerical_cast::<f64>(b)));\n\
             }\n"),
        ("app/haven.toml",
            "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\n\
             [dependencies]\nexample_lib = { path = \"../example_lib\" }\n"),
        ("app/src/main.hv",
            "import example_lib { sumsq }\n\
             proc main() i32 {\n\
             \x20   println(sumsq(5, 3));\n\
             \x20   return 0;\n\
             }\n"),
    ]);
}

/// Replace `app`'s manifest, keeping the rest of the layout.
fn set_app_manifest(root: &Path, body: &str) {
    std::fs::write(root.join("app").join("haven.toml"), body).unwrap();
}

#[test]
fn builds_dependency_then_program() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    let app = dir.path().join("app");

    let res = haven(&app, &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));

    // the library is built first, into its own `.haven/target/`, as a `.hvmeta`.
    let stdout = out(&res);
    let lib_at = stdout.find("Compiling example_lib").expect("library not built");
    let app_at = stdout.find("Compiling app").expect("program not built");
    assert!(lib_at < app_at, "the dependency must build first:\n{stdout}");
    assert!(dir.path().join("example_lib/.haven/target/example-lib.hvmeta").is_file(),
        "dependency should emit a .hvmeta into its own target dir");
}

#[test]
fn runs_with_dependency() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    let res = haven(&dir.path().join("app"), &["run"]);
    assert!(res.status.success(), "run failed: {}", err(&res));
    // 5*5 + 3*3 == 34; also proves std linked once across the boundary.
    assert!(out(&res).contains("34"), "unexpected program output:\n{}", out(&res));
}

/// v1 resolves direct dependencies only. A dependency that declares its own
/// dependencies is rejected with a message that says so, rather than silently
/// building a library whose imports cannot resolve at the leaf.
#[test]
fn rejects_transitive_dependency() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    scaffold(dir.path(), &[
        ("deep/haven.toml",
            "[project]\nname = \"deep\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n"),
        ("deep/src/lib.hv", "pub proc d() i32 { return 1; }\n"),
    ]);
    // make the library itself depend on `deep`.
    std::fs::write(dir.path().join("example_lib/haven.toml"),
        "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n\n\
         [dependencies]\ndeep = { path = \"../deep\" }\n").unwrap();

    let res = haven(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a transitive dependency must be rejected");
    let e = err(&res);
    assert!(e.contains("transitive"), "should name the limitation; got: {e}");
    assert!(e.contains("deep"), "should name the offending dependency; got: {e}");
}

/// The key must be the library's own package name: it is what anchors the
/// library's emitted symbols, so a mismatch would compile its items under a
/// namespace that disagrees with the library's own build.
#[test]
fn rejects_name_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    set_app_manifest(dir.path(),
        "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\n\
         [dependencies]\nwrong_name = { path = \"../example_lib\" }\n");

    let res = haven(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a mismatched dependency key must be rejected");
    let e = err(&res);
    assert!(e.contains("wrong_name") && e.contains("example_lib"),
        "should name both the key and the real package; got: {e}");
}

/// Only `kind = ["lib"]` produces a `.hvmeta`. A `cdylib`/`staticlib` builds a
/// native artifact that cannot be depended on this way.
#[test]
fn rejects_non_lib_dependency() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    std::fs::write(dir.path().join("example_lib/haven.toml"),
        "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"cdylib\"]\n").unwrap();

    let res = haven(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a non-lib dependency must be rejected");
    assert!(err(&res).contains("not a Haven library"), "got: {}", err(&res));
}

#[test]
fn reports_missing_dependency_path() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    set_app_manifest(dir.path(),
        "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\n\
         [dependencies]\nexample_lib = { path = \"../nowhere\" }\n");

    let res = haven(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a missing dependency path must be reported");
    let e = err(&res);
    assert!(e.contains("haven.toml") && e.contains("example_lib"), "got: {e}");
}

/// A bare string (`example_lib = "../example_lib"`) is not the supported form;
/// the error should say what is.
#[test]
fn rejects_unsupported_dependency_form() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    set_app_manifest(dir.path(),
        "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\n\
         [dependencies]\nexample_lib = \"../example_lib\"\n");

    let res = haven(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a bare-string dependency must be rejected");
    assert!(err(&res).contains("path"), "should point at the path form; got: {}", err(&res));
}

/// A `[[c]]` table on a *binary* project: `haven` forwards its `files` as
/// `--c-file` and `libs` as `--link-lib`, so a program can ship C glue and link a
/// system library. Proves the manifest wiring end to end - the C symbol resolves
/// and the program runs. (`libs = ["m"]` stands in for a real system lib like
/// raylib; libm is guaranteed on the test host.)
#[test]
fn binary_with_c_table_builds_and_runs() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("app/haven.toml",
            "[project]\nname = \"app\"\nversion = \"0.1.0\"\nkind = [\"bin\"]\n\n\
             [[c]]\nfiles = [\"c/glue.c\"]\nlibs = [\"m\"]\n"),
        ("app/c/glue.c",
            "#include <math.h>\ndouble glue_hypot(double a, double b) { return hypot(a, b); }\n"),
        ("app/src/main.hv",
            "extern glue_hypot(a: f64, b: f64) f64;\n\
             proc main() i32 {\n\
             \x20   println(numerical_cast::<i32>(glue_hypot(3.0f64, 4.0f64)));\n\
             \x20   return 0;\n\
             }\n"),
    ]);

    let res = haven(&dir.path().join("app"), &["run"]);
    assert!(res.status.success(), "a bin with a [[c]] table must build and run: {}", err(&res));
    // hypot(3, 4) == 5: the C from `files` compiled in, and `-lm` from `libs` linked.
    assert!(out(&res).contains('5'), "unexpected program output:\n{}", out(&res));
}
