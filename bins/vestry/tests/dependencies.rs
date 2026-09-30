//! `[dependencies]` wiring: `vestry build`/`vestry run` on a project with a path
//! dependency on a Haven library.

use std::path::Path;
use std::process::Command;

mod common;
use common::{err, vestry, out, scaffold};

/// An `app` binary depending on an `example_lib` library by path — the layout the
/// `[dependencies]` table is for. The library imports `std/math`, so this also
/// covers std staying shared across the dependency boundary.
fn workspace(root: &Path) {
    scaffold(root, &[
        ("example_lib/vestry.toml",
            "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n"),
        ("example_lib/src/lib.hv",
            "import std/math { square }\n\
             pub proc sumsq(a: i32, b: i32) i32 {\n\
             \x20   return numerical_cast::<i32>(square(numerical_cast::<f64>(a)) \
             + square(numerical_cast::<f64>(b)));\n\
             }\n"),
        ("app/vestry.toml",
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
    std::fs::write(root.join("app").join("vestry.toml"), body).unwrap();
}

#[test]
fn builds_dependency_then_program() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    let app = dir.path().join("app");

    let res = vestry(&app, &["build"]);
    assert!(res.status.success(), "build failed: {}", err(&res));

    // the library is built first, into its own `.vestry/target/`, as a `.hvmeta`.
    let stdout = out(&res);
    let lib_at = stdout.find("Compiling example_lib").expect("library not built");
    let app_at = stdout.find("Compiling app").expect("program not built");
    assert!(lib_at < app_at, "the dependency must build first:\n{stdout}");
    assert!(dir.path().join("example_lib/.vestry/target/example-lib.hvmeta").is_file(),
        "dependency should emit a .hvmeta into its own target dir");
}

#[test]
fn runs_with_dependency() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    let res = vestry(&dir.path().join("app"), &["run"]);
    assert!(res.status.success(), "run failed: {}", err(&res));
    // 5*5 + 3*3 == 34; also proves std linked once across the boundary.
    assert!(out(&res).contains("34"), "unexpected program output:\n{}", out(&res));
}

/// A dependency may declare dependencies of its own: `vestry build` walks the
/// whole closure, builds each package once bottom-up, and binds every transitive
/// artifact at the leaf so the intermediate library's imports resolve there too.
#[test]
fn resolves_transitive_dependency() {
    let dir = tempfile::tempdir().unwrap();
    workspace(dir.path());
    scaffold(dir.path(), &[
        ("deep/vestry.toml",
            "[project]\nname = \"deep\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n"),
        ("deep/src/lib.hv", "pub proc bonus() i32 { return 100; }\n"),
    ]);
    // `example_lib` now depends on `deep` and uses its symbol, so `deep` must be
    // bound when `example_lib` compiles *and* when `app` (the leaf) re-resolves
    // it. The leaf also checks the library's recorded direct-dependency list.
    std::fs::write(dir.path().join("example_lib/vestry.toml"),
        "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"lib\"]\n\n\
         [dependencies]\ndeep = { path = \"../deep\" }\n").unwrap();
    std::fs::write(dir.path().join("example_lib/src/lib.hv"),
        "import deep { bonus }\n\
         pub proc sumsq(a: i32, b: i32) i32 { return a * a + b * b + bonus(); }\n").unwrap();

    let res = vestry(&dir.path().join("app"), &["run"]);
    assert!(res.status.success(), "transitive build/run failed: {}", err(&res));
    // 5*5 + 3*3 + 100 == 134: proves `deep` resolved at the leaf `app`.
    assert!(out(&res).contains("134"), "unexpected program output:\n{}", out(&res));
}

/// A dependency's artifacts remain available for recompiling its source, but
/// its names do not enter the importing package's direct dependency scope.
fn scoped_workspace(root: &Path) {
    scaffold(root, &[
        ("facade/vestry.toml", "[project]\nname = \"facade\"\nkind = [\"lib\"]\n\n\
             [dependencies]\ncore = { path = \"../core\" }\n"),
        ("facade/src/lib.hv", "pub import core/option\n\
             pub import core/math { square }\n"),
        ("facade2/vestry.toml", "[project]\nname = \"facade2\"\nkind = [\"lib\"]\n\n\
             [dependencies]\nfacade = { path = \"../facade\" }\n"),
        ("facade2/src/lib.hv", "pub import facade/option\n\
             pub import facade { square }\n"),
        ("app/vestry.toml", "[project]\nname = \"app\"\nkind = [\"bin\"]\n\n\
             [dependencies]\nfacade = { path = \"../facade\" }\n"),
        ("app/src/main.hv", "import facade { square }\n\
             import facade\n\
             import facade/option { Option }\n\
             proc main() i32 {\n\
             \x20   let o: Option<i32> = Option::Some(7);\n\
             \x20   if (o.unwrap() == 7 && square(4.0f64) == 16.0f64) { return 0; }\n\
             \x20   return 1;\n\
             }\n"),
    ]);
    let core = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().join("stdlib/core");
    let core = core.to_string_lossy().replace('\\', "/");
    std::fs::write(root.join("facade/vestry.toml"), format!(
        "[project]\nname = \"facade\"\nkind = [\"lib\"]\n\n[dependencies]\ncore = {{ path = \"{core}\" }}\n"
    )).unwrap();
}

#[test]
fn rejects_distinct_packages_with_the_same_name() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("left/vestry.toml", "[project]\nname = \"shared\"\nkind = [\"lib\"]\n"),
        ("left/src/lib.hv", "pub proc left() i32 { return 1; }\n"),
        ("right/vestry.toml", "[project]\nname = \"shared\"\nkind = [\"lib\"]\n"),
        ("right/src/lib.hv", "pub proc right() i32 { return 2; }\n"),
        ("facade/vestry.toml", "[project]\nname = \"facade\"\nkind = [\"lib\"]\n\n[dependencies]\nshared = { path = \"../right\" }\n"),
        ("facade/src/lib.hv", "import shared { right }\npub proc value() i32 { return right(); }\n"),
        ("app/vestry.toml", "[project]\nname = \"app\"\nkind = [\"bin\"]\n\n[dependencies]\nshared = { path = \"../left\" }\nfacade = { path = \"../facade\" }\n"),
        ("app/src/main.hv", "proc main() i32 { return 0; }\n"),
    ]);
    let res = vestry(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "two distinct 'shared' packages must conflict");
    assert!(err(&res).contains("shared"), "wrong diagnostic: {}", err(&res));
}

#[test]
fn direct_dependency_scope_preserves_public_reexports() {
    let dir = tempfile::tempdir().unwrap();
    scoped_workspace(dir.path());
    let app = dir.path().join("app");

    let res = vestry(&app, &["run"]);
    assert!(res.status.success(), "direct facade, re-exported core APIs failed: {}", err(&res));
    let facade_meta = haven_meta::read(&dir.path().join("facade/.vestry/target/facade.hvmeta")).unwrap();
    assert_eq!(facade_meta.direct_deps, ["core"]);

    std::fs::write(app.join("src/main.hv"),
        "import core/option { Option }\nproc main() i32 { return 0; }\n").unwrap();
    let res = vestry(&app, &["build"]);
    assert!(!res.status.success(), "app must not import transitive core");
    assert!(err(&res).contains("package 'core' is not a direct dependency of 'app'"),
        "wrong diagnostic: {}", err(&res));

    let core = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().join("stdlib/core");
    let core = core.to_string_lossy().replace('\\', "/");
    set_app_manifest(dir.path(), &format!(
        "[project]\nname = \"app\"\nkind = [\"bin\"]\n\n[dependencies]\nfacade = {{ path = \"../facade\" }}\ncore = {{ path = \"{core}\" }}\n"));
    let res = vestry(&app, &["build"]);
    assert!(res.status.success(), "direct core import should work: {}", err(&res));

    std::fs::write(app.join("src/main.hv"),
        "import core/option { Option }\nproc main() i32 { println(1); return 0; }\n").unwrap();
    let res = vestry(&app, &["run"]);
    assert!(res.status.success(), "direct core should retain std's default prelude: {}", err(&res));
}

#[test]
fn chained_reexports_do_not_expose_intermediate_packages() {
    let dir = tempfile::tempdir().unwrap();
    scoped_workspace(dir.path());
    let app = dir.path().join("app");
    set_app_manifest(dir.path(),
        "[project]\nname = \"app\"\nkind = [\"bin\"]\n\n\
         [dependencies]\nfacade2 = { path = \"../facade2\" }\n");
    std::fs::write(app.join("src/main.hv"),
        "import facade2 { square }\n\
         import facade2/option { Option }\n\
         proc main() i32 {\n\
         \x20   let o: Option<i32> = Option::Some(3);\n\
         \x20   if (o.unwrap() == 3 && square(4.0f64) == 16.0f64) { return 0; }\n\
         \x20   return 1;\n\
         }\n").unwrap();
    let res = vestry(&app, &["run"]);
    assert!(res.status.success(), "chained re-exports failed: {}", err(&res));

    for hidden in ["facade", "core"] {
        std::fs::write(app.join("src/main.hv"), format!(
            "import {hidden}/option\nproc main() i32 {{ return 0; }}\n")).unwrap();
        let res = vestry(&app, &["build"]);
        assert!(!res.status.success(), "{hidden} must stay transitive");
        assert!(err(&res).contains(&format!(
            "package '{hidden}' is not a direct dependency of 'app'")),
            "wrong diagnostic: {}", err(&res));
    }
}

#[test]
fn core_can_compile_without_std() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("main.hv");
    let executable = dir.path().join(if cfg!(windows) { "app.exe" } else { "app" });
    std::fs::write(&source, "import core/option { Option }\n\
        import core/math { square }\n\
        proc main() i32 {\n\
        \x20 let x: Option<i32> = Option::Some(5);\n\
        \x20 if (x.unwrap() == 5 && square(3.0f64) == 9.0f64) { return 0; }\n\
        \x20 return 1;\n\
        }\n").unwrap();
    let core = common::std_meta().with_file_name("core.hvmeta");
    let build = Command::new(common::havenc())
        .arg(&source)
        .args(["--package-name", "app", "--prelude", "core"])
        .arg("--dep").arg(format!("core={}", core.display()))
        .arg("-o").arg(&executable)
        .env("HAVEN_STD", "")
        .output().unwrap();
    assert!(build.status.success(), "core-only build failed: {}", err(&build));
    let run = Command::new(&executable).output().unwrap();
    assert!(run.status.success(), "core-only program failed: {}", err(&run));
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

    let res = vestry(&dir.path().join("app"), &["build"]);
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
    std::fs::write(dir.path().join("example_lib/vestry.toml"),
        "[project]\nname = \"example_lib\"\nversion = \"0.1.0\"\nkind = [\"cdylib\"]\n").unwrap();

    let res = vestry(&dir.path().join("app"), &["build"]);
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

    let res = vestry(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a missing dependency path must be reported");
    let e = err(&res);
    assert!(e.contains("vestry.toml") && e.contains("example_lib"), "got: {e}");
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

    let res = vestry(&dir.path().join("app"), &["build"]);
    assert!(!res.status.success(), "a bare-string dependency must be rejected");
    assert!(err(&res).contains("path"), "should point at the path form; got: {}", err(&res));
}

/// A `[[c]]` table on a *binary* project: `vestry` forwards its `files` as
/// `--c-file` and `libs` as `--link-lib`, so a program can ship C glue and link a
/// system library. Proves the manifest wiring end to end - the C symbol resolves
/// and the program runs. (`libs = ["m"]` stands in for a real system lib like
/// raylib; libm is guaranteed on the test host.)
#[test]
fn binary_with_c_table_builds_and_runs() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("app/vestry.toml",
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

    let res = vestry(&dir.path().join("app"), &["run"]);
    assert!(res.status.success(), "a bin with a [[c]] table must build and run: {}", err(&res));
    // hypot(3, 4) == 5: the C from `files` compiled in, and `-lm` from `libs` linked.
    assert!(out(&res).contains('5'), "unexpected program output:\n{}", out(&res));
}
