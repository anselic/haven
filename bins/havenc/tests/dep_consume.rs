//! Consumer tests for `--dep name=path.hvmeta`: building a program against a
//! compiled Haven library.
//!
//! Each test lays out a tiny library `foo` and a program `app` in fresh temp
//! dirs, produces `foo.hvmeta` with `havenc --lib`, then builds `app` with
//! `--dep foo=foo.hvmeta` and inspects the result — exercising the whole
//! producer→consumer path end to end.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

mod common;

/// Write `(relative path, contents)` files under `root`, creating parent dirs.
fn scaffold(root: &Path, files: &[(&str, &str)]) {
    for (rel, contents) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
    }
}

/// `havenc <entry> --package-name <pkg> --lib -o <out>` — produce a `.hvmeta`.
fn build_lib(cwd: &Path, entry: &str, pkg: &str, out: &Path) -> std::process::Output {
    common::havenc_cmd()
        .current_dir(cwd)
        .args([entry, "--package-name", pkg, "--lib", "-o"])
        .arg(out)
        .output()
        .expect("failed to spawn havenc")
}

/// `havenc <entry> --package-name <pkg> [--dep ...]... -o <out> --emit-ir` —
/// build a program and keep the `.ll` so tests can inspect emitted symbols.
fn build_app(cwd: &Path, entry: &str, pkg: &str, deps: &[&str], out: &Path)
    -> std::process::Output
{
    let mut cmd = common::havenc_cmd();
    cmd.current_dir(cwd)
        .args([entry, "--package-name", pkg])
        .arg("-o").arg(out)
        .arg("--emit-ir");
    for d in deps {
        cmd.arg("--dep").arg(d);
    }
    cmd.output().expect("failed to spawn havenc")
}

/// The set of function symbols the module *defines*, extracted from LLVM IR:
/// the name between `@` and `(` on each `define` line (quoted or not). Enough to
/// assert two builds emit the same symbols and that nothing is defined twice.
fn defined_symbols(ll: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in ll.lines() {
        let line = line.trim_start();
        if !line.starts_with("define") { continue; }
        let Some(at) = line.find('@') else { continue };
        let rest = &line[at + 1..];
        // the symbol runs to the next `(`; it may be `"..."`-quoted.
        let end = rest.find('(').unwrap_or(rest.len());
        let sym = rest[..end].trim().trim_matches('"');
        out.insert(sym.to_string());
    }
    out
}

fn stdout_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n")
}

// The library `foo`: a root aggregator, a sibling module, and a nested one, plus
// an `import std/math` (whose module must resolve to the leaf's single embedded
// std, not travel in the artifact).
fn foo_lib() -> Vec<(&'static str, &'static str)> {
    vec![
        ("foo/lib.hv", "import geo { Point }\n\
                        import dsp/osc { ramp }\n\
                        pub proc scale(p: Point, k: i32) i32 { return (p.x + p.y) * k + ramp(1); }\n"),
        ("foo/geo.hv", "pub struct Point { x: i32, y: i32 }\n\
                        pub proc mk(x: i32, y: i32) Point { return Point { x: x, y: y }; }\n"),
        ("foo/dsp/osc.hv", "pub proc ramp(n: i32) i32 { return n * 10; }\n"),
    ]
}

const APP_MAIN: &str = "import foo/geo { Point, mk }\n\
                        import foo { scale }\n\
                        proc main() i32 {\n\
                        \x20   let p: Point = mk(3, 4);\n\
                        \x20   println(scale(p, 2));\n\
                        \x20   return 0;\n\
                        }\n";

/// End to end: a program builds against a dep, runs correctly, and the dep's
/// items are emitted under the *dep's* package-anchored symbols (`foo.geo$mk`,
/// the root's bare `foo$scale`) — not the leaf's `app.*`. That symbol agreement,
/// with no symbol-table exchange, is the whole point of package-anchored naming.
#[test]
fn builds_runs_and_anchors_dep_symbols() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &foo_lib());
    std::fs::write(dir.path().join("main.hv"), APP_MAIN).unwrap();

    let foo = dir.path().join("foo");
    assert!(build_lib(dir.path(), "foo/lib.hv", "foo", &foo).status.success());

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["foo=foo.hvmeta"], &app);
    assert!(res.status.success(), "app build failed: {}", String::from_utf8_lossy(&res.stderr));

    // runtime behavior: (3+4)*2 + ramp(1)==10 -> 24.
    let run = Command::new(exe(&app)).output().unwrap();
    assert!(run.status.success());
    assert_eq!(stdout_of(&run), "24\n");

    // the dep's symbols are anchored to `foo`, exactly what `foo` emits standalone.
    let syms = defined_symbols(&std::fs::read_to_string(app.with_extension("ll")).unwrap());
    assert!(syms.contains("foo$scale"), "missing foo$scale in {syms:?}");
    assert!(syms.contains("foo.geo$mk"), "missing foo.geo$mk in {syms:?}");
    assert!(syms.contains("foo.dsp.osc$ramp"), "missing foo.dsp.osc$ramp in {syms:?}");
    // nothing from the dep leaked into the leaf's `app.*` namespace.
    assert!(!syms.iter().any(|s| s.starts_with("app.")), "dep item slugged under app: {syms:?}");
}

/// The fingerprint/naming payoff: building the same program+dep from two
/// unrelated absolute locations emits byte-identical symbols, and no checkout
/// path leaks into the artifact. (The dep's slugs are a pure function of its
/// package name and relative keys, both of which the `.hvmeta` carries.)
#[test]
fn dep_build_is_location_independent() {
    let build_at = |root: &Path| -> BTreeSet<String> {
        scaffold(root, &foo_lib());
        std::fs::write(root.join("main.hv"), APP_MAIN).unwrap();
        let foo = root.join("foo");
        assert!(build_lib(root, "foo/lib.hv", "foo", &foo).status.success());
        let app = root.join("app");
        assert!(build_app(root, "main.hv", "app", &["foo=foo.hvmeta"], &app).status.success());
        defined_symbols(&std::fs::read_to_string(app.with_extension("ll")).unwrap())
    };
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    assert_eq!(build_at(a.path()), build_at(b.path()),
        "symbol set differs by build location");
}

/// Behavioral equivalence of inline vs. dep. Path A puts `foo`'s source in the
/// program's own tree (merged from disk the pre-existing way); Path B consumes it
/// as a dep. The *runtime behavior* is identical — which is what a consumer cares
/// about. (Symbols differ by design: inline, `foo`'s code is part of package
/// `app` and slugs `app.*`; as a dep it stays package `foo` and slugs `foo.*`.
/// A single `havenc` invocation compiles one package, so inline cannot reproduce
/// the `foo.*` symbols — only the observable behavior is the shared invariant.)
#[test]
fn inline_and_dep_behave_identically() {
    // Path A: foo's modules live in app's own tree, imported relatively.
    let a = tempfile::tempdir().unwrap();
    scaffold(a.path(), &[
        ("geo.hv", "pub struct Point { x: i32, y: i32 }\n\
                    pub proc mk(x: i32, y: i32) Point { return Point { x: x, y: y }; }\n"),
        ("dsp/osc.hv", "pub proc ramp(n: i32) i32 { return n * 10; }\n"),
        ("lib.hv", "import geo { Point }\n\
                    import dsp/osc { ramp }\n\
                    pub proc scale(p: Point, k: i32) i32 { return (p.x + p.y) * k + ramp(1); }\n"),
        ("main.hv", "import geo { Point, mk }\n\
                     import lib { scale }\n\
                     proc main() i32 { let p: Point = mk(3, 4); println(scale(p, 2)); return 0; }\n"),
    ]);
    let a_out = a.path().join("app");
    assert!(build_app(a.path(), "main.hv", "app", &[], &a_out).status.success());
    let a_run = Command::new(exe(&a_out)).output().unwrap();

    // Path B: same program, foo consumed as a dep.
    let b = tempfile::tempdir().unwrap();
    scaffold(b.path(), &foo_lib());
    std::fs::write(b.path().join("main.hv"), APP_MAIN).unwrap();
    assert!(build_lib(b.path(), "foo/lib.hv", "foo", &b.path().join("foo")).status.success());
    let b_out = b.path().join("app");
    assert!(build_app(b.path(), "main.hv", "app", &["foo=foo.hvmeta"], &b_out).status.success());
    let b_run = Command::new(exe(&b_out)).output().unwrap();

    assert_eq!(stdout_of(&a_run), stdout_of(&b_run));
    assert_eq!(stdout_of(&a_run), "24\n");
}

/// Generics cross the boundary: the dep exposes a generic struct + constructor,
/// the program instantiates them with a program-defined type. The leaf's
/// monomorphizer specializes the dep's *template* bodies with `app`'s type, so
/// this proves source (not pre-monomorphized code) is what has to ship.
#[test]
fn generics_across_boundary() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("bar/lib.hv", "pub struct Pair<T> { a: T, b: T }\n\
                        pub proc mkpair<T>(a: T, b: T) Pair<T> { return Pair::<T> { a: a, b: b }; }\n"),
    ]);
    let bar = dir.path().join("bar");
    assert!(build_lib(dir.path(), "bar/lib.hv", "bar", &bar).status.success(),
        "generic lib should validate and emit");

    std::fs::write(dir.path().join("main.hv"),
        "import bar { Pair, mkpair }\n\
         struct V { n: i32 }\n\
         proc main() i32 {\n\
         \x20   let p: Pair<V> = mkpair::<V>(V { n: 5 }, V { n: 7 });\n\
         \x20   println(p.a.n + p.b.n);\n\
         \x20   return 0;\n\
         }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["bar=bar.hvmeta"], &app);
    assert!(res.status.success(), "generic dep build failed: {}", String::from_utf8_lossy(&res.stderr));
    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(stdout_of(&run), "12\n");
}

/// The dep and the program both `import std/math`. std must resolve to the leaf's
/// single embedded copy for both, so `square` is defined exactly once — a second
/// copy would be a duplicate-symbol link error. The program links and runs.
#[test]
fn std_is_shared_not_doubled() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("baz/lib.hv", "import std/math { square }\n\
                        pub proc energy(x: f64) f64 { return square(x) + 1.0f64; }\n"),
    ]);
    let baz = dir.path().join("baz");
    assert!(build_lib(dir.path(), "baz/lib.hv", "baz", &baz).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "import baz { energy }\n\
         import std/math { square }\n\
         proc main() i32 {\n\
         \x20   let ok: bool = energy(3.0f64) + square(2.0f64) > 13.0f64;\n\
         \x20   println(ok);\n\
         \x20   return 0;\n\
         }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["baz=baz.hvmeta"], &app);
    // linking is itself the test: if std were loaded twice, `std.math$square`
    // would be a duplicate LLVM symbol and this link would fail.
    assert!(res.status.success(), "std-shared build failed: {}", String::from_utf8_lossy(&res.stderr));

    // and, directly: exactly one definition of the mangled std proc.
    let ll = std::fs::read_to_string(app.with_extension("ll")).unwrap();
    let defs = ll.lines()
        .filter(|l| l.trim_start().starts_with("define") && l.contains("std.math$square"))
        .count();
    assert_eq!(defs, 1, "std/math::square defined {defs} times (std double-loaded?)");

    // (9+1) + 4 == 14 > 13.
    let run = Command::new(exe(&app)).output().unwrap();
    assert!(run.status.success());
    assert_eq!(stdout_of(&run), "true\n");
}

/// Deferred checking: a lib generic that only violates ownership *for some
/// instantiations* passes the lib build (no concrete instance exists there) and
/// is caught at the leaf, where it is instantiated — with a span pointing into
/// the *library's* source, which the leaf can quote because the source travels in
/// the artifact. This is the whole reason `.hvmeta` ships source, not checked code.
#[test]
fn ownership_error_in_dep_generic_surfaces_at_leaf() {
    let dir = tempfile::tempdir().unwrap();
    // `twice` uses its owned `x` after moving it — fine for a `Copy` T, an
    // ownership error for a non-`Copy` one. The lib itself has no instances, so
    // it builds clean.
    scaffold(dir.path(), &[
        ("leak/lib.hv", "pub struct Wrap<T> { v: T }\n\
                         pub proc twice<T>(x: T) Wrap<T> {\n\
                         \x20   let a: Wrap<T> = Wrap::<T> { v: x };\n\
                         \x20   let b: Wrap<T> = Wrap::<T> { v: x };\n\
                         \x20   return a;\n\
                         }\n"),
    ]);
    let leak = dir.path().join("leak");
    assert!(build_lib(dir.path(), "leak/lib.hv", "leak", &leak).status.success(),
        "the generic template has no instance in the lib, so it must build clean");

    // instantiate it with a non-`Copy` type (`String`), triggering the misuse.
    std::fs::write(dir.path().join("main.hv"),
        "import leak { twice, Wrap }\n\
         import std/string { String }\n\
         proc main() i32 {\n\
         \x20   let s: String = String::from_str(\"hi\");\n\
         \x20   let w: Wrap<String> = twice::<String>(s);\n\
         \x20   return 0;\n\
         }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["leak=leak.hvmeta"], &app);
    assert!(!res.status.success(), "the instantiated ownership violation must fail the leaf build");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("Ownership error"), "expected an ownership diagnostic; got: {err}");
    // the span resolves into the *library's* source, not the leaf's.
    assert!(err.contains("leak/lib.hv"), "span should point into the dep's source; got: {err}");
}

/// Importing a module the dep does not have fails with a diagnostic naming the
/// package and the module — not a panic and not a misleading on-disk lookup.
#[test]
fn missing_dep_module_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &foo_lib());
    assert!(build_lib(dir.path(), "foo/lib.hv", "foo", &dir.path().join("foo")).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "import foo/nope { thing }\nproc main() i32 { return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["foo=foo.hvmeta"], &app);
    assert!(!res.status.success(), "importing a missing dep module must fail");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("foo") && err.contains("nope"),
        "diagnostic should name package and module; got: {err}");
    assert!(!err.contains("panic"), "should be a diagnostic, not a panic: {err}");
    assert!(!app.with_extension("ll").exists(), "no artifact on a resolution error");
}

/// A `.hvmeta` whose `format_version` this `havenc` doesn't accept fails with a
/// clear diagnostic rather than miscompiling. Built directly through `haven_meta`
/// so the version can be set to an unsupported value.
#[test]
fn version_mismatch_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let modules = vec![haven_meta::MetaModule {
        key: "lib.hv".into(),
        source: "pub proc thing() i32 { return 1; }\n".into(),
        is_root: true,
    }];
    let fp = haven_meta::fingerprint("qux", "0.0.0", &modules, &[], &[]);
    let bad = haven_meta::HavenMeta {
        header: haven_meta::Header {
            format_version: haven_meta::FORMAT_VERSION + 1,
            havenc_version: "0.0.0".into(),
            package_name: "qux".into(),
            fingerprint: fp,
        },
        modules,
        native: vec![],
        link_libs: vec![],
    };
    let art = dir.path().join("qux.hvmeta");
    haven_meta::write(&art, &bad).unwrap();

    std::fs::write(dir.path().join("main.hv"),
        "import qux { thing }\nproc main() i32 { return thing(); }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app",
        &[&format!("qux={}", art.display())], &app);
    assert!(!res.status.success(), "an incompatible artifact must fail the build");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("version"), "diagnostic should mention the version; got: {err}");
    assert!(!err.contains("panic"), "should be a diagnostic, not a panic: {err}");
}

/// A malformed `--dep` value (no `=`) is rejected up front with a usage-shaped
/// diagnostic, before any compilation.
#[test]
fn malformed_dep_spec_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.hv"), "proc main() i32 { return 0; }\n").unwrap();
    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["foo"], &app);
    assert!(!res.status.success(), "a --dep without `=` must be rejected");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("NAME=PATH"), "should explain the expected shape; got: {err}");
}

/// The name a dep is bound to must match the artifact's own package name, or an
/// `import <name>/...` would resolve to modules slugged under a different
/// namespace than written. Caught before compilation.
#[test]
fn mismatched_dep_name_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &foo_lib());
    assert!(build_lib(dir.path(), "foo/lib.hv", "foo", &dir.path().join("foo")).status.success());
    std::fs::write(dir.path().join("main.hv"),
        "import wrongname { scale }\nproc main() i32 { return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["wrongname=foo.hvmeta"], &app);
    assert!(!res.status.success(), "binding a dep under the wrong name must fail");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("foo") && err.contains("wrongname"),
        "diagnostic should name both the bound and actual package; got: {err}");
}

/// A package whose nested module needs one from the package *root*. An ordinary
/// import only reaches downward — its segments are pushed onto the importing
/// module's own directory — so `nest/deep/inner.hv` cannot say `import util`:
/// that would name `nest/deep/util.hv`. `self/util` anchors at the root instead.
///
/// Both resolution paths are exercised: on disk while `nest` is compiled, and
/// against the artifact when the program consumes it (a dep's internal imports
/// resolve inside its `.hvmeta`, not on the filesystem).
#[test]
fn self_import_anchors_at_the_package_root() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("nest/lib.hv", "import deep/inner { boost }\n\
                         pub proc go(n: i32) i32 { return boost(n); }\n"),
        ("nest/util.hv", "pub proc scale(n: i32) i32 { return n * 5; }\n"),
        ("nest/deep/inner.hv", "import self/util { scale }\n\
                                pub proc boost(n: i32) i32 { return scale(n) + 1; }\n"),
    ]);
    let nest = dir.path().join("nest");
    let built = build_lib(dir.path(), "nest/lib.hv", "nest", &nest);
    assert!(built.status.success(),
        "on-disk `self/` should resolve: {}", String::from_utf8_lossy(&built.stderr));

    std::fs::write(dir.path().join("main.hv"),
        "import nest { go }\nproc main() i32 { println(go(4)); return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["nest=nest.hvmeta"], &app);
    assert!(res.status.success(),
        "`self/` inside a dep should resolve: {}", String::from_utf8_lossy(&res.stderr));

    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(stdout_of(&run), "21\n"); // 4*5 + 1

    // exactly one `util` — `self/util` must not be a second copy of anything.
    let syms = defined_symbols(&std::fs::read_to_string(app.with_extension("ll")).unwrap());
    let utils: Vec<_> = syms.iter().filter(|s| s.contains("util$scale")).collect();
    assert_eq!(utils, vec!["nest.util$scale"], "wrong or duplicated util module: {syms:?}");
}

/// `self` is reserved: it means the package root even when a dependency is bound
/// under that name, so an import can never mean two things depending on flags.
#[test]
fn self_wins_over_a_dependency_named_self() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        // a dep that would answer to `self/thing` if `self` were an ordinary name.
        ("self/lib.hv", "pub proc thing() i32 { return 99; }\n"),
        ("self/thing.hv", "pub proc thing() i32 { return 99; }\n"),
        // the program's own root-anchored module of the same path.
        ("thing.hv", "pub proc thing() i32 { return 7; }\n"),
    ]);
    assert!(build_lib(dir.path(), "self/lib.hv", "self", &dir.path().join("self"))
        .status.success());

    std::fs::write(dir.path().join("main.hv"),
        "import self/thing { thing }\nproc main() i32 { return thing(); }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["self=self.hvmeta"], &app);
    assert!(res.status.success(), "build failed: {}", String::from_utf8_lossy(&res.stderr));
    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(run.status.code(), Some(7), "`self/` must name the package root, not the dep");
}

/// `havenc <entry> --lib --package-name <pkg> [--c-file..] [--link-lib..] -o out`
/// — produce a `.hvmeta` that ships native C and/or declares libraries to link.
fn build_lib_native(cwd: &Path, entry: &str, pkg: &str, c_files: &[&str],
                    libs: &[&str], out: &Path) -> std::process::Output {
    let mut cmd = common::havenc_cmd();
    cmd.current_dir(cwd)
        .args([entry, "--package-name", pkg, "--lib", "-o"])
        .arg(out);
    for c in c_files { cmd.arg("--c-file").arg(c); }
    for l in libs { cmd.arg("--link-lib").arg(l); }
    cmd.output().expect("failed to spawn havenc")
}

/// The Stage 3 payoff: a dep ships C *source* in its `.hvmeta`, and the leaf
/// compiles and links it — the package's native code travels without the
/// compiler embedding it. The C here leans on libm, so the dep also declares
/// `--link-lib m`; the program runs only if both the object and the `-lm` it
/// needs made it onto the link line.
#[test]
fn dep_ships_c_compiled_and_linked_at_leaf() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("mathx/lib.hv", "extern mathx_root(x: i32) i32;\n\
                          pub proc root(x: i32) i32 { return mathx_root(x); }\n"),
        ("mathx/root.c", "#include <math.h>\n\
                          int mathx_root(int x) { return (int)llround(sqrt((double)x)); }\n"),
    ]);
    let built = build_lib_native(dir.path(), "mathx/lib.hv", "mathx",
        &["mathx/root.c"], &["m"], &dir.path().join("mathx"));
    assert!(built.status.success(), "lib build failed: {}",
        String::from_utf8_lossy(&built.stderr));

    std::fs::write(dir.path().join("main.hv"),
        "import mathx { root }\nproc main() i32 { println(root(16)); return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["mathx=mathx.hvmeta"], &app);
    assert!(res.status.success(),
        "leaf should compile and link the dep's C: {}",
        String::from_utf8_lossy(&res.stderr));

    let run = Command::new(exe(&app)).output().unwrap();
    assert!(run.status.success());
    assert_eq!(stdout_of(&run), "4\n"); // llround(sqrt(16))
}

/// A dep's declared `--link-lib` becomes a real `-l` on the leaf's link line,
/// independent of the always-on `-lm`. Proven negatively: a library that does
/// not exist makes the *link* fail, naming it — so the flag demonstrably
/// reached the linker rather than being dropped.
#[test]
fn dep_link_lib_reaches_the_linker() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("z/lib.hv", "pub proc id(x: i32) i32 { return x; }\n"),
    ]);
    let built = build_lib_native(dir.path(), "z/lib.hv", "z",
        &[], &["nonexistentlib_zzz"], &dir.path().join("z"));
    assert!(built.status.success());

    std::fs::write(dir.path().join("main.hv"),
        "import z { id }\nproc main() i32 { return id(0); }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", "app", &["z=z.hvmeta"], &app);
    assert!(!res.status.success(),
        "a dep's declared `-l` must reach the linker (and here fail to resolve)");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("nonexistentlib_zzz"),
        "linker should report the missing dep lib; got: {err}");
}

/// Windows appends `.exe`; elsewhere the executable is the bare output path.
fn exe(base: &Path) -> std::path::PathBuf {
    if cfg!(target_os = "windows") { base.with_extension("exe") } else { base.to_path_buf() }
}
