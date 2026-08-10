//! `--prelude <package>`: taking the prelude, and the lang items that come with
//! it, from somewhere other than the embedded stdlib.
//!
//! The package under test is `mini`, a two-module stand-in for a standard
//! library: it marks itself `@!prelude`, declares `@lang(delete)`, and is built
//! by nominating *itself* — which is the only way a stdlib can be compiled, its
//! claims being ones no other package would honour on its behalf.

use std::path::Path;
use std::process::Command;

const HAVENC: &str = env!("CARGO_BIN_EXE_havenc");

fn scaffold(root: &Path, files: &[(&str, &str)]) {
    for (rel, contents) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
    }
}

/// `havenc <entry> --package-name <pkg> --lib -o <out>`, plus any extra flags.
fn build_lib(cwd: &Path, entry: &str, pkg: &str, extra: &[&str], out: &Path)
    -> std::process::Output
{
    Command::new(HAVENC)
        .current_dir(cwd)
        .args([entry, "--package-name", pkg, "--lib"])
        .args(extra)
        .arg("-o").arg(out)
        .output()
        .expect("failed to spawn havenc")
}

/// `havenc <entry> --package-name app [flags] -o <out>` — build a program.
fn build_app(cwd: &Path, entry: &str, flags: &[&str], out: &Path) -> std::process::Output {
    Command::new(HAVENC)
        .current_dir(cwd)
        .args([entry, "--package-name", "app"])
        .args(flags)
        .arg("-o").arg(out)
        .output()
        .expect("failed to spawn havenc")
}

fn exe(base: &Path) -> std::path::PathBuf {
    if cfg!(target_os = "windows") { base.with_extension("exe") } else { base.to_path_buf() }
}

fn stdout_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n")
}

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A miniature stdlib. `lib.hv` is the prelude (a root module, not a file named
/// `prelude` — the point being that the compiler reads the mark, not the name);
/// `text.hv` is an ordinary sibling module, present to show that a prelude
/// package's *own* modules see the prelude too.
///
/// `rt_puts` comes from the runtime archive every Haven program links, so `mini`
/// can print without borrowing anything from the embedded std.
fn mini_lib() -> Vec<(&'static str, &'static str)> {
    vec![
        ("mini/lib.hv",
         "@!prelude\n\
          import text { shout }\n\
          extern rt_puts(s: str) void;\n\
          @lang(delete)\n\
          pub trait Delete {\n\
          \x20   proc delete(*self);\n\
          }\n\
          pub proc say(s: str) void { rt_puts(s); rt_puts(\"\\n\"); }\n\
          pub proc yell(s: str) void { say(shout(s)); }\n\
          pub proc double(x: i32) i32 { return x * 2; }\n"),
        // no import of `lib` here: `shout` calls `say`, which is in scope only
        // because this module's own package supplies the prelude.
        ("mini/text.hv",
         "pub proc shout(s: str) str { say(\"(shouting)\"); return s; }\n"),
    ]
}

/// Build `mini` into an artifact, nominating itself as its own prelude.
fn build_mini(dir: &Path) -> std::process::Output {
    build_lib(dir, "mini/lib.hv", "mini", &["--prelude", "mini"], &dir.join("mini"))
}

/// The whole point: a program's prelude comes from a dependency. `say` and
/// `double` are called with no import at all, and the program never mentions
/// `mini`.
#[test]
fn prelude_comes_from_the_nominated_package() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &mini_lib());
    let built = build_mini(dir.path());
    assert!(built.status.success(),
        "a package that supplies its own prelude must build: {}", stderr_of(&built));

    std::fs::write(dir.path().join("main.hv"),
        "proc main() i32 {\n\
         \x20   yell(\"hello from mini\");\n\
         \x20   return double(21);\n\
         }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv",
        &["--dep", "mini=mini.hvmeta", "--prelude", "mini"], &app);
    assert!(res.status.success(), "prelude-from-dep build failed: {}", stderr_of(&res));

    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(stdout_of(&run), "(shouting)\nhello from mini\n");
    assert_eq!(run.status.code(), Some(42), "double(21) should be the exit code");
}

/// ...and the embedded stdlib's prelude is *gone*, not merely supplemented.
/// `println` is std's; nominating another package must take it out of scope,
/// otherwise the two preludes would silently merge.
#[test]
fn nominating_a_prelude_displaces_the_embedded_one() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &mini_lib());
    assert!(build_mini(dir.path()).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "proc main() i32 { println(1); return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv",
        &["--dep", "mini=mini.hvmeta", "--prelude", "mini"], &app);
    assert!(!res.status.success(), "std's println must not be in scope");
    let err = stderr_of(&res);
    assert!(err.contains("println"), "diagnostic should name the unknown call; got: {err}");
}

/// A dependency that advertises a prelude (its source carries `@!prelude`) is
/// *discovered* as the program's prelude with no `--prelude` flag — the mark is
/// the single source of truth. `mini`'s `yell`/`double` come into scope
/// unimported, and the embedded std is displaced (one prelude, never two merged).
#[test]
fn a_dependency_prelude_is_discovered_without_a_flag() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &mini_lib());
    assert!(build_mini(dir.path()).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "proc main() i32 { yell(\"hi\"); return double(21); }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", &["--dep", "mini=mini.hvmeta"], &app);
    assert!(res.status.success(), "mini should be discovered as the prelude: {}", stderr_of(&res));
    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(stdout_of(&run), "(shouting)\nhi\n");
    assert_eq!(run.status.code(), Some(42), "double(21) should be the exit code");
}

/// The leaf cannot also declare `@!prelude` while its prelude comes from a
/// dependency. A dependency's mark is discovered (above); the *leaf's* own mark on
/// top of that is a contradiction — a program has one prelude — so it errors,
/// naming the supplier and how to make this package the prelude instead. This is
/// the `@lang`-like uniqueness you get once marking is the source of truth.
#[test]
fn leaf_cannot_declare_a_prelude_when_a_dependency_supplies_one() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &mini_lib());
    assert!(build_mini(dir.path()).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "@!prelude\n\
         proc main() i32 { return double(0); }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", &["--dep", "mini=mini.hvmeta"], &app);
    assert!(!res.status.success(), "leaf + dependency prelude is one too many");
    let err = stderr_of(&res);
    assert!(err.contains("has no effect") && err.contains("mini"),
        "should name the supplier and that the leaf's mark does nothing; got: {err}");
}

/// Two dependencies that each supply a prelude is the ambiguity `@!prelude` now
/// rejects. Rather than silently pick, it asks for `--prelude`; nominating one
/// resolves it and makes the other's mark inert (so that package stays usable for
/// its other modules — the "usable as a plain dependency" escape).
#[test]
fn two_prelude_dependencies_conflict_until_one_is_nominated() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &mini_lib());
    assert!(build_mini(dir.path()).status.success());
    // a second, independent prelude-bearing package.
    scaffold(dir.path(), &[
        ("mini2/lib.hv",
         "@!prelude\n\
          extern rt_puts(s: str) void;\n\
          @lang(delete)\n\
          pub trait Delete {\n\
          \x20   proc delete(*self);\n\
          }\n\
          pub proc hush(s: str) void { rt_puts(s); }\n"),
    ]);
    assert!(build_lib(dir.path(), "mini2/lib.hv", "mini2", &["--prelude", "mini2"],
                      &dir.path().join("mini2")).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "proc main() i32 { return double(21); }\n").unwrap();

    // both provide a prelude, no `--prelude`: ambiguous, and it says so.
    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv",
        &["--dep", "mini=mini.hvmeta", "--dep", "mini2=mini2.hvmeta"], &app);
    assert!(!res.status.success(), "two prelude providers must not silently pick one");
    let err = stderr_of(&res);
    assert!(err.contains("more than one dependency supplies a prelude") && err.contains("--prelude"),
        "got: {err}");

    // nominate mini: mini2's mark goes inert, mini is the prelude, `double` in scope.
    let res2 = build_app(dir.path(), "main.hv",
        &["--dep", "mini=mini.hvmeta", "--dep", "mini2=mini2.hvmeta", "--prelude", "mini"], &app);
    assert!(res2.status.success(), "nominating one must resolve the ambiguity: {}", stderr_of(&res2));
    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(run.status.code(), Some(42), "double(21) should be the exit code");
}

/// Nominating a package that is not bound as a dependency is caught up front,
/// with the fix in the message — not as a cascade of unknown-name errors.
#[test]
fn prelude_flag_requires_a_bound_package() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.hv"), "proc main() i32 { return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", &["--prelude", "ghost"], &app);
    assert!(!res.status.success());
    let err = stderr_of(&res);
    assert!(err.contains("ghost") && err.contains("--dep"),
        "should name the package and how to bind it; got: {err}");
}

/// Nominating a package that does not claim to be a prelude fails, rather than
/// leaving the program with an empty implicit scope.
#[test]
fn nominated_package_must_claim_to_be_a_prelude() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("plain/lib.hv", "pub proc thing() i32 { return 1; }\n"),
    ]);
    assert!(build_lib(dir.path(), "plain/lib.hv", "plain", &[],
                      &dir.path().join("plain")).status.success());
    std::fs::write(dir.path().join("main.hv"), "proc main() i32 { return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv",
        &["--dep", "plain=plain.hvmeta", "--prelude", "plain"], &app);
    assert!(!res.status.success());
    let err = stderr_of(&res);
    assert!(err.contains("supplies no prelude"), "got: {err}");
}

/// Supplying the prelude and supplying the lang items are one job. A package
/// that takes over the first and leaves `Delete` to nobody would compile a
/// program in which every owning type is silently `Copy` — so it is rejected, at
/// the point the package claims the prelude role (building it as its own prelude).
#[test]
fn a_prelude_package_must_supply_the_lang_items() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("half/lib.hv",
         "@!prelude\n\
          pub proc thing() i32 { return 1; }\n"),
    ]);
    // half nominates itself the prelude but declares no `@lang(delete)`.
    let res = build_lib(dir.path(), "half/lib.hv", "half", &["--prelude", "half"],
                        &dir.path().join("half"));
    assert!(!res.status.success(), "a prelude with no `Delete` must be rejected");
    assert!(stderr_of(&res).contains("declares no `@lang(delete)`"), "got: {}", stderr_of(&res));
}

/// `--no-prelude` and `--prelude` ask for contradictory things; clap rejects the
/// pair rather than one silently winning.
#[test]
fn no_prelude_and_prelude_are_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("main.hv"), "proc main() i32 { return 0; }\n").unwrap();
    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", &["--no-prelude", "--prelude", "mini"], &app);
    assert!(!res.status.success());
    assert!(stderr_of(&res).contains("cannot be used with"), "got: {}", stderr_of(&res));
}

/// A self-preluding stdlib stub built under the package name `std`, so it can be
/// bound as `--dep std=...` — the "std is an ordinary dependency" case. `text.hv`
/// is shipped only because the root reaches it, and the embedded stdlib has *no*
/// `text` module: an `import std/text` that resolves is proof the dependency, not
/// the embedded tree, served it.
fn std_stub() -> Vec<(&'static str, &'static str)> {
    vec![
        ("stdsrc/lib.hv",
         "@!prelude\n\
          import text { shout }\n\
          extern rt_puts(s: str) void;\n\
          @lang(delete)\n\
          pub trait Delete {\n\
          \x20   proc delete(*self);\n\
          }\n\
          pub proc say(s: str) void { rt_puts(s); rt_puts(\"\\n\"); }\n\
          pub proc double(x: i32) i32 { return x * 2; }\n"),
        ("stdsrc/text.hv",
         "pub proc shout(s: str) str { say(\"(shouting)\"); return s; }\n"),
    ]
}

/// Build the stub into a `std.hvmeta` artifact, nominating itself as its own
/// prelude (the only way a stdlib compiles). Its package name is `std`, which is
/// what lets `--dep std=std.hvmeta` bind it without a name mismatch.
fn build_std_stub(dir: &Path) -> std::process::Output {
    build_lib(dir, "stdsrc/lib.hv", "std", &["--prelude", "std"], &dir.join("std"))
}

/// Binding `--dep std=...` makes that package *be* std: `import std/text` reaches
/// the dependency, not the embedded tree (which has no `text` module and would
/// error). Both the imported `shout` and the implicit-prelude `say`/`double` come
/// from the dep. This is the gap that used to leave a std-bound package reachable
/// only as the prelude.
#[test]
fn std_can_be_supplied_by_a_dependency() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &std_stub());
    assert!(build_std_stub(dir.path()).status.success(),
        "the std stub must build: {}", stderr_of(&build_std_stub(dir.path())));

    std::fs::write(dir.path().join("main.hv"),
        "import std/text { shout }\n\
         proc main() i32 {\n\
         \x20   say(shout(\"world\"));\n\
         \x20   return double(21);\n\
         }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv",
        &["--dep", "std=std.hvmeta", "--prelude", "std"], &app);
    assert!(res.status.success(), "std-from-dep build failed: {}", stderr_of(&res));

    let run = Command::new(exe(&app)).output().unwrap();
    // `shout` prints "(shouting)" and returns its arg; `say` then prints it.
    assert_eq!(stdout_of(&run), "(shouting)\nworld\n");
    assert_eq!(run.status.code(), Some(42), "double(21) should be the exit code");
}

/// With std bound as a dep, the *default* prelude (`PreludeSource::Auto`) is
/// discovered there: no `--prelude` is passed, yet `say`/`double` are in implicit
/// scope, which only holds if the std dep's `@!prelude` module became the prelude. The
/// embedded stdlib must not be seeded alongside it — that pairing (embedded
/// `std/prelude` + the dep's module under the same `std.` slug) is the silent
/// double-load the fnv1a suffix hides. `--emit-ir` + a single-definition check is
/// the only way to see it: a double-load *links fine*, under two distinct slugs.
#[test]
fn binding_std_moves_the_default_prelude_and_does_not_double_load() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &std_stub());
    assert!(build_std_stub(dir.path()).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "import std/text { shout }\n\
         proc main() i32 {\n\
         \x20   say(shout(\"hi\"));\n\
         \x20   return double(0);\n\
         }\n").unwrap();

    // note: no `--prelude`. The default would be the embedded std's prelude;
    // binding std redirects it to the dep.
    let app = dir.path().join("app");
    let res = Command::new(HAVENC)
        .current_dir(dir.path())
        .args(["main.hv", "--package-name", "app", "--dep", "std=std.hvmeta",
               "--emit-ir", "-o"])
        .arg(&app)
        .output()
        .expect("failed to spawn havenc");
    assert!(res.status.success(),
        "default prelude should follow std to the dep: {}", stderr_of(&res));

    // exactly one definition of the imported std proc: a second (fnv1a-suffixed)
    // copy would mean the embedded tree was loaded alongside the dep.
    let ll = std::fs::read_to_string(app.with_extension("ll")).unwrap();
    let defs = ll.lines()
        .filter(|l| l.trim_start().starts_with("define") && l.contains("$shout"))
        .count();
    assert_eq!(defs, 1, "std/text::shout defined {defs} times (std double-loaded?)");

    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(stdout_of(&run), "(shouting)\nhi\n");
    assert_eq!(run.status.code(), Some(0));
}
