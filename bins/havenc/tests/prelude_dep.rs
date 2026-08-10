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

/// A prelude package used as an *ordinary* dependency is a conflict, and a loud
/// one. Its `@lang(delete)` would be a second answer to what owning a resource
/// means, and the ownership pass can honour only one — so the other package's
/// owners would quietly compile as `Copy`. Two stdlibs do not go in one program.
#[test]
fn unnominated_prelude_package_conflicts_over_lang_items() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &mini_lib());
    assert!(build_mini(dir.path()).status.success());

    std::fs::write(dir.path().join("main.hv"),
        "import mini { double }\n\
         proc main() i32 { return double(1); }\n").unwrap();

    // note: no `--prelude`, so this program's prelude is the embedded std's.
    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", &["--dep", "mini=mini.hvmeta"], &app);
    assert!(!res.status.success(), "two claimants for `@lang(delete)` must fail");
    let err = stderr_of(&res);
    assert!(err.contains("may only be declared by 'std'"),
        "diagnostic should name the package that does supply the prelude; got: {err}");
}

/// The asymmetric half: a stray `@!prelude` in a dependency is *inert*. Supplying
/// a prelude is a legitimate thing for a package to do, and that same package is
/// an ordinary library to whoever merely depends on it — its mark must not follow
/// it in and displace the consumer's own prelude. (Contrast the test above: a
/// lang item cannot be ignored this way without miscompiling.)
#[test]
fn stray_prelude_mark_in_a_dependency_is_inert() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("hopeful/lib.hv",
         "@!prelude\n\
          pub proc triple(x: i32) i32 { return x * 3; }\n"),
    ]);
    assert!(build_lib(dir.path(), "hopeful/lib.hv", "hopeful", &[],
                      &dir.path().join("hopeful")).status.success());

    // `triple` still has to be imported, and std's `println` still works.
    std::fs::write(dir.path().join("main.hv"),
        "import hopeful { triple }\n\
         proc main() i32 { println(triple(4)); return 0; }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv", &["--dep", "hopeful=hopeful.hvmeta"], &app);
    assert!(res.status.success(), "an unnominated `@!prelude` must be ignored: {}", stderr_of(&res));
    let run = Command::new(exe(&app)).output().unwrap();
    assert_eq!(stdout_of(&run), "12\n");
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
/// program in which every owning type is silently `Copy` — so it is rejected.
///
/// This path existed before `--prelude` and could not be reached from the CLI:
/// the stdlib was embedded, and it does declare `Delete`.
#[test]
fn nominated_package_must_supply_the_lang_items() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("half/lib.hv",
         "@!prelude\n\
          pub proc thing() i32 { return 1; }\n"),
    ]);
    assert!(build_lib(dir.path(), "half/lib.hv", "half", &[],
                      &dir.path().join("half")).status.success());
    std::fs::write(dir.path().join("main.hv"), "proc main() i32 { return thing(); }\n").unwrap();

    let app = dir.path().join("app");
    let res = build_app(dir.path(), "main.hv",
        &["--dep", "half=half.hvmeta", "--prelude", "half"], &app);
    assert!(!res.status.success(), "a prelude with no `Delete` must be rejected");
    let err = stderr_of(&res);
    assert!(err.contains("declares no `@lang(delete)`"), "got: {err}");
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
