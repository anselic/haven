//! Native code on a *non-library* build: a binary that ships its own C
//! (`--c-file`) or links a system library (`--link-lib`). A `--lib` stores these
//! in its `.hvmeta` for consumers to compile (covered by `lib_meta.rs`); here the
//! leaf compiles the C and links the libraries into the executable directly.

use std::path::Path;

mod common;

/// Write `(relative path, contents)` files under `root`, creating parent dirs.
fn scaffold(root: &Path, files: &[(&str, &str)]) {
    for (rel, contents) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
    }
}

/// A `--c-file` on an executable build is compiled and linked in, so a symbol the
/// Haven program declares `extern` and calls resolves at link time and runs.
#[test]
fn c_file_is_compiled_and_linked_into_executable() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("main.hv", "extern c_double(x: i32) i32;\n\
                     proc main() i32 {\n\
                     \x20   println(c_double(21));\n\
                     \x20   return 0;\n\
                     }\n"),
        ("glue.c", "int c_double(int x) { return x * 2; }\n"),
    ]);
    let out = dir.path().join("prog");

    let res = common::havenc_cmd()
        .current_dir(dir.path())
        .arg("main.hv")
        .arg("--c-file").arg("glue.c")
        .arg("-o").arg(&out)
        .output().expect("failed to spawn havenc");
    assert!(res.status.success(),
        "compiling a bin with --c-file failed: {}", String::from_utf8_lossy(&res.stderr));

    let run = std::process::Command::new(&out).output().expect("failed to run built binary");
    assert!(run.status.success(), "the built binary exited non-zero");
    assert_eq!(String::from_utf8_lossy(&run.stdout), "42\n");
}

/// A `--link-lib` name reaches the executable's link line as `-l<name>`: a library
/// that does not exist makes the link fail, naming it. This is the observable
/// proof that the flag is emitted (a real system lib would just link silently).
#[test]
fn link_lib_reaches_the_executable_link_line() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("main.hv", "proc main() i32 { println(\"x\"); return 0; }\n"),
    ]);
    let out = dir.path().join("prog");

    let res = common::havenc_cmd()
        .current_dir(dir.path())
        .arg("main.hv")
        .arg("--link-lib").arg("hav_no_such_lib_xyz")
        .arg("-o").arg(&out)
        .output().expect("failed to spawn havenc");
    assert!(!res.status.success(), "linking against a missing library must fail the build");
    assert!(String::from_utf8_lossy(&res.stderr).contains("hav_no_such_lib_xyz"),
        "the linker error should name the missing library, proving `-l` was passed");
}
