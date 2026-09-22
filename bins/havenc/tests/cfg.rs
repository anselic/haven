//! Conditional-compilation behavior at the compiler boundary.

use std::path::Path;

use haven_common::target::{CInteger, TargetSpec};

mod common;

fn compile(dir: &Path, source: &str) -> std::process::Output {
    let entry = dir.join("main.hv");
    std::fs::write(&entry, source).unwrap();
    common::havenc_cmd()
        .arg(&entry)
        .args(["--no-prelude", "--emit-ir", "-o"])
        .arg(dir.join("app"))
        .output()
        .expect("failed to spawn havenc")
}

#[test]
fn cfg_selects_one_duplicate_alias_before_name_resolution() {
    let dir = tempfile::tempdir().unwrap();
    let output = compile(dir.path(), r#"
@cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))
import missing/platform/module

@cfg(target_c_long_width = "32")
type word = i32;

@cfg(target_c_long_width = "64")
type word = i64;

@cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))
proc excluded(x: ThisTypeMustNotResolve) ThisTypeMustNotResolve { return x; }

@export
proc identity(x: word) word { return x; }
proc main() i32 { return 0; }
"#);
    assert!(output.status.success(), "compile failed:\n{}",
        String::from_utf8_lossy(&output.stderr));

    let ir = std::fs::read_to_string(dir.path().join("app.ll")).unwrap();
    let target = TargetSpec::host().unwrap();
    let bits = target.c.integer(CInteger::Long).bits;
    assert!(ir.lines().any(|line|
        line.starts_with("define ") && line.contains("@identity")
            && line.contains(&format!("i{bits}"))), "wrong alias in:\n{ir}");
    assert!(!ir.contains("$excluded"));
}

#[test]
fn cfg_rejects_unknown_predicates_instead_of_disabling_silently() {
    let dir = tempfile::tempdir().unwrap();
    let output = compile(dir.path(), r#"
@cfg(target_c_lnog_width = "32")
proc typo() {}
proc main() i32 { return 0; }
"#);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown cfg predicate 'target_c_lnog_width'"),
        "unexpected diagnostic:\n{stderr}");
}

#[test]
fn cfg_rejects_unknown_values_instead_of_disabling_silently() {
    let dir = tempfile::tempdir().unwrap();
    let output = compile(dir.path(), r#"
@cfg(target_c_long_width = "banana")
proc typo() {}
proc main() i32 { return 0; }
"#);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid value 'banana'"), "unexpected diagnostic:\n{stderr}");
}
