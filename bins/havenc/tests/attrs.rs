//! The codegen-hint attributes, `@inline` and `@fastmath`, checked at the IR
//! they produce: a run fixture cannot see whether `alwaysinline` or a
//! fast-math flag was actually emitted.

use std::path::Path;

mod common;

const SRC: &str = "\
@inline(always)
proc hot(x: i32) i32 { return x + 1; }

@inline(never)
proc cold(x: i32) i32 { return x - 1; }

@fastmath(fast)
proc scaled(x: f32) f32 { return x * 2.0; }

// follows `scaled`: its flags must not leak into this one
proc plain(x: f32) f32 { return x * 2.0; }

proc main() i32 {
    let a: i32 = hot(1) + cold(1);
    let b: f32 = scaled(1.0) + plain(1.0);
    return 0;
}
";

/// Compile `SRC` with `--emit-ir` and return the `.ll` text.
fn ir(dir: &Path) -> String {
    let entry = dir.join("main.hv");
    std::fs::write(&entry, SRC).unwrap();
    let out = dir.join("app");
    let o = common::havenc_cmd()
        .arg(&entry)
        .args(["--no-prelude", "--emit-ir", "-o"])
        .arg(&out)
        .output()
        .expect("failed to spawn havenc");
    assert!(o.status.success(), "compile failed:\n{}", String::from_utf8_lossy(&o.stderr));
    std::fs::read_to_string(out.with_extension("ll")).expect("the kept .ll")
}

/// The `define ...` line of the function whose symbol contains `name`.
fn define_line<'a>(ir: &'a str, name: &str) -> &'a str {
    ir.lines()
        .find(|l| l.starts_with("define ") && l.contains(name))
        .unwrap_or_else(|| panic!("no define for {name} in:\n{ir}"))
}

#[test]
fn inline_hints_become_llvm_function_attributes() {
    let dir = tempfile::tempdir().unwrap();
    let ir = ir(dir.path());
    assert!(define_line(&ir, "$hot").contains("alwaysinline"));
    assert!(define_line(&ir, "$cold").contains("noinline"));
    assert!(!define_line(&ir, "$plain").contains("inline"));
}

#[test]
fn fastmath_flags_apply_to_that_function_only() {
    let dir = tempfile::tempdir().unwrap();
    let ir = ir(dir.path());
    let fast = ir.matches("fmul fast").count();
    let all = ir.matches("fmul").count();
    assert_eq!(fast, 1, "one `fmul` carries the flag - `scaled`'s:\n{ir}");
    assert_eq!(all, 2, "`plain` has its own, unflagged `fmul`:\n{ir}");
}
