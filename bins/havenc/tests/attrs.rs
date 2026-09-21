//! Codegen attributes checked at the IR they produce: a run fixture cannot see
//! whether LLVM received inline, fast-math, or aliasing information.

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

@noalias
proc disjoint(a: *f32, b: *f32) void {}

proc may_alias(a: *f32, b: *f32) void {}

proc main() i32 {
    let a: i32 = hot(1) + cold(1);
    let b: f32 = scaled(1.0) + plain(1.0);
    disjoint(null::<*f32>(), null::<*f32>());
    may_alias(null::<*f32>(), null::<*f32>());
    return 0;
}
";

/// Compile `SRC` with `--emit-ir` and return the `.ll` text.
fn ir(dir: &Path) -> String {
    compile_ir(dir, SRC)
}

fn compile_ir(dir: &Path, source: &str) -> String {
    let entry = dir.join("main.hv");
    std::fs::write(&entry, source).unwrap();
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

#[test]
fn explicit_noalias_marks_pointer_parameters_even_without_accesses() {
    let dir = tempfile::tempdir().unwrap();
    let ir = ir(dir.path());
    assert_eq!(define_line(&ir, "$disjoint").matches("noalias").count(), 2);
    assert!(!define_line(&ir, "$may_alias").contains("noalias"));
}

#[test]
fn inferred_noalias_requires_proven_memory_origins() {
    let dir = tempfile::tempdir().unwrap();
    let ir = compile_ir(dir.path(), r#"
struct Pair { x: i32, y: i32 }
const GLOBAL: i32 = 7;

proc leaf(p: *i32, unused: *i32, n: i32) i32 {
    let alias: *i32 = p;
    *alias = *alias + n;
    return *p;
}
proc fields(p: *Pair) i32 {
    p.x = p.y + 1;
    return p.x;
}
proc fill(p: *i32, n: u64) {
    let i: u64 = 0u64;
    while (i < n) { p[i] = 9; i = i + 1u64; }
}
proc cast(p: *i32) i32 {
    let bytes: *u8 = ptr_cast::<*u8>(p);
    bytes[0u64] = 0u8;
    return *p;
}
proc overlap(a: *i32, b: *i32) i32 {
    *a = 1;
    *b = 2;
    return *a;
}
proc read_both(a: *i32, b: *i32) i32 { return *a + *b; }
proc indirect(p: **i32, q: *i32) i32 { **p = 3; return *q; }
proc reassigned(p: *i32, q: *i32, change: bool) i32 {
    if (change) { p = q; }
    *p = 4;
    return *q;
}
proc escaped_slot(p: *i32, q: *i32) i32 {
    let slot: **i32 = &p;
    *slot = q;
    *p = 5;
    return *q;
}
proc with_global(p: *i32) i32 { *p = 6; return GLOBAL; }
proc callback_body() {}
proc callback(p: *i32, f: proc() void) { *p = 7; f(); }
proc calls_leaf(p: *i32) i32 { return leaf(p, p, 1); }
proc slice_access(p: *i32, s: [i32]) i32 { *p = 8; return s[0u64]; }
proc aggregate(p: *Pair) Pair { return *p; }

proc main() i32 {
    let x: i32 = 0;
    let p: *i32 = &x;
    if (leaf(p, p, 1) != 1) { return 1; }
    let pair: Pair = Pair { x: 1, y: 2 };
    if (fields(&pair) != 3) { return 2; }
    fill(p, 1u64);
    cast(p);
    if (overlap(p, p) != 2) { return 3; }
    read_both(p, p);
    indirect(&p, p);
    reassigned(p, p, true);
    escaped_slot(p, p);
    with_global(p);
    callback(p, callback_body);
    calls_leaf(p);
    let s: [i32] = [1, 2];
    slice_access(p, s);
    let copy: Pair = aggregate(&pair);
    return 0;
}
"#);
    for name in ["$leaf", "$fields", "$fill", "$cast"] {
        assert_eq!(define_line(&ir, name).matches("noalias").count(), 1,
            "expected one proven pointer in {name}");
    }
    // The unused second pointer must not inherit the first pointer's fact.
    assert!(define_line(&ir, "$leaf").contains("ptr noalias"));
    assert!(define_line(&ir, "$leaf").contains(", ptr %"));
    for name in ["$overlap", "$read_both", "$indirect", "$reassigned",
        "$escaped_slot", "$with_global", "$callback(", "$calls_leaf",
        "$slice_access", "$aggregate"] {
        assert!(!define_line(&ir, name).contains("noalias"),
            "unproven aliasing in {name}");
    }
    let status = std::process::Command::new(dir.path().join("app"))
        .status().unwrap();
    assert!(status.success(), "alias-preserving execution failed: {status}");
}
