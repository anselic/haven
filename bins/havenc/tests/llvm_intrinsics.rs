//! `@llvm_intrinsic_exact` / `@llvm_intrinsic_infer` externs, checked at the IR
//! they produce. A wrong symbol or a wrong suffix order still links (LLVM knows
//! the intrinsic either way, or rejects it outright), so the run fixtures cannot
//! catch it; the declarations and calls have to be read.
//!
//! The diagnostics live in `cases/fail/test_llvm_intrinsic_*.hv`, and the
//! values the intrinsics compute in `cases/run/test_llvm_intrinsic.hv`.

use std::path::Path;

mod common;

/// Compile `source` freestanding with `--emit-ir` (plus `extra` flags) and
/// return the `.ll` text.
fn compile_ir(dir: &Path, source: &str, extra: &[&str]) -> String {
    let entry = dir.join("main.hv");
    std::fs::write(&entry, source).unwrap();
    let out = dir.join("app");
    let o = common::havenc_cmd()
        .arg(&entry)
        .args(["--no-prelude", "--emit-ir"])
        .args(extra)
        .arg("-o").arg(&out)
        .output()
        .expect("failed to spawn havenc");
    assert!(o.status.success(), "compile failed:\n{}", String::from_utf8_lossy(&o.stderr));
    std::fs::read_to_string(out.with_extension("ll")).expect("the kept .ll")
}

/// The `declare` lines naming `symbol`.
fn declares<'a>(ir: &'a str, symbol: &str) -> Vec<&'a str> {
    let needle = format!("@{symbol}(");
    ir.lines().filter(|l| l.starts_with("declare ") && l.contains(&needle)).collect()
}

/// The `call` lines naming `symbol`.
fn calls<'a>(ir: &'a str, symbol: &str) -> Vec<&'a str> {
    let needle = format!("@{symbol}(");
    ir.lines().filter(|l| l.contains(" call ") && l.contains(&needle)).collect()
}

/// The body of the function whose symbol contains `name`, from its `define`
/// line to the closing brace.
fn function_body<'a>(ir: &'a str, name: &str) -> &'a str {
    let start = ir.lines()
        .find(|l| l.starts_with("define ") && l.contains(name))
        .and_then(|l| ir.find(l))
        .unwrap_or_else(|| panic!("no define for {name} in:\n{ir}"));
    let len = ir[start..].find("\n}").expect("function has a closing brace");
    &ir[start..start + len]
}

const BINDINGS: &str = r#"
@llvm_intrinsic_infer("llvm.sqrt")
extern vsqrt<T>(x: T) T with [];

@llvm_intrinsic_infer("llvm.fptosi.sat", return, x)
extern sat<R, T>(x: T) R with [];

@llvm_intrinsic_infer("llvm.vector.reduce.fadd", v)
extern reduce_fadd<T, const N: u64>(acc: T, v: simd<T, N>) T with [];

@llvm_intrinsic_infer("llvm.ctpop")
extern popcount<T>(x: T) T with [];

@llvm_intrinsic_exact("llvm.fma.f64")
extern fma64(a: f64, b: f64, c: f64) f64 with [];

@llvm_intrinsic_infer("llvm.prefetch", p)
extern prefetch(p: *f32, rw: i32, locality: i32, cache: i32) void with [];

@llvm_intrinsic_infer("llvm.cos")
extern never_called<T>(x: T) T with [];
"#;

fn bindings_program(body: &str) -> String {
    format!("{BINDINGS}\nproc main() i32 {{\n{body}\n    return 0;\n}}\n")
}

#[test]
fn infer_suffixes_follow_each_concrete_instance() {
    let dir = tempfile::tempdir().unwrap();
    let ir = compile_ir(dir.path(), &bindings_program("
    let a: f32 = vsqrt(4.0f32);
    let b: f64 = vsqrt(9.0f64);
    let v: simd<f32, 4> = vsqrt(__simd_splat::<f32, 4>(4.0f32));
    let w: simd<f64, 2> = vsqrt(__simd_splat::<f64, 2>(4.0f64));
    let c: u8 = popcount(255u8);
    let d: i64 = popcount(7i64);
    "), &[]);
    for (symbol, decl) in [
        ("llvm.sqrt.f32", "declare float @llvm.sqrt.f32(float)"),
        ("llvm.sqrt.f64", "declare double @llvm.sqrt.f64(double)"),
        ("llvm.sqrt.v4f32", "declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)"),
        ("llvm.sqrt.v2f64", "declare <2 x double> @llvm.sqrt.v2f64(<2 x double>)"),
        // signedness is not part of an LLVM type: `u8` is `i8`
        ("llvm.ctpop.i8", "declare i8 @llvm.ctpop.i8(i8)"),
        ("llvm.ctpop.i64", "declare i64 @llvm.ctpop.i64(i64)"),
    ] {
        assert_eq!(declares(&ir, symbol), [decl], "{symbol}:\n{ir}");
        assert_eq!(calls(&ir, symbol).len(), 1, "{symbol}:\n{ir}");
    }
    // the template name itself never reaches LLVM
    assert!(!ir.contains("@vsqrt") && !ir.contains("@popcount"), "{ir}");
}

#[test]
fn listed_sources_build_the_suffix_in_the_order_written() {
    let dir = tempfile::tempdir().unwrap();
    let ir = compile_ir(dir.path(), &bindings_program("
    let i: i16 = sat::<i16, f64>(3.9f64);
    let s: f32 = reduce_fadd::<f32, 4>(0.0f32, __simd_splat::<f32, 4>(1.0f32));
    let x: f32 = 1.0f32;
    prefetch(&x, 0, 3, 1);
    "), &[]);
    // `return, x`: the result type first, then the argument's
    assert_eq!(declares(&ir, "llvm.fptosi.sat.i16.f64"),
        ["declare i16 @llvm.fptosi.sat.i16.f64(double)"], "{ir}");
    // `v`: a parameter that is not the first, with a const-generic lane count
    assert_eq!(declares(&ir, "llvm.vector.reduce.fadd.v4f32"),
        ["declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)"], "{ir}");
    // a pointer is `p0`, and a void intrinsic names its source parameter
    assert_eq!(declares(&ir, "llvm.prefetch.p0"),
        ["declare void @llvm.prefetch.p0(ptr, i32, i32, i32)"], "{ir}");
}

#[test]
fn exact_symbol_is_used_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let ir = compile_ir(dir.path(), &bindings_program("
    let d: f64 = fma64(2.0, 3.0, 1.0);
    "), &[]);
    assert_eq!(declares(&ir, "llvm.fma.f64"),
        ["declare double @llvm.fma.f64(double, double, double)"], "{ir}");
    assert_eq!(calls(&ir, "llvm.fma.f64").len(), 1, "{ir}");
    assert!(!ir.contains("@fma64"), "{ir}");
}

#[test]
fn each_symbol_is_declared_once_and_only_when_called() {
    let dir = tempfile::tempdir().unwrap();
    let ir = compile_ir(dir.path(), &bindings_program("
    let a: f32 = vsqrt(4.0f32);
    let b: f32 = vsqrt(9.0f32);
    let c: f32 = vsqrt(a + b);
    // the prelude already declares this one, for `numerical_cast`
    let i: i32 = sat::<i32, f32>(3.9f32);
    "), &[]);
    assert_eq!(declares(&ir, "llvm.sqrt.f32").len(), 1, "{ir}");
    assert_eq!(calls(&ir, "llvm.sqrt.f32").len(), 3, "{ir}");
    // a second `declare` of the same intrinsic is a redefinition to LLVM
    assert_eq!(declares(&ir, "llvm.fptosi.sat.i32.f32").len(), 1, "{ir}");
    // an intrinsic extern nobody calls declares nothing
    assert!(!ir.contains("llvm.cos"), "{ir}");
}

#[test]
fn float_intrinsic_calls_carry_the_callers_fastmath_flags() {
    let dir = tempfile::tempdir().unwrap();
    let source = format!("{BINDINGS}
@fastmath(reassoc, nsz, afn)
proc flagged(v: simd<f32, 4>) f32 {{ return reduce_fadd::<f32, 4>(0.0f32, vsqrt(v)); }}

@fastmath(fast)
proc count(x: i64) i64 {{ return popcount(x); }}

proc plain(v: simd<f32, 4>) f32 {{ return reduce_fadd::<f32, 4>(0.0f32, vsqrt(v)); }}

proc main() i32 {{
    let v: simd<f32, 4> = __simd_splat::<f32, 4>(4.0f32);
    let a: f32 = flagged(v) + plain(v);
    let b: i64 = count(7i64);
    return 0;
}}
");
    let ir = compile_ir(dir.path(), &source, &[]);
    let flagged = function_body(&ir, "$flagged");
    assert!(flagged.contains("call nsz afn reassoc <4 x float> @llvm.sqrt.v4f32("), "{flagged}");
    assert!(flagged.contains("call nsz afn reassoc float @llvm.vector.reduce.fadd.v4f32("), "{flagged}");
    // LLVM rejects fast-math flags on a call that does not return a float
    let count = function_body(&ir, "$count");
    assert!(count.contains("call i64 @llvm.ctpop.i64("), "{count}");
    let plain = function_body(&ir, "$plain");
    assert!(plain.contains("call <4 x float> @llvm.sqrt.v4f32("), "{plain}");
    assert!(plain.contains("call float @llvm.vector.reduce.fadd.v4f32("), "{plain}");
}

/// The reason this exists: libm's `sqrtf` may set `errno`, so a loop calling
/// it stays scalar. The intrinsic has no side effects, and LLVM vectorizes it.
#[test]
fn a_sqrt_loop_vectorizes_through_the_intrinsic() {
    let dir = tempfile::tempdir().unwrap();
    let source = format!("{BINDINGS}
@inline(never)
proc roots(buf: *f32, n: u64) void {{
    let i: u64 = 0u64;
    while (i < n) {{
        buf[i] = vsqrt(buf[i]);
        i = i + 1u64;
    }}
}}

proc main() i32 {{
    let buf: [f32; 64] = [4.0f32; 64];
    roots(&buf[0], 64u64);
    return 0;
}}
");
    compile_ir(dir.path(), &source, &["--emit-optimized-ir"]);
    let opt = std::fs::read_to_string(dir.path().join("app.opt.ll")).expect("the .opt.ll");
    assert!(opt.contains("@llvm.sqrt.v4f32") || opt.contains("@llvm.sqrt.v8f32"),
        "the loop should call the vector intrinsic:\n{opt}");
}

/// The standard library is the intended user: a binding in one package, called
/// through a typed wrapper from another, must still mint the right symbol.
#[test]
fn a_binding_in_a_library_is_instantiated_by_its_consumer() {
    let lib = tempfile::tempdir().unwrap();
    std::fs::write(lib.path().join("lib.hv"), r#"
@llvm_intrinsic_infer("llvm.sqrt")
extern llvm_sqrt<T>(x: T) T with [];

pub proc root<T>(x: T) T without Alloc { return llvm_sqrt(x); }
"#).unwrap();
    let meta = lib.path().join("dspmath.hvmeta");
    let o = common::havenc_cmd()
        .current_dir(lib.path())
        .args(["lib.hv", "--package-name", "dspmath", "--lib", "-o"]).arg(&meta)
        .output().expect("failed to spawn havenc");
    assert!(o.status.success(), "lib build failed:\n{}", String::from_utf8_lossy(&o.stderr));

    let app = tempfile::tempdir().unwrap();
    std::fs::write(app.path().join("main.hv"), r#"
import dspmath { root }

@export
proc main() i32 {
    let a: f32 = root(16.0f32);
    let b: f64 = root(81.0f64);
    if (a == 4.0f32 && b == 9.0f64) { return 0; }
    return 1;
}
"#).unwrap();
    let out = app.path().join("app");
    let o = common::havenc_cmd()
        .current_dir(app.path())
        .args(["main.hv", "--package-name", "app", "--emit-ir", "-o"]).arg(&out)
        .arg("--dep").arg(format!("dspmath={}", meta.display()))
        .output().expect("failed to spawn havenc");
    assert!(o.status.success(), "app build failed:\n{}", String::from_utf8_lossy(&o.stderr));
    let ir = std::fs::read_to_string(out.with_extension("ll")).unwrap();
    assert_eq!(declares(&ir, "llvm.sqrt.f32").len(), 1, "{ir}");
    assert_eq!(declares(&ir, "llvm.sqrt.f64").len(), 1, "{ir}");

    let run = std::process::Command::new(&out).output().expect("failed to run app");
    assert!(run.status.success(), "wrong roots: exit {:?}", run.status.code());
}
