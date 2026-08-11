//! Producer tests for the `.hvmeta` native-library artifact.
//!
//! Each test lays out a small lib project in a fresh temp dir, runs the real
//! `havenc --lib` on it, and inspects the emitted artifact through the
//! `haven_meta` reader — so these exercise the whole producer path, not the
//! serializer in isolation.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

const HAVENC: &str = env!("CARGO_BIN_EXE_havenc");

/// Write `(relative path, contents)` files under `root`, creating parent dirs.
fn scaffold(root: &Path, files: &[(&str, &str)]) {
    for (rel, contents) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
    }
}

/// Run `havenc <entry> --package-name <pkg> --lib -o <out>` from `cwd`.
fn build_lib(cwd: &Path, entry: &str, pkg: &str, out: &Path) -> std::process::Output {
    build_lib_with_deps(cwd, entry, pkg, &[], out)
}

/// As [`build_lib`], with one `--dep name=path` per entry of `deps` — a library
/// that is itself built against other libraries.
fn build_lib_with_deps(cwd: &Path, entry: &str, pkg: &str, deps: &[&str], out: &Path)
    -> std::process::Output
{
    let mut cmd = Command::new(HAVENC);
    cmd.current_dir(cwd)
        .arg(entry)
        .arg("--package-name").arg(pkg)
        .arg("--lib")
        .arg("-o").arg(out);
    for d in deps {
        cmd.arg("--dep").arg(d);
    }
    cmd.output().expect("failed to spawn havenc")
}

/// A representative multi-module lib: a root, a sibling module, a nested one, and
/// an `import std/...` (whose module must not travel in the artifact).
fn sample_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("src/lib.hv", "import std/math { sqrtf }\n\
                        import geo { Point }\n\
                        import dsp/osc { ramp }\n\
                        pub proc scale(p: Point, k: i32) i32 { return (p.x + p.y) * k + ramp(1); }\n"),
        ("src/geo.hv", "pub struct Point { x: i32, y: i32 }\n"),
        ("src/dsp/osc.hv", "pub proc ramp(n: i32) i32 { return n * 10; }\n"),
    ]
}

#[test]
fn emits_metadata_and_no_llvm() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &sample_files());
    let out = dir.path().join("libp");

    let res = build_lib(dir.path(), "src/lib.hv", "libp", &out);
    assert!(res.status.success(), "havenc --lib failed: {}", String::from_utf8_lossy(&res.stderr));

    assert!(out.with_extension("hvmeta").is_file(), "no .hvmeta produced");
    // a lib stops before codegen: none of these may appear.
    for ext in ["ll", "o", "exe", "a", "lib", "dll", "so"] {
        assert!(!out.with_extension(ext).exists(), "unexpected codegen artifact .{ext}");
    }
}

#[test]
fn embeds_c_sources_and_link_libs() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("src/lib.hv", "pub proc one() i32 { return 1; }\n"),
        ("c/rt.c", "void rt_thing(void) {}\n"),
        ("c/util.c", "int util_thing(void) { return 7; }\n"),
    ]);
    let out = dir.path().join("libp");

    let res = Command::new(HAVENC)
        .current_dir(dir.path())
        .arg("src/lib.hv")
        .arg("--package-name").arg("libp")
        .arg("--lib")
        .arg("--c-file").arg("c/rt.c")
        .arg("--c-file").arg("c/util.c")
        .arg("--link-lib").arg("m")
        .arg("-o").arg(&out)
        .output().expect("failed to spawn havenc");
    assert!(res.status.success(), "havenc --lib failed: {}", String::from_utf8_lossy(&res.stderr));

    let meta = haven_meta::read(&out.with_extension("hvmeta")).unwrap();
    // C rides by base name, source verbatim, directory layout stripped.
    let by_name: BTreeMap<_, _> = meta.native.iter().map(|n| (n.name.as_str(), n)).collect();
    assert_eq!(by_name.len(), 2);
    assert!(by_name["rt.c"].source.contains("rt_thing"));
    assert!(by_name["util.c"].source.contains("util_thing"));
    assert_eq!(meta.link_libs, vec!["m".to_string()]);

    // the native code is in the fingerprint: the header matches a fresh recompute
    // that includes it, and dropping it changes the digest.
    let recomputed = haven_meta::fingerprint(
        &meta.header.package_name, &meta.header.havenc_version, &meta.modules,
        &meta.native, &meta.link_libs);
    assert_eq!(meta.header.fingerprint, recomputed);
    let without_native = haven_meta::fingerprint(
        &meta.header.package_name, &meta.header.havenc_version, &meta.modules, &[], &[]);
    assert_ne!(meta.header.fingerprint, without_native);
}

#[test]
fn rejects_c_files_sharing_a_base_name() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("src/lib.hv", "pub proc one() i32 { return 1; }\n"),
        ("a/rt.c", "void a(void) {}\n"),
        ("b/rt.c", "void b(void) {}\n"),
    ]);
    let out = dir.path().join("libp");

    let res = Command::new(HAVENC)
        .current_dir(dir.path())
        .arg("src/lib.hv")
        .arg("--package-name").arg("libp")
        .arg("--lib")
        .arg("--c-file").arg("a/rt.c")
        .arg("--c-file").arg("b/rt.c")
        .arg("-o").arg(&out)
        .output().expect("failed to spawn havenc");
    assert!(!res.status.success(), "expected a base-name collision to fail the build");
    assert!(String::from_utf8_lossy(&res.stderr).contains("rt.c"));
    assert!(!out.with_extension("hvmeta").exists(), "no artifact should be written on error");
}

#[test]
fn round_trips_with_correct_modules_and_no_std() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &sample_files());
    let out = dir.path().join("libp");
    assert!(build_lib(dir.path(), "src/lib.hv", "libp", &out).status.success());

    let meta = haven_meta::read(&out.with_extension("hvmeta")).unwrap();
    assert_eq!(meta.header.format_version, haven_meta::FORMAT_VERSION);
    assert_eq!(meta.header.package_name, "libp");
    assert!(!meta.header.havenc_version.is_empty());

    // exactly the three own modules, keyed package-root-relative and forward-slashed.
    let by_key: BTreeMap<&str, &haven_meta::MetaModule> =
        meta.modules.iter().map(|m| (m.key.as_str(), m)).collect();
    let keys: Vec<&str> = by_key.keys().copied().collect();
    assert_eq!(keys, vec!["dsp/osc.hv", "geo.hv", "lib.hv"]);

    // no std/prelude module travels in the artifact (the lib's `import std/math`
    // line lives in lib.hv's *source*, but no std *module* is bundled).
    assert!(meta.modules.iter().all(|m| !m.key.starts_with("std/")));

    // is_root marks only the entry/root module.
    assert!(by_key["lib.hv"].is_root);
    assert!(!by_key["geo.hv"].is_root);
    assert!(!by_key["dsp/osc.hv"].is_root);

    // source is carried verbatim.
    assert!(by_key["geo.hv"].source.contains("struct Point"));
    assert!(by_key["lib.hv"].source.contains("proc scale"));

    // fingerprint in the header matches a fresh recompute over the modules.
    let recomputed = haven_meta::fingerprint(
        &meta.header.package_name, &meta.header.havenc_version, &meta.modules,
        &meta.native, &meta.link_libs);
    assert_eq!(meta.header.fingerprint, recomputed);
}

/// A library may itself be built against another library, and its artifact then
/// carries its *own* source only — not a copy of the dependency's.
///
/// The producer decides what belongs to this package by asking each module where
/// its source came from. It used to ask the module *key* instead, excluding
/// anything under `std/`; a dependency module is keyed `deep/lib.hv`, which is
/// neither `std/` nor under this package's root, so it fell through to the
/// "outside the package root" error and building such a library was impossible.
///
/// Shipping the dependency's source would be wrong even if it worked: the
/// consumer resolves `deep` from `deep.hvmeta` itself, so a second copy inside
/// `mid.hvmeta` is either redundant or — once versions exist — a conflict.
#[test]
fn library_with_a_dependency_ships_only_its_own_source() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        ("deep/lib.hv", "pub proc base(n: i32) i32 { return n + 1; }\n"),
        ("mid/lib.hv", "import deep { base }\n\
                        import util { twice }\n\
                        pub proc bump(n: i32) i32 { return twice(base(n)); }\n"),
        ("mid/util.hv", "pub proc twice(n: i32) i32 { return n * 2; }\n"),
    ]);

    let deep_out = dir.path().join("deep_art");
    assert!(build_lib(dir.path(), "deep/lib.hv", "deep", &deep_out).status.success());
    let deep_art = deep_out.with_extension("hvmeta");

    let mid_out = dir.path().join("mid_art");
    let dep = format!("deep={}", deep_art.display());
    let res = build_lib_with_deps(dir.path(), "mid/lib.hv", "mid", &[&dep], &mid_out);
    assert!(res.status.success(),
        "a library with a dependency must build: {}", String::from_utf8_lossy(&res.stderr));

    let meta = haven_meta::read(&mid_out.with_extension("hvmeta")).unwrap();
    assert_eq!(meta.header.package_name, "mid");

    // exactly mid's own two modules: no `deep/...` module and no std module.
    let keys: Vec<&str> = {
        let mut k: Vec<&str> = meta.modules.iter().map(|m| m.key.as_str()).collect();
        k.sort();
        k
    };
    assert_eq!(keys, vec!["lib.hv", "util.hv"], "artifact carries foreign modules");
    assert!(meta.modules.iter().all(|m| !m.source.contains("proc base")),
        "the dependency's source travelled inside the artifact");
}

#[test]
fn fingerprint_is_location_independent() {
    let files = sample_files();

    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    scaffold(a.path(), &files);
    scaffold(b.path(), &files);

    let out_a = a.path().join("libp");
    let out_b = b.path().join("libp");
    assert!(build_lib(a.path(), "src/lib.hv", "libp", &out_a).status.success());
    assert!(build_lib(b.path(), "src/lib.hv", "libp", &out_b).status.success());

    let ma = haven_meta::read(&out_a.with_extension("hvmeta")).unwrap();
    let mb = haven_meta::read(&out_b.with_extension("hvmeta")).unwrap();

    assert_eq!(ma.header.fingerprint, mb.header.fingerprint, "fingerprint differs by location");

    // the (key, source) sets are byte-identical regardless of where it was built.
    let pairs = |m: &haven_meta::HavenMeta| -> BTreeMap<String, String> {
        m.modules.iter().map(|x| (x.key.clone(), x.source.clone())).collect()
    };
    assert_eq!(pairs(&ma), pairs(&mb));

    // no absolute path from either checkout is baked into the artifact bytes.
    let bytes_a = std::fs::read(out_a.with_extension("hvmeta")).unwrap();
    for needle in [a.path().to_string_lossy().to_string(), "AppData".to_string()] {
        assert!(!contains(&bytes_a, needle.as_bytes()), "artifact leaks path fragment: {needle}");
    }
}

#[test]
fn fingerprint_ignores_module_load_order() {
    // Import order controls the order modules are *loaded* into the artifact's
    // `modules` list. Reordering the `import` lines themselves would change the
    // root module's *source bytes* (and so, correctly, the fingerprint), so the
    // real invariant is that hashing the same module set in any order yields one
    // fingerprint - which is exactly what lets import order not matter. The
    // fingerprint sorts by key internally; verify that directly.
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &sample_files());
    let out = dir.path().join("libp");
    assert!(build_lib(dir.path(), "src/lib.hv", "libp", &out).status.success());

    let meta = haven_meta::read(&out.with_extension("hvmeta")).unwrap();
    let mut reversed = meta.modules.clone();
    reversed.reverse();
    assert_eq!(
        haven_meta::fingerprint(&meta.header.package_name, &meta.header.havenc_version, &meta.modules, &meta.native, &meta.link_libs),
        haven_meta::fingerprint(&meta.header.package_name, &meta.header.havenc_version, &reversed, &meta.native, &meta.link_libs),
    );
}

#[test]
fn type_error_fails_and_writes_no_artifact() {
    let dir = tempfile::tempdir().unwrap();
    scaffold(dir.path(), &[
        // `p.x + true` is a type error in the exposed template.
        ("src/lib.hv", "pub proc bad() i32 { let x: i32 = 1; return x + true; }\n"),
    ]);
    let out = dir.path().join("libp");
    let res = build_lib(dir.path(), "src/lib.hv", "libp", &out);

    assert!(!res.status.success(), "a lib with a type error must fail the build");
    assert!(!out.with_extension("hvmeta").exists(), "no artifact may be written on type error");
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() { return false; }
    haystack.windows(needle.len()).any(|w| w == needle)
}
