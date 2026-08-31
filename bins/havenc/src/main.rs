use std::io::Write;
use clap::Parser;

// The compiler stages live in the workspace crates, in pipeline order:
//   haven_common - AST + diagnostics, shared by everything
//   haven_front  - lex/parse, module load + merge (imports, mangling, prelude)
//   haven_mid    - typecheck, safety-check, monomorphize, lower to MIL
//   haven_back   - ABI/layout, LLVM IR emission
use haven_common::{ast, diag};
use haven_common::defs::Origin;
use haven_front::module;
use haven_mid::{typecheck, mono, own, safecheck, mil};
use haven_back::llvm;

mod args;

fn main() {
    let args = args::Args::parse();
    let input = &args.input;

    // Pick the diagnostic format before any stage can emit one, so every
    // `diag::report*` call below - including those inside `load_and_merge` -
    // renders in the requested format.
    diag::set_format(args.message_format.into());

    // arena backing every `&'a str` in the AST (module sources, token streams,
    // and the synthetic mangled/prefixed names minted during module resolution
    // and monomorphization)
    // owned here so it outlives the whole compilation
    let arena = bumpalo::Bump::new();

    // Compiled-library dependencies: each `--dep name=path.hvmeta` is read and
    // validated up front, so a malformed or version-incompatible artifact fails
    // as a clean diagnostic before compilation rather than mid-resolution. The
    // parsed source travels into `load_and_merge`, which resolves an `import
    // name/...` against it exactly the way it resolves `std/...`.
    let mut deps = load_deps(&args.dep);

    // The compiler embeds no std of its own: it discovers the `std` library on
    // disk (see `discover_std`) and binds it exactly like a `--dep std=...` would,
    // so `import std/...` resolves to it and its `@!prelude` becomes the program's
    // prelude. Skipped when the user already bound their own `std`, or when the
    // package being compiled *is* std - a package cannot depend on itself, and
    // this is how std itself is built. `default_std` marks the binding as the
    // compiler's own fallback-priority one, so a user's prelude-bearing dep still
    // wins prelude discovery. Discovery failing is not fatal *yet*: the embedded
    // tree remains the fallback until it is removed.
    let self_pkg = args.package_name.clone().unwrap_or_else(|| {
        args.input.file_stem().map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "pkg".to_string())
    });
    let mut default_std: Option<&str> = None;
    if self_pkg != "std" && !deps.contains_key("std") {
        if let Some(std_meta) = discover_std() {
            deps.insert("std".to_string(), std_meta);
            default_std = Some("std");
        }
    }

    // where the prelude comes from. `--no-prelude` and `--prelude <name>` are
    // mutually exclusive at the CLI, so the three cases are disjoint.
    let prelude = match (&args.prelude, args.no_prelude) {
        (_, true) => module::PreludeSource::None,
        (Some(name), _) => module::PreludeSource::Package(name),
        // no flag: discover the prelude from `@!prelude` marks (a bound dependency
        // that advertises one, else the embedded stdlib).
        (None, false) => module::PreludeSource::Auto,
    };

    // load the entry file + every module it (transitively) imports, inject the
    // prelude unless disabled, and merge into one flat name-mangled program with
    // imports resolved away (see `crate::module`). `files` holds every loaded
    // module's path + source, indexed by the `FileId` its spans carry, so the
    // diagnostics below quote the span's owning module, not just the entry file.
    // `defs` owns every definition's identity: it produced the symbol names now
    // in `ast`, and carries the member table both typecheck passes use.
    let (mut ast, files, mut defs, impls, package_name) = match module::load_and_merge(input, args.package_name.as_deref(), prelude, &deps, default_std, &arena) {
        Ok(loaded) => loaded,
        Err(()) => std::process::exit(1),
    };

    // `main` is the program entry point, so the C runtime that calls it needs the
    // symbol to have external linkage. Inject `@export` automatically rather than
    // making every user write it by hand. Module merging leaves the entry `main`
    // unmangled (see `haven_front::module`), so the only function literally named
    // "main" is the entry one; a non-entry module's `main` was mangled away.
    for item in &mut ast {
        if let ast::TopLevelNode::Function { name: "main", attributes, .. } = &mut item.value {
            if !attributes.iter().any(|a| a.value.name == "export") {
                attributes.push(ast::Metadata::new(
                    ast::AttributeNode::new("export", None),
                    ast::Span::unknown(),
                ));
            }
        }
    }

    {
        let mut cx = typecheck::Context::new();
        let typecheck_errs = typecheck::typecheck_program(&mut cx, &ast, &impls, &defs);

        // check if there is no main function when compiling an executable. A
        // library of any kind (native shared/static, or a `.hvmeta` source lib)
        // has no `main`, so the requirement is lifted for all three.
        if !args.shared && !args.static_lib && !args.lib {
            let main_fn = ast.iter().find_map(|item| {
                if let ast::TopLevelNode::Function { name, .. } = item.value {
                    if name == "main" {
                        Some(())
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            if main_fn.is_none() {
                diag::report_plain(
                    "Error",
                    "No 'main' function defined. If you intended to compile a \
                     shared/static library, use the --shared or --static-lib \
                     flag. Otherwise, add a 'main' function to your program.",
                );
                std::process::exit(1);
            }
        }

        if !typecheck_errs.is_empty() {
            typecheck_errs.iter()
                .for_each(|e| diag::report_error("Typecheck error", e, &files));
            std::process::exit(1);
        } else if args.lib {
            // A native Haven library: emit a `.hvmeta` source-blob artifact and
            // stop. The pre-mono typecheck above already ran as validation - it
            // checks the generic *templates* a lib exposes, so a lib author's type
            // error surfaces here rather than in a consumer's build. Everything
            // past this point (mono, MIL, LLVM, and the post-mono ownership/alloc
            // checks) operates on concrete instances a lib does not have; those
            // are deferred to the leaf, where instantiation happens.
            write_lib_metadata(input, &package_name, &defs, &files, &args.c_file, &args.link_lib, &args.output);
            return;
        } else {
            // expand generics into concrete instances, then re-typecheck the
            // now fully-concrete program so node_types is populated for the
            // fresh instantiations
            // TODO: this re-checks the *whole* program (prelude, std, every
            // concrete fn) from scratch and throws away the first `cx`, when
            // only the new instances actually need checking
            // mono extends `defs` with one entry per instance it mints, which is
            // what lets the second typecheck pass below match a match-arm pattern
            // (which names the template) against a scrutinee whose type names the
            // instance.
            let mut mono_ast = mono::monomorphize(&ast, &mut defs, &arena, &cx.node_types, &cx.inferred_type_args, &impls).unwrap_or_else(|e| {
                diag::report_error("Monomorphization error", &e, &files);
                std::process::exit(1);
            });
            // mono dropped all trait nodes and substituted every bounded type
            // param, so the concrete program has no impls left to check.
            // A method of a *generic* `extend` is a template, and mono rewrote
            // every call to one into a direct call on the instance it minted, so
            // what is left in the member table for this pass to resolve is the
            // concrete impls - plus the per-instance destructors mono added,
            // which the ownership pass below reads.
            let mut cx = typecheck::Context::new();
            let mono_errs = typecheck::typecheck_program(&mut cx, &mono_ast, &[], &defs);
            if !mono_errs.is_empty() {
                mono_errs.iter()
                    .for_each(|e| diag::report_error("Typecheck error", e, &files));
                std::process::exit(1);
            }

            // ownership: reject use-after-move and insert the `delete` calls
            // that destroy every owner exactly once. Runs on the concrete
            // program, where every type's `Copy`-ness is decidable, and before
            // the alloc check, so a `@alloc(false)` function is judged on the
            // destructors it actually ends up calling.
            // the lang item comes from `defs`, which outlives mono - so unlike
            // the trait *declarations* mono drops, it needs no capturing here.
            own::ownership_check(&mut mono_ast, &mut cx, defs.lang().delete, &impls)
                .unwrap_or_else(|errs| {
                    for err in &errs {
                        diag::report_error("Ownership error", err, &files);
                    }
                    std::process::exit(1);
                });

            safecheck::alloc_check_program(&mono_ast, &defs, &cx.node_types).unwrap_or_else(|errs| {
                for err in &errs {
                    diag::report_error("Check error", err, &files);
                }
                std::process::exit(1);
            });

            let mil = mil::lower(&mono_ast, &cx, &defs, &arena);
            let llvm_ir = llvm::emit(mil);

            let llvm_ir_output_path = args.output.with_extension("ll");

            // If the output is in a directory (that may or may not exist), create the directory first
            if let Some(parent_dir) = llvm_ir_output_path.parent() {
                std::fs::create_dir_all(parent_dir).expect("Failed to create output directory");
            }

            std::fs::write(&llvm_ir_output_path, llvm_ir)
                .expect("Failed to write LLVM IR to file");

            if args.emit_optimized_ir {
                let optimized_ir_output_path = args.output.with_extension("opt.ll");

                let status = std::process::Command::new(&args.compiler)
                    .arg(&llvm_ir_output_path)
                    // .arg(temp_runtime.path())
                    .arg("-S")
                    .arg("-emit-llvm")
                    .args(&args.compiler_flags.split_whitespace().collect::<Vec<_>>())
                    .arg("-o")
                    .arg(&optimized_ir_output_path)
                    .status()
                    .expect("Failed to execute compiler for optimized IR");

                if !status.success() {
                    eprintln!("Compiler exited with non-zero status when generating optimized IR: {}", status);
                    std::process::exit(1);
                }
            }

            if args.emit_asm {
                let asm_output_path = args.output.with_extension("s");

                let status = std::process::Command::new(&args.compiler)
                    .arg(&llvm_ir_output_path)
                    .arg("-S")
                    .args(&args.compiler_flags.split_whitespace().collect::<Vec<_>>())
                    .arg("-o")
                    .arg(&asm_output_path)
                    .status()
                    .expect("Failed to execute compiler for assembly");

                if !status.success() {
                    eprintln!("Compiler exited with non-zero status when generating assembly: {}", status);
                    std::process::exit(1);
                }

                std::process::exit(0);
            }

            // Turn each dependency's shipped C into machine code the leaf can
            // link. A `.hvmeta` carries C *source* (it must stay
            // target-independent), so the leaf is where it becomes an object -
            // compiled here at -O3 to match the embedded runtime, which
            // `bins/havenc/build.rs` builds with the `cc` crate at -O3, so a
            // dependency's runtime does not silently regress against the
            // compiler's own. Objects rather than a throwaway shared lib, so they
            // drop into an executable, a shared lib, or a static archive the same
            // way. Deps are visited in name order so the link line is
            // reproducible; the temp `.c`/`.o` files are held in `native_temps`
            // until the link command below has run and read them.
            let mut native_temps: Vec<tempfile::NamedTempFile> = Vec::new();
            let mut native_objs: Vec<std::path::PathBuf> = Vec::new();
            let mut link_libs: Vec<String> = Vec::new();
            // Only deps whose modules were actually loaded contribute native C and
            // link libs. A dep that is bound but never imported (and is not the
            // prelude) pulls nothing in - so the discovered std costs a freestanding
            // `--no-prelude` build nothing, and an unused explicit `--dep` no longer
            // drags its C onto the link line. Dep modules key as `<depname>/<rel>`
            // (see `dep_key`), so the leading path segment recovers the dep name.
            let loaded_deps: std::collections::HashSet<&str> = defs.modules().iter()
                .filter(|m| m.origin == Origin::Dep)
                .filter_map(|m| m.key.split('/').next())
                .collect();
            let mut dep_names: Vec<&String> = deps.keys().collect();
            dep_names.sort();
            for name in &dep_names {
                if !loaded_deps.contains(name.as_str()) { continue; }
                let meta = &deps[*name];
                for lib in &meta.link_libs {
                    if !link_libs.iter().any(|l| l == lib) {
                        link_libs.push(lib.clone());
                    }
                }
                for src in &meta.native {
                    let mut c_temp = tempfile::Builder::new()
                        .suffix(".c")
                        .tempfile()
                        .expect("Failed to create temp C file");
                    c_temp.write_all(src.source.as_bytes())
                        .expect("Failed to write dependency C source to temp file");
                    let o_temp = tempfile::Builder::new()
                        .suffix(".o")
                        .tempfile()
                        .expect("Failed to create temp object file");
                    let status = std::process::Command::new(&args.compiler)
                        .arg("-c")
                        .arg("-O3")
                        .arg(c_temp.path())
                        .arg("-o")
                        .arg(o_temp.path())
                        .status()
                        .expect("Failed to execute compiler for dependency C source");
                    if !status.success() {
                        eprintln!(
                            "Compiler exited with non-zero status compiling '{}' from dependency '{}': {}",
                            src.name, name, status);
                        std::process::exit(1);
                    }
                    native_objs.push(o_temp.path().to_path_buf());
                    native_temps.push(c_temp);
                    native_temps.push(o_temp);
                }
            }

            // The leaf's own native code, from its `[[c]]`/`--c-file`/`--link-lib`
            // - a binary that ships C glue or links a system library (raylib, SDL,
            // ...). A `--lib` build never reaches here: it stores its `--c-file` in
            // the `.hvmeta` for its consumers' leaves to compile instead (see
            // `write_lib_metadata`). Own C sources are real files on disk, so they
            // compile straight to an object (no temp `.c` needed), at the same -O3
            // as dependency C. The objects join `native_objs`; the lib names join
            // `link_libs` after the deps' so a lib both a dep and the leaf ask for
            // collapses to a single `-l`.
            for path in &args.c_file {
                let o_temp = tempfile::Builder::new()
                    .suffix(".o")
                    .tempfile()
                    .expect("Failed to create temp object file");
                let status = std::process::Command::new(&args.compiler)
                    .arg("-c")
                    .arg("-O3")
                    .arg(path)
                    .arg("-o")
                    .arg(o_temp.path())
                    .status()
                    .expect("Failed to execute compiler for --c-file source");
                if !status.success() {
                    eprintln!(
                        "Compiler exited with non-zero status compiling '{}': {}",
                        path.display(), status);
                    std::process::exit(1);
                }
                native_objs.push(o_temp.path().to_path_buf());
                native_temps.push(o_temp);
            }
            for lib in &args.link_lib {
                if !link_libs.iter().any(|l| l == lib) {
                    link_libs.push(lib.clone());
                }
            }

            // `-l` flags for every library the deps and the leaf declared (e.g.
            // std's `libs = ["m"]`, or a binary's `libs = ["raylib"]`). Owned here
            // so they outlive the borrowed `compiler_args` below.
            //
            // `m` is dropped on Windows, for the same reason the hardcoded `-lm`
            // below is: libm is part of the CRT under both MSVC and MinGW, and
            // there is no `m.lib` on disk to find - `lld-link` fails the entire
            // link over a library that was never needed. A manifest says what its
            // C *needs* (`libs = ["m"]` is true of std's runtime everywhere);
            // translating that into flags for the host is this side's job.
            let lib_flags: Vec<String> = link_libs
                .iter()
                .filter(|l| !(cfg!(target_os = "windows") && *l == "m"))
                .map(|l| format!("-l{}", l))
                .collect();

            // The compiler embeds no runtime of its own. Every program's C runtime
            // rides in a dependency (std ships `rt.c`/`env.c`/..., compiled to
            // `native_objs` above), so the link line carries only those objects. A
            // freestanding build with no runtime-bearing dep links none - which is
            // what `--no-prelude` means.
            let mut compiler_args = vec![llvm_ir_output_path.to_str().unwrap()];
            compiler_args.extend(args.compiler_flags.split_whitespace());
            // dependency C objects, ahead of the `-l` flags they may reference
            compiler_args.extend(native_objs.iter().map(|p| p.to_str().unwrap()));

            // add -lm on non-Windows platforms because math library is
            // in the CRT for MSVC and MinGW. Skipped when a dependency already
            // declares `m`, which is where this hardcode goes to die: once the
            // runtime is std's and std ships `libs = ["m"]`, the compiler stops
            // asserting libm on its own.
            if !cfg!(target_os = "windows") && !link_libs.iter().any(|l| l == "m") {
                compiler_args.push("-lm");
            }
            // libraries the dependencies asked for, after every object
            compiler_args.extend(lib_flags.iter().map(|s| s.as_str()));

            let status = if args.shared {
                let shared_output_path = if cfg!(target_os = "windows") {
                    args.output.with_extension("dll")
                } else if cfg!(target_os = "macos") {
                    args.output.with_extension("dylib")
                } else {
                    // default to .so for linux & other platforms
                    args.output.with_extension("so")
                };

                std::process::Command::new(&args.compiler)
                    .args(&compiler_args)
                    .arg("-shared")
                    .arg("-o")
                    .arg(&shared_output_path)
                    .status()
                    .expect("Failed to execute compiler for shared library")
            } else if args.static_lib {
                // clang won't archive for us, so compile the IR to a single
                // object first, then bundle it with the dependency objects into one
                // static library the host can link against.
                let obj_path = args.output.with_extension("o");
                let obj_status = std::process::Command::new(&args.compiler)
                    .arg(&llvm_ir_output_path)
                    .arg("-c")
                    .args(&args.compiler_flags.split_whitespace().collect::<Vec<_>>())
                    .arg("-o")
                    .arg(&obj_path)
                    .status()
                    .expect("Failed to execute compiler for object file");

                if !obj_status.success() {
                    eprintln!("Compiler exited with non-zero status when generating object file: {}", obj_status);
                    std::process::exit(1);
                }

                // On Windows the conventional static lib is a `.lib` produced by
                // llvm-lib (MSVC-style archive); elsewhere it's a `.a` from
                // llvm-ar. Both understand our object + the dependency objects.
                let (archiver, lib_ext) = if cfg!(target_os = "windows") {
                    ("llvm-lib", "lib")
                } else {
                    ("llvm-ar", "a")
                };
                let lib_path = args.output.with_extension(lib_ext);

                let mut archive_cmd = std::process::Command::new(archiver);
                if cfg!(target_os = "windows") {
                    // llvm-lib: /OUT:foo.lib foo.o dep0.o ...
                    archive_cmd
                        .arg(format!("/OUT:{}", lib_path.display()))
                        .arg(&obj_path)
                        .args(&native_objs);
                } else {
                    // llvm-ar: crs foo.a foo.o dep0.o ...
                    archive_cmd
                        .arg("crs")
                        .arg(&lib_path)
                        .arg(&obj_path)
                        .args(&native_objs);
                }

                let archive_status = archive_cmd
                    .status()
                    .unwrap_or_else(|e| panic!("Failed to execute archiver '{}': {}", archiver, e));

                if !obj_path.exists() || std::fs::remove_file(&obj_path).is_err() {
                    // best-effort cleanup of the intermediate object
                }

                archive_status
            } else {
                let output = if cfg!(target_os = "windows") {
                    args.output.with_extension("exe")
                } else {
                    args.output
                };

                std::process::Command::new(&args.compiler)
                    .args(&compiler_args)
                    .arg("-o")
                    .arg(&output)
                    .status()
                    .expect("Failed to execute compiler for executable")
            };

            if !status.success() {
                eprintln!("Compiler exited with non-zero status: {}", status);
                std::process::exit(1);
            }

            if !args.emit_ir {
                std::fs::remove_file(llvm_ir_output_path).expect("Failed to remove LLVM IR file");
            }
        }
    }
}

/// Find the default `std` library on disk, the compiler having none embedded.
/// Precedence:
///   1. `$HAVEN_STD` - an explicit path (or empty, meaning "no std, don't
///      discover"; the escape hatch for freestanding builds and the test harness).
///      Set-but-unusable is a hard error: the operator meant to supply std.
///   2. a path relative to the `havenc` binary: `<dir>/std.hvmeta`, then
///      `<dir>/../lib/haven/std.hvmeta` for an installed `bin`/`lib` layout.
/// Returns `None` when nothing is set and no sibling artifact exists, letting the
/// caller fall back to the (soon-to-be-removed) embedded tree. A found-but-wrong
/// artifact is a hard error rather than a silent fallback.
fn discover_std() -> Option<haven_meta::HavenMeta> {
    if let Some(val) = std::env::var_os("HAVEN_STD") {
        if val.is_empty() {
            return None; // explicit opt-out
        }
        return Some(load_std_from(std::path::Path::new(&val), "$HAVEN_STD"));
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    for rel in ["std.hvmeta", "../lib/haven/std.hvmeta"] {
        let cand = dir.join(rel);
        if cand.is_file() {
            return Some(load_std_from(&cand, "the compiler's sysroot"));
        }
    }
    None
}

/// Read a `std.hvmeta` from `path` and insist it really is package `std`. `whence`
/// names where the path came from, for the diagnostic. Exits the process on any
/// failure - a std the operator pointed at that cannot be used is fatal, never a
/// quiet fallback.
fn load_std_from(path: &std::path::Path, whence: &str) -> haven_meta::HavenMeta {
    let meta = match haven_meta::read(path) {
        Ok(m) => m,
        Err(e) => {
            diag::report_plain("Error", &format!(
                "cannot load the std library from '{}' ({}): {}", path.display(), whence, e));
            std::process::exit(1);
        }
    };
    if meta.header.package_name != "std" {
        diag::report_plain("Error", &format!(
            "the artifact at '{}' ({}) is package '{}', not 'std'; it cannot serve \
             as the standard library", path.display(), whence, meta.header.package_name));
        std::process::exit(1);
    }
    meta
}

/// Parse and load every `--dep name=path.hvmeta` into a `name -> artifact` map.
///
/// Each artifact is read and version-checked here so a bad dependency surfaces as
/// one clear diagnostic before any compilation happens. The bound name must match
/// the artifact's own package name: symbols are slugged under the name the import
/// path uses, so binding `foo`'s artifact as `bar=` would resolve `import bar/...`
/// to modules slugged `foo.*` and silently disagree with the library's own build.
/// Exits the process on any malformed spec, unreadable/incompatible artifact, or
/// duplicate name.
fn load_deps(specs: &[String]) -> std::collections::HashMap<String, haven_meta::HavenMeta> {
    let mut deps = std::collections::HashMap::new();
    for spec in specs {
        let (name, path) = match spec.split_once('=') {
            Some((n, p)) if !n.is_empty() && !p.is_empty() => (n, p),
            _ => {
                diag::report_plain("Error", &format!(
                    "invalid --dep '{}': expected NAME=PATH, e.g. --dep foo=foo.hvmeta", spec));
                std::process::exit(1);
            }
        };
        let meta = match haven_meta::read(std::path::Path::new(path)) {
            Ok(m) => m,
            Err(e) => {
                diag::report_plain("Error", &format!(
                    "cannot load dependency '{}' from '{}': {}", name, path, e));
                std::process::exit(1);
            }
        };
        if meta.header.package_name != name {
            diag::report_plain("Error", &format!(
                "dependency bound as '{}' is actually package '{}'; bind it as \
                 `--dep {}={}` so `import {}/...` names it", name,
                meta.header.package_name, meta.header.package_name, path,
                meta.header.package_name));
            std::process::exit(1);
        }
        if deps.insert(name.to_string(), meta).is_some() {
            diag::report_plain("Error", &format!(
                "dependency '{}' is specified more than once", name));
            std::process::exit(1);
        }
    }
    deps
}

/// Assemble and write a native library's `.hvmeta` artifact: a header, every one
/// of the package's OWN source modules, and its native code (the `--c-file`
/// sources and `--link-lib` names). `std`/prelude are excluded - they are embedded
/// in every `havenc`, so a consumer re-resolves `import std/...` against its own
/// copy. Module keys are made package-root-relative and forward-slashed, so the
/// artifact carries no absolute path and fingerprints identically from any checkout
/// location; the C files are carried by base name only, for the same reason. Exits
/// the process on any error.
fn write_lib_metadata(
    entry: &std::path::Path,
    package_name: &str,
    defs: &haven_common::defs::Defs<'_>,
    files: &haven_common::diag::Files<'_>,
    c_files: &[std::path::PathBuf],
    link_libs: &[String],
    output: &std::path::Path,
) {
    // the package root is the entry file's directory; module keys are the
    // canonical absolute paths `load_and_merge` recorded, so relativize against
    // the same canonical root to strip the checkout location back off.
    let root = std::fs::canonicalize(entry).ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    let mut modules = Vec::new();
    for m in defs.modules() {
        // a library artifact carries this package's own source and nothing else.
        // Everything else `load_and_merge` pulled in is already available to the
        // consumer from the same place this build got it - the compiler's
        // embedded std, or the dependency's own `.hvmeta` - and shipping a second
        // copy inside this one would give the consumer two of it.
        //
        // Asked of `Origin` rather than of the key. Testing `!key.starts_with(
        // "std/")` looks equivalent and is not: a *dependency* module's key is
        // `foo/geo.hv`, which passes that test, is not under this package's root,
        // and so tripped the error below - making a library that has dependencies
        // of its own impossible to build.
        if m.origin != Origin::Own { continue; }
        let rel = match root.as_deref()
            .and_then(|r| std::path::Path::new(&m.key).strip_prefix(r).ok())
        {
            Some(rel) => rel.to_string_lossy().replace('\\', "/"),
            // unreachable in practice - `Defs::add_module` already refuses a
            // module of this package that resolves outside its root, since it
            // has no location-independent slug to give it. Kept as the local
            // statement of the same invariant, this being the point where a
            // location would otherwise be written into a shipped artifact.
            None => {
                diag::report_plain("Error", &format!(
                    "library module '{}' is outside the package root; cannot record \
                     a location-independent key for it", m.key));
                std::process::exit(1);
            }
        };
        let source = files.src(m.file).unwrap_or_default().to_string();
        modules.push(haven_meta::MetaModule { key: rel, source, is_root: m.is_entry });
    }

    // read each `--c-file` and carry it by base name. The source travels verbatim;
    // a consumer writes it back out and compiles it for its own target (see the
    // leaf link step). Two files sharing a base name would shadow each other at the
    // consumer, so reject it here rather than silently ship one.
    let mut native = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
    for path in c_files {
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => {
                diag::report_plain("Error", &format!(
                    "C source '{}' has no usable file name", path.display()));
                std::process::exit(1);
            }
        };
        if !seen_names.insert(name.clone()) {
            diag::report_plain("Error", &format!(
                "two --c-file arguments share the base name '{}'; the artifact carries \
                 C by name, so they would collide at a consumer", name));
            std::process::exit(1);
        }
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                diag::report_plain("Error", &format!(
                    "cannot read C source '{}': {}", path.display(), e));
                std::process::exit(1);
            }
        };
        native.push(haven_meta::NativeSource { name, source });
    }
    let link_libs = link_libs.to_vec();

    let havenc_version = env!("CARGO_PKG_VERSION").to_string();
    let fingerprint = haven_meta::fingerprint(package_name, &havenc_version, &modules, &native, &link_libs);
    let meta = haven_meta::HavenMeta {
        header: haven_meta::Header {
            format_version: haven_meta::FORMAT_VERSION,
            havenc_version,
            package_name: package_name.to_string(),
            fingerprint,
        },
        modules,
        native,
        link_libs,
    };

    let out_path = output.with_extension("hvmeta");
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("Failed to create output directory");
        }
    }
    if let Err(e) = haven_meta::write(&out_path, &meta) {
        diag::report_plain("Error", &format!(
            "cannot write library metadata '{}': {}", out_path.display(), e));
        std::process::exit(1);
    }
}
