use std::io::Write;
use clap::Parser;

// The compiler stages live in the workspace crates, in pipeline order:
//   haven_common - AST + diagnostics, shared by everything
//   haven_front  - lex/parse, module load + merge (imports, mangling, prelude)
//   haven_mid    - typecheck, safety-check, monomorphize, lower to MIL
//   haven_back   - ABI/layout, LLVM IR emission
use haven_common::{ast, diag};
use haven_front::module;
use haven_mid::{typecheck, mono, own, safecheck, mil};
use haven_back::llvm;

mod args;

const RUNTIME_ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libruntime.a"));

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

    // load the entry file + every module it (transitively) imports, inject the
    // prelude unless disabled, merge into one flat name-mangled program with all
    // imports resolved away. see `crate::module`
    // the prelude comes from the embedded std tree under its own `std/prelude`
    // key, so an explicit `import std/prelude` reuses it rather than loading a
    // second copy.
    // `files` holds every loaded module's path + source, indexed by the `FileId`
    // its spans carry, so diagnostics below quote the span's owning module - not
    // just the entry file.
    // `defs` owns every top-level definition's identity: it produced the symbol
    // names now in `ast`, and it carries the member table both typecheck passes
    // use to resolve method calls.
    let (mut ast, files, mut defs, impls) = match module::load_and_merge(input, args.package_name.as_deref(), !args.no_prelude, &arena) {
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

        // check if there is no main function when compiling an executable
        if !args.shared && !args.static_lib {
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
            // the `Delete` lang item, captured before mono drops every trait
            // node: the post-mono pass below has no trait declarations left to
            // find it in.
            let delete_trait = cx.delete_trait;
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
            own::ownership_check(&mut mono_ast, &mut cx, delete_trait, &impls)
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

            // Dump the embedded runtime archive into a temporary file
            let mut temp_runtime = tempfile::NamedTempFile::new().expect("Failed to create temp file");
            temp_runtime.write_all(RUNTIME_ARCHIVE).expect("Failed to write runtime archive to temp file");

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

            let mut compiler_args = vec![
                llvm_ir_output_path.to_str().unwrap(),
                temp_runtime.path().to_str().unwrap(),
            ];
            compiler_args.extend(args.compiler_flags.split_whitespace());

            // add -lm on non-Windows platforms because math library is
            // in the CRT for MSVC and MinGW
            if !cfg!(target_os = "windows") {
                compiler_args.push("-lm");
            }

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
                // object first, then bundle it with the runtime archive into one
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
                // llvm-ar. Both understand our object + the runtime archive's
                // members.
                let (archiver, lib_ext) = if cfg!(target_os = "windows") {
                    ("llvm-lib", "lib")
                } else {
                    ("llvm-ar", "a")
                };
                let lib_path = args.output.with_extension(lib_ext);

                let mut archive_cmd = std::process::Command::new(archiver);
                if cfg!(target_os = "windows") {
                    // llvm-lib: /OUT:foo.lib foo.o libruntime.a
                    archive_cmd
                        .arg(format!("/OUT:{}", lib_path.display()))
                        .arg(&obj_path)
                        .arg(temp_runtime.path());
                } else {
                    // llvm-ar: crs foo.a foo.o libruntime.a
                    archive_cmd
                        .arg("crs")
                        .arg(&lib_path)
                        .arg(&obj_path)
                        .arg(temp_runtime.path());
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
