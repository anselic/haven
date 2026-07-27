use std::io::Write;
use clap::Parser;

// The compiler stages live in the workspace crates, in pipeline order:
//   haven_common - AST + diagnostics, shared by everything
//   haven_front  - lex/parse, module load + merge (imports, mangling, prelude)
//   haven_mid    - typecheck, safety-check, monomorphize, lower to MIL
//   haven_back   - ABI/layout, LLVM IR emission
use haven_common::{ast, diag};
use haven_front::module;
use haven_mid::{typecheck, mono, safecheck, mil};
use haven_back::llvm;

mod args;

const RUNTIME_ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libruntime.a"));

/// Prelude source, embedded at compile time. Parsed and injected ahead of every
/// user program (see `main`). Backed by the C runtime in `crt/rt.c`.
const PRELUDE_SRC: &str = include_str!("../../../std/prelude.hv");

fn main() {
    let args = args::Args::parse();
    let input = &args.input;

    // arena backing every `&'a str` in the AST (module sources, token streams,
    // and the synthetic mangled/prefixed names minted during module resolution
    // and monomorphization)
    // owned here so it outlives the whole compilation
    let arena = bumpalo::Bump::new();

    // load the entry file + every module it (transitively) imports, inject the
    // prelude unless disabled, merge into one flat name-mangled program with all
    // imports resolved away. see `crate::module`
    // let prelude = if args.no_prelude { None } else { Some(PRELUDE_SRC) };
    // `files` holds every loaded module's path + source, indexed by the `FileId`
    // its spans carry, so diagnostics below quote the span's owning module - not
    // just the entry file.
    // `defs` owns every top-level definition's identity: it produced the symbol
    // names now in `ast`, and it carries the member table both typecheck passes
    // use to resolve method calls.
    let (ast, files, mut defs, impls) = match module::load_and_merge(input, Some(PRELUDE_SRC), &arena) {
        Ok(loaded) => loaded,
        Err(()) => std::process::exit(1),
    };

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
                eprintln!("Error: No 'main' function exported");
                eprintln!("If you intended to compile a shared/static library, use the --shared or --static-lib flag.");
                eprintln!("Otherwise, add a 'main' function to your program and an @export attribute to it.");
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
            let mono_ast = mono::monomorphize(&ast, &mut defs, &arena).unwrap_or_else(|e| {
                diag::report_error("Monomorphization error", &e, &files);
                std::process::exit(1);
            });
            // mono dropped all trait nodes and substituted every bounded type
            // param, so the concrete program has no impls left to check.
            // methods are never generic (`extend` on a generic type is rejected),
            // so monomorphization can't introduce or invalidate a member - the
            // same table applies to the concrete program.
            let mut cx = typecheck::Context::new();
            let mono_errs = typecheck::typecheck_program(&mut cx, &mono_ast, &[], &defs);
            if !mono_errs.is_empty() {
                mono_errs.iter()
                    .for_each(|e| diag::report_error("Typecheck error", e, &files));
                std::process::exit(1);
            }

            safecheck::alloc_check_program(&mono_ast, &defs).unwrap_or_else(|errs| {
                for err in &errs {
                    diag::report_error("Check error", err, &files);
                }
                std::process::exit(1);
            });

            let mil = mil::lower(&mono_ast, &cx, &defs, &arena);
            let llvm_ir = llvm::emit(mil);

            let llvm_ir_output_path = args.output.with_extension("ll");

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
                println!("Compiling as a shared library...");

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
                println!("Compiling as a static library...");

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
                println!("Compiling as an executable...");

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
