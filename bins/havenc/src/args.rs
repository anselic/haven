use std::path::PathBuf;
use clap::{Parser, ValueEnum};

use haven_common::diag;

/// How `havenc` prints diagnostics. `human` is the ariadne pretty-printer;
/// `json` emits one machine-readable object per line (NDJSON) on stderr for the
/// LSP and the `haven` build orchestrator to consume.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum MessageFormat {
    Human,
    Json,
}

impl From<MessageFormat> for diag::Format {
    fn from(m: MessageFormat) -> Self {
        match m {
            MessageFormat::Human => diag::Format::Human,
            MessageFormat::Json => diag::Format::Json,
        }
    }
}

#[derive(Parser, Debug)]
pub struct Args {
    /// The main input source file to compile
    #[arg(required = true)]
    pub input: PathBuf,

    /// The output file path for the compiled binary
    /// Defaults to `a.out` if not specified.
    #[arg(short, long, value_name = "OUTPUT", default_value = "output")]
    pub output: PathBuf,

    /// The LLVM IR compiler to use (e.g., `clang`, `llc`, etc.)
    /// Defaults to `clang` if not specified.
    #[arg(short, long, value_name = "COMPILER", default_value = "clang")]
    pub compiler: String,

    /// LLVM IR compiler flags to pass to the compiler (e.g. `-O3 -Wall`, etc.)
    #[arg(short='F', long, value_name = "FLAGS", default_value = "-O3 -Wno-override-module")]
    pub compiler_flags: String,

    /// Compile as a shared dynamic library (.so / .dll / .dylib)
    #[arg(long, conflicts_with = "static_lib")]
    pub shared: bool,

    /// Compile as a static library (.a / .lib)
    #[arg(long, conflicts_with = "shared")]
    pub static_lib: bool,

    /// Keep the generated LLVM IR file instead of cleaning it up after compilation
    #[arg(long)]
    pub emit_ir: bool,

    /// Emit back the optimized LLVM IR from the LLVM IR compiler (.opt.ll)
    /// Will follow the optimization flags passed to the LLVM IR compiler
    #[arg(long)]
    pub emit_optimized_ir: bool,

    /// Emit assembly via the LLVM IR compiler (.s)
    #[arg(
        long,
        conflicts_with = "shared",
        conflicts_with = "static_lib"
    )]
    pub emit_asm: bool,

    /// Do not inject the implicit prelude (print/println/... become undefined
    /// unless declared manually). Useful for freestanding builds.
    #[arg(long)]
    pub no_prelude: bool,

    /// Diagnostic output format. `human` (default) is the pretty terminal
    /// renderer; `json` emits one NDJSON diagnostic per line on stderr for
    /// tooling (LSP, the `haven` build orchestrator) to parse.
    #[arg(long, value_enum, default_value_t = MessageFormat::Human)]
    pub message_format: MessageFormat,
}