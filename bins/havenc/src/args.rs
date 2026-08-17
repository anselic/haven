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
    ///
    /// `allow_hyphen_values`, because every realistic value opens with `-O` and
    /// would otherwise be read as another option unless spelled `-F=<flags>`.
    #[arg(short='F', long, value_name = "FLAGS", allow_hyphen_values = true,
          default_value = "-O3 -Wno-override-module")]
    pub compiler_flags: String,

    /// Compile as a shared dynamic library (.so / .dll / .dylib)
    #[arg(long, conflicts_with = "static_lib")]
    pub shared: bool,

    /// Compile as a static library (.a / .lib)
    #[arg(long, conflicts_with = "shared")]
    pub static_lib: bool,

    /// Compile as a native Haven library: emit a `.hvmeta` source-blob artifact
    /// (for consumption by other Haven packages) instead of driving to LLVM. Stops
    /// after the validating typecheck; runs no mono/codegen and needs no `main`.
    #[arg(long, conflicts_with = "shared", conflicts_with = "static_lib",
          conflicts_with = "emit_asm")]
    pub lib: bool,

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

    /// Take the prelude from a package other than the embedded stdlib:
    /// `--prelude <name>`, where `<name>` is bound by a `--dep` (or is the
    /// package being compiled, which is how a stdlib is built). That package
    /// supplies both the implicitly imported items and the lang items
    /// (`@lang(delete)`), and is loaded whether or not the program imports it.
    /// Which of its modules is the prelude is the package's own business, said
    /// with `@!prelude` in its source.
    #[arg(long, value_name = "NAME", conflicts_with = "no_prelude")]
    pub prelude: Option<String>,

    /// The package name that anchors emitted symbol names
    /// (`<package>.<module>$<item>`). Defaults to the entry file's stem, so a
    /// bare `havenc foo.hv` names its package `foo`. The `haven` build tool
    /// forwards the manifest's `name` here.
    #[arg(long, value_name = "NAME")]
    pub package_name: Option<String>,

    /// Consume a compiled Haven library: `--dep <name>=<path.hvmeta>`. Repeatable.
    /// An `import <name>/<module>` in this program then resolves against the named
    /// artifact's source (produced by `havenc --lib`) instead of the filesystem,
    /// merged under package name `<name>` so it re-derives the library's own
    /// package-anchored symbols. v1: one explicit dep per flag, no version or
    /// lockfile resolution and no transitive deps.
    #[arg(long = "dep", value_name = "NAME=PATH")]
    pub dep: Vec<String>,

    /// A C source file this package ships (`--c-file <path>`, repeatable). Only
    /// meaningful with `--lib`: the source is embedded verbatim into the emitted
    /// `.hvmeta`, and a consumer compiles and links it when it builds a program.
    /// This is how a package's native code travels without the compiler embedding
    /// it. Ignored (with no error) for a non-`--lib` build, which links its own C.
    #[arg(long = "c-file", value_name = "PATH")]
    pub c_file: Vec<PathBuf>,

    /// A native library this package needs linked (`--link-lib <name>`, e.g.
    /// `--link-lib m` for `-lm`; repeatable). With `--lib` it is recorded in the
    /// `.hvmeta` so a consumer adds the `-l` flag transitively; the package need
    /// not know who links it.
    #[arg(long = "link-lib", value_name = "NAME")]
    pub link_lib: Vec<String>,

    /// Diagnostic output format. `human` (default) is the pretty terminal
    /// renderer; `json` emits one NDJSON diagnostic per line on stderr for
    /// tooling (LSP, the `haven` build orchestrator) to parse.
    #[arg(long, value_enum, default_value_t = MessageFormat::Human)]
    pub message_format: MessageFormat,
}