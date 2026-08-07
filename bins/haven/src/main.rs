//! `haven`: the Haven build orchestrator, in the spirit of Cargo.
//!
//! A project is a directory with a `haven.toml` manifest and a `src/` tree (see
//! `examples/example_project`). `haven` locates the manifest, then drives the
//! lower-level tools - `havenc` to compile, `havendoc` to document - writing all
//! artifacts under `.haven/` so the source tree stays clean.
//!
//! Subcommands:
//!   new <name>   scaffold a fresh project
//!   build        compile the entry file to `.haven/target/`
//!   run          build an executable and run it
//!   doc          generate docs into `.haven/doc/`

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::{Parser, Subcommand, ValueEnum};

mod config;

use config::Project;

#[derive(Parser)]
#[command(
    name = "haven",
    about = "Haven package manager",
    version,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a new Haven project in a new directory.
    New {
        /// Directory to create for the project.
        path: PathBuf,
    },

    /// Compile the project to `.haven/target/`.
    Build {
        /// Diagnostic output format forwarded to `havenc`.
        #[arg(long, value_enum, default_value_t = MessageFormat::Human)]
        message_format: MessageFormat,
    },

    /// Build an executable and run it. Arguments after `--` go to the program.
    Run {
        #[arg(long, value_enum, default_value_t = MessageFormat::Human)]
        message_format: MessageFormat,

        /// Arguments passed through to the built program.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Generate documentation into `.haven/doc/` via `havendoc`.
    Doc,
}

/// Mirrors `havenc`'s `--message-format`, so `haven` can ask for machine-readable
/// diagnostics (for an LSP or editor integration) and forward them untouched.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum MessageFormat {
    Human,
    Json,
}

impl MessageFormat {
    fn as_str(self) -> &'static str {
        match self {
            MessageFormat::Human => "human",
            MessageFormat::Json => "json",
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::New { path } => cmd_new(&path),
        Cmd::Build { message_format } => cmd_build(message_format).map(|_| ()),
        Cmd::Run { message_format, args } => cmd_run(message_format, &args),
        Cmd::Doc => cmd_doc(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// new
// ---------------------------------------------------------------------------

fn cmd_new(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("destination `{}` already exists", path.display()));
    }
    // A project name derived from the final path component; falls back to the
    // whole path if it has no usable file name.
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("haven-project")
        .to_string();

    let src_dir = path.join("src");
    std::fs::create_dir_all(&src_dir)
        .map_err(|e| format!("cannot create `{}`: {}", src_dir.display(), e))?;

    let manifest = format!(
        "[project]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         \n\
         # Optional: entry source file (default: src/main.hv)\n\
         entry = \"src/main.hv\"\n",
    );
    write_new_file(&path.join(config::MANIFEST), &manifest)?;

    let main_hv = "proc main() i32 {\n    println(\"Hello, Haven!\");\n    return 0;\n}\n";
    write_new_file(&src_dir.join("main.hv"), main_hv)?;

    // Keep the build directory out of version control.
    write_new_file(&path.join(".gitignore"), "/.haven\n")?;

    println!("Created Haven project `{}` at `{}`", name, path.display());
    Ok(())
}

fn write_new_file(path: &Path, contents: &str) -> Result<(), String> {
    std::fs::write(path, contents)
        .map_err(|e| format!("cannot write `{}`: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

/// Compile the current project. Returns the path to the produced artifact on
/// success (the executable, or the library `havenc` emitted).
fn cmd_build(fmt: MessageFormat) -> Result<PathBuf, String> {
    let project = Project::find_and_load(&cwd()?)?;
    build_project(&project, fmt, /*force_executable=*/ false)
}

/// The shared compile path used by both `build` and `run`. When
/// `force_executable` is set (as `run` requires), the manifest's
/// `shared`/`static_lib` flags are ignored so there is a binary to launch.
fn build_project(
    project: &Project,
    fmt: MessageFormat,
    force_executable: bool,
) -> Result<PathBuf, String> {
    let entry = project.entry_path();
    if !entry.is_file() {
        return Err(format!("entry file `{}` does not exist", entry.display()));
    }

    let target_dir = project.target_dir();
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("cannot create `{}`: {}", target_dir.display(), e))?;

    let out_base = target_dir.join(project.bin_name());

    let havenc = tool_path("havenc");
    let mut cmd = Command::new(&havenc);
    cmd.arg(&entry)
        .arg("--output")
        .arg(&out_base)
        .arg("--message-format")
        .arg(fmt.as_str());

    let shared = !force_executable && project.project.shared;
    let static_lib = !force_executable && project.project.static_lib;
    if shared {
        cmd.arg("--shared");
    } else if static_lib {
        cmd.arg("--static-lib");
    }

    let kind = if shared {
        "shared library"
    } else if static_lib {
        "static library"
    } else {
        "executable"
    };
    let ver = project.version_display();
    let ver = if ver.is_empty() { String::new() } else { format!(" v{ver}") };
    println!("Compiling {}{} ({})", project.project.name, ver, kind);

    let status = cmd
        .status()
        .map_err(|e| format!("failed to run `{}`: {}", havenc.display(), e))?;
    if !status.success() {
        return Err("compilation failed".to_string());
    }

    // Resolve the actual artifact path from the output kind, mirroring `havenc`'s
    // extension choices, so callers (chiefly `run`) know what to launch.
    let artifact = artifact_path(&out_base, shared, static_lib);
    println!("Finished: {}", artifact.display());
    Ok(artifact)
}

/// The on-disk path `havenc` writes for a given output base and library kind,
/// matching its per-platform extension logic.
fn artifact_path(base: &Path, shared: bool, static_lib: bool) -> PathBuf {
    if shared {
        let ext = if cfg!(target_os = "windows") {
            "dll"
        } else if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        base.with_extension(ext)
    } else if static_lib {
        let ext = if cfg!(target_os = "windows") { "lib" } else { "a" };
        base.with_extension(ext)
    } else if cfg!(target_os = "windows") {
        base.with_extension("exe")
    } else {
        base.to_path_buf()
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(fmt: MessageFormat, args: &[String]) -> Result<(), String> {
    let project = Project::find_and_load(&cwd()?)?;
    let bin = build_project(&project, fmt, /*force_executable=*/ true)?;

    println!("Running `{}`", bin.display());
    let status = Command::new(&bin)
        .args(args)
        .status()
        .map_err(|e| format!("failed to run `{}`: {}", bin.display(), e))?;
    if !status.success() {
        // Surface the program's own exit status rather than masking it.
        return Err(match status.code() {
            Some(code) => format!("program exited with status {}", code),
            None => "program terminated by signal".to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// doc
// ---------------------------------------------------------------------------

fn cmd_doc() -> Result<(), String> {
    let project = Project::find_and_load(&cwd()?)?;

    // Document the whole `src/` tree if it exists, else fall back to the entry
    // file's directory. `havendoc` recurses into directories for `.hv` files.
    let src_dir = project.root.join("src");
    let input = if src_dir.is_dir() { src_dir } else {
        project.entry_path().parent().map(Path::to_path_buf).unwrap_or(project.root.clone())
    };

    let out = project.doc_dir();
    std::fs::create_dir_all(&out)
        .map_err(|e| format!("cannot create `{}`: {}", out.display(), e))?;

    let havendoc = tool_path("havendoc");
    println!("Documenting {} -> {}", project.project.name, out.display());
    let status = Command::new(&havendoc)
        .arg(&input)
        .arg("--out")
        .arg(&out)
        .status()
        .map_err(|e| format!("failed to run `{}`: {}", havendoc.display(), e))?;
    if !status.success() {
        return Err("documentation generation failed".to_string());
    }
    println!("Finished: {}", out.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn cwd() -> Result<PathBuf, String> {
    std::env::current_dir().map_err(|e| format!("cannot determine current directory: {}", e))
}

/// Locate a sibling tool (`havenc`, `havendoc`). Prefer one next to the running
/// `haven` binary - in a workspace build all three live in the same
/// `target/<profile>/` dir - and fall back to the bare name so a `PATH` install
/// still works.
fn tool_path(name: &str) -> PathBuf {
    let exe_name = if cfg!(target_os = "windows") {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(&exe_name);
            if sibling.is_file() {
                return sibling;
            }
        }
    }
    PathBuf::from(name)
}
