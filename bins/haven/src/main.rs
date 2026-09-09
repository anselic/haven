//! `haven`: the Haven build orchestrator, in the spirit of Cargo.
//!
//! A project is a directory with a `haven.toml` manifest and a `src/` tree (see
//! `examples/example_plugin`). `haven` locates the manifest, then drives the
//! lower-level tools - `havenc` to compile, `havendoc` to document - writing all
//! artifacts under `.haven/` so the source tree stays clean.
//!
//! Subcommands:
//!   new <name>   scaffold a fresh project
//!   build        compile the entry file to `.haven/target/`
//!   run          build an executable and run it
//!   doc          generate docs into `.haven/doc/`

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::{Args, Parser, Subcommand, ValueEnum};

mod config;
mod git;

use config::{Output, Project};

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

        /// Scaffold a library (`src/lib.hv`, `kind = ["lib"]`).
        #[arg(long, conflicts_with = "bin")]
        lib: bool,

        /// Scaffold an executable (`src/main.hv`, `kind = ["bin"]`). This is the
        /// default when neither `--lib` nor `--bin` is given.
        #[arg(long)]
        bin: bool,
    },

    /// Compile the project to `.haven/target/`.
    Build {
        /// Diagnostic output format forwarded to `havenc`.
        #[arg(long, value_enum, default_value_t = MessageFormat::Human)]
        message_format: MessageFormat,

        #[command(flatten)]
        compiler_flags: CompilerFlags,
    },

    /// Build an executable and run it. Arguments after `--` go to the program.
    Run {
        #[arg(long, value_enum, default_value_t = MessageFormat::Human)]
        message_format: MessageFormat,

        #[command(flatten)]
        compiler_flags: CompilerFlags,

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

/// Mirrors `havenc`'s `-F`/`--compiler-flags`, shared by `build` and `run`.
#[derive(Args, Clone, Debug)]
struct CompilerFlags {
    /// Flags for the LLVM IR compiler, forwarded verbatim to
    /// `havenc --compiler-flags`.
    ///
    /// This *replaces* `havenc`'s default rather than adding to it, so carry
    /// `-O3 -Wno-override-module` along unless you mean to drop them. The usual
    /// reason to reach for this is an optimization report:
    ///
    ///     haven build -F "-O3 -Wno-override-module -Rpass=loop-vectorize \
    ///                     -Rpass-missed=loop-vectorize"
    ///
    /// Unlike `havenc`'s own flag, a leading `-` in the value needs no `=` or
    /// escaping here.
    #[arg(short = 'F', long, value_name = "FLAGS", allow_hyphen_values = true)]
    compiler_flags: Option<String>,

    /// Keep the generated LLVM IR (`.ll`) beside the artifact in
    /// `.haven/target/`, forwarded to `havenc --emit-ir`.
    #[arg(long)]
    emit_ir: bool,

    /// Emit the optimized LLVM IR (`.opt.ll`) beside the artifact, following the
    /// optimization flags, forwarded to `havenc --emit-optimized-ir`.
    #[arg(long)]
    emit_optimized_ir: bool,
}

/// The settings one `haven build`/`haven run` applies to every `havenc` it
/// drives, threaded through the dependency walk so a leaf and its libraries are
/// compiled alike. Grouped rather than passed one parameter at a time because
/// the walk hands them down through four frames untouched.
#[derive(Copy, Clone)]
struct BuildOpts<'a> {
    fmt: MessageFormat,
    /// `havenc --compiler-flags`, when the user overrode it. `None` leaves
    /// `havenc`'s own default in force, so the default lives in one place.
    compiler_flags: Option<&'a str>,
    /// Forwarded to `havenc --emit-ir`: keep the generated `.ll`.
    emit_ir: bool,
    /// Forwarded to `havenc --emit-optimized-ir`: emit the `.opt.ll`.
    emit_optimized_ir: bool,
}

impl<'a> BuildOpts<'a> {
    fn new(fmt: MessageFormat, flags: &'a CompilerFlags) -> Self {
        BuildOpts {
            fmt,
            compiler_flags: flags.compiler_flags.as_deref(),
            emit_ir: flags.emit_ir,
            emit_optimized_ir: flags.emit_optimized_ir,
        }
    }
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
    // `haven` prints human text only (it never speaks JSON itself), so the
    // default format is the right one for its own crash report.
    haven_common::diag::install_ice_hook("haven");
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::New { path, lib, .. } => cmd_new(&path, lib),
        Cmd::Build { message_format, compiler_flags } =>
            cmd_build(BuildOpts::new(message_format, &compiler_flags)).map(|_| ()),
        Cmd::Run { message_format, compiler_flags, args } =>
            cmd_run(BuildOpts::new(message_format, &compiler_flags), &args),
        Cmd::Doc => cmd_doc(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            status(Status::Error, e);
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// new
// ---------------------------------------------------------------------------

fn cmd_new(path: &Path, is_lib: bool) -> Result<(), String> {
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

    let kind = if is_lib { "lib" } else { "bin" };
    let manifest = format!(
        "[project]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         kind = [\"{kind}\"]\n",
    );
    write_new_file(&path.join(config::MANIFEST), &manifest)?;

    if is_lib {
        let lib_hv = "/// Add two integers.\n\
                      pub proc add(a: i32, b: i32) i32 {\n\
                      \x20   return a + b;\n\
                      }\n";
        write_new_file(&src_dir.join("lib.hv"), lib_hv)?;
    } else {
        let main_hv =
            "proc main() i32 {\n    println(\"Hello, World!\");\n    return 0;\n}\n";
        write_new_file(&src_dir.join("main.hv"), main_hv)?;
    }

    // Keep the build directory out of version control.
    write_new_file(&path.join(".gitignore"), "/.haven\n")?;

    let what = if is_lib { "library" } else { "executable" };
    status(Status::Created, format_args!("{} `{}` at `{}`", what, name, path.display()));
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
fn cmd_build(opts: BuildOpts<'_>) -> Result<PathBuf, String> {
    let project = Project::find_and_load(&cwd()?)?;
    build_project(&project, opts, /*force_executable=*/ false)
}

/// The shared compile path used by both `build` and `run`: build `project`'s
/// full dependency closure, then compile `project` against it. When
/// `force_executable` is set (as `run` requires), the manifest's `kind` is
/// overridden to build an executable so there is a binary to launch.
fn build_project(
    project: &Project,
    opts: BuildOpts<'_>,
    force_executable: bool,
) -> Result<PathBuf, String> {
    project.validate()?;
    // Build the whole dependency graph (each package once), then compile this
    // project against the flattened closure. The root is seeded onto the build
    // stack so a cycle back to it is caught here, not one frame down.
    let mut cache: HashMap<PathBuf, (PathBuf, Vec<(String, PathBuf)>)> = HashMap::new();
    let mut on_stack: Vec<PathBuf> = vec![canonical_root(project)];
    let deps = dep_closure(project, opts, &mut cache, &mut on_stack)?;
    compile_project(project, &deps, force_executable, opts)
}

/// The canonicalized package root: the identity a package is memoized and
/// cycle-checked by, so two spellings of one directory count as a single node.
fn canonical_root(project: &Project) -> PathBuf {
    std::fs::canonicalize(&project.root).unwrap_or_else(|_| project.root.clone())
}

/// Build every package reachable from `project` through `[dependencies]` and
/// return the flat `(name, artifact)` set to bind as `--dep` when `project`
/// compiles. The set is *transitive*: an intermediate library's own
/// dependencies have to be bound at the leaf too, because a `.hvmeta` records no
/// dependency list, so the leaf re-resolves the intermediate's `import`s itself.
fn dep_closure(
    project: &Project,
    opts: BuildOpts<'_>,
    cache: &mut HashMap<PathBuf, (PathBuf, Vec<(String, PathBuf)>)>,
    on_stack: &mut Vec<PathBuf>,
) -> Result<Vec<(String, PathBuf)>, String> {
    // BTreeMap dedups a diamond by package name and keeps the command line
    // deterministic; a direct dependency wins over the same name reached only
    // transitively.
    let mut flat: BTreeMap<String, PathBuf> = BTreeMap::new();
    for dep in project.dependencies()? {
        let (artifact, sub) = build_dependency(&dep.project, opts, cache, on_stack)
            .map_err(|e| format!("dependency `{}`: {}", dep.name, e))?;
        for (name, path) in sub {
            flat.entry(name).or_insert(path);
        }
        flat.insert(dep.name.clone(), artifact);
    }
    Ok(flat.into_iter().collect())
}

/// Build one library dependency to its `.hvmeta`, memoized so a package shared
/// by several dependents is built once. Returns the artifact and the package's
/// own transitive closure (for a dependent to merge). A package already on the
/// build stack is a dependency cycle.
fn build_dependency(
    project: &Project,
    opts: BuildOpts<'_>,
    cache: &mut HashMap<PathBuf, (PathBuf, Vec<(String, PathBuf)>)>,
    on_stack: &mut Vec<PathBuf>,
) -> Result<(PathBuf, Vec<(String, PathBuf)>), String> {
    let key = canonical_root(project);
    if let Some(hit) = cache.get(&key) {
        return Ok(hit.clone());
    }
    if on_stack.contains(&key) {
        return Err(format!("dependency cycle through `{}`", project.project.name));
    }
    on_stack.push(key.clone());
    project.validate()?;
    let deps = dep_closure(project, opts, cache, on_stack)?;
    let artifact = compile_project(project, &deps, /*force_executable=*/ false, opts)?;
    on_stack.pop();
    cache.insert(key, (artifact.clone(), deps.clone()));
    Ok((artifact, deps))
}

/// Compile one project's entry file with `havenc`, binding `deps` as `--dep`
/// and running its post-build script if it declared one. Dependency resolution
/// and building is the caller's job; this is the single compiler invocation.
fn compile_project(
    project: &Project,
    deps: &[(String, PathBuf)],
    force_executable: bool,
    opts: BuildOpts<'_>,
) -> Result<PathBuf, String> {
    let entry = project.entry_path();
    if !entry.is_file() {
        return Err(format!("entry file `{}` does not exist", entry.display()));
    }

    let target_dir = project.target_dir();
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| format!("cannot create `{}`: {}", target_dir.display(), e))?;

    let out_base = target_dir.join(project.bin_name());

    let output = if force_executable {
        Output::Executable
    } else {
        project.output_kind()
    };

    let havenc = tool_path("havenc");
    let mut cmd = Command::new(&havenc);
    cmd.arg(&entry)
        .arg("--output")
        .arg(&out_base)
        .arg("--package-name")
        .arg(&project.project.name)
        .arg("--message-format")
        .arg(opts.fmt.as_str());

    // Given to every package in the build, not just the leaf, so a report covers
    // the libraries too. A `lib` dependency stops before codegen and never
    // consults these, which is why the flags are still worth passing down: the
    // one package in the graph that does reach LLVM should not depend on whether
    // it happened to be the one the user was standing in.
    //
    // Spelled `--flag=value` in one argv entry: the value all but always opens
    // with `-O`, and a separate entry would be read as another option.
    if let Some(flags) = opts.compiler_flags {
        cmd.arg(format!("--compiler-flags={}", flags));
    }

    // IR-emitting flags ride along the same walk as the compiler flags: the one
    // package that reaches LLVM drops its `.ll`/`.opt.ll` beside the artifact in
    // `.haven/target/`. A `lib` dependency stops before codegen and ignores them.
    if opts.emit_ir {
        cmd.arg("--emit-ir");
    }
    if opts.emit_optimized_ir {
        cmd.arg("--emit-optimized-ir");
    }

    match output {
        Output::Shared => { cmd.arg("--shared"); }
        Output::Static => { cmd.arg("--static-lib"); }
        Output::Lib => { cmd.arg("--lib"); }
        Output::Executable => {}
    }

    for (name, artifact) in deps {
        cmd.arg("--dep").arg(format!("{}={}", name, artifact.display()));
    }

    // A prelude-providing package (only `std` today) nominates itself: its
    // `@!prelude` mark is invisible until its modules load, so the manifest says
    // so and we pass it through. `havenc` still checks the mark is really there.
    if project.project.provides_prelude {
        cmd.arg("--prelude").arg(&project.project.name);
    }

    // `[[c]]` native code. For a `lib` these ride into the `.hvmeta` for its
    // consumers' leaves to compile; for a `bin`/`cdylib`/`staticlib` `havenc`
    // compiles the sources and links the libraries into the artifact directly (so
    // a binary can link a system library like raylib). The flags are the same
    // either way. Caveat: a `staticlib`'s `libs` cannot be recorded in a `.a`
    // archive, so the host links those - the same limitation dependency libs have.
    for file in project.c_source_files() {
        cmd.arg("--c-file").arg(file);
    }
    for lib in project.link_libs() {
        cmd.arg("--link-lib").arg(lib);
    }

    let label = describe(project);
    status(Status::Compiling, format_args!("{} ({})", label, output.label()));

    let exit = cmd
        .status()
        .map_err(|e| format!("failed to run `{}`: {}", havenc.display(), e))?;
    havenc_outcome(exit, "compilation failed")?;

    // Resolve the actual artifact path from the output kind, mirroring `havenc`'s
    // extension choices, so callers (chiefly `run`) know what to launch.
    let artifact = artifact_path(&out_base, output);

    // The artifact exists; hand it to the project's own post-build script, if it
    // declared one. Runs for dependencies too, since a library's packaging step
    // is as much its own business as a leaf's.
    if let Some(script) = project.build_script() {
        // `opts.fmt` only: a packaging script is a host program compiled against
        // `std` alone, and flags aimed at the artifact's codegen (an optimization
        // report, say) have no business turning up in its build.
        run_build_script(project, &script, &artifact, output, opts.fmt)?;
    }

    status(Status::Finished, &label);
    Ok(artifact)
}

/// The on-disk path `havenc` writes for a given output base and output kind,
/// matching its per-platform extension logic.
fn artifact_path(base: &Path, output: Output) -> PathBuf {
    match output {
        Output::Shared => {
            let ext = if cfg!(target_os = "windows") {
                "dll"
            } else if cfg!(target_os = "macos") {
                "dylib"
            } else {
                "so"
            };
            base.with_extension(ext)
        }
        Output::Static => {
            let ext = if cfg!(target_os = "windows") { "lib" } else { "a" };
            base.with_extension(ext)
        }
        Output::Executable => {
            if cfg!(target_os = "windows") {
                base.with_extension("exe")
            } else {
                base.to_path_buf()
            }
        }
        Output::Lib => base.with_extension("hvmeta"),
    }
}

// ---------------------------------------------------------------------------
// post-build script
// ---------------------------------------------------------------------------

/// Compile a stale build script and run it after the project artifact is ready.
///
/// Build scripts receive artifact details through the environment and compile
/// against `std` only, without the project's dependencies. A nonzero exit fails
/// the build; stdout and stderr are inherited.
fn run_build_script(
    project: &Project,
    script: &Path,
    artifact: &Path,
    output: Output,
    fmt: MessageFormat,
) -> Result<(), String> {
    let shown = relative_to(&project.root, script).display().to_string();
    if !script.is_file() {
        return Err(format!(
            "build script `{}` does not exist (declared as `build` in {})",
            shown, config::MANIFEST));
    }

    let build_dir = project.build_dir();
    std::fs::create_dir_all(&build_dir)
        .map_err(|e| format!("cannot create `{}`: {}", build_dir.display(), e))?;

    let stem = script.file_stem().and_then(|s| s.to_str()).unwrap_or("build");
    let out_base = build_dir.join(stem);
    let exe = artifact_path(&out_base, Output::Executable);

    if is_stale(script, &exe) {
        status(Status::Compiling, format_args!("{} (build script)", shown));
        let havenc = tool_path("havenc");
        let exit = Command::new(&havenc)
            .arg(script)
            .arg("--output")
            .arg(&out_base)
            .arg("--message-format")
            .arg(fmt.as_str())
            .status()
            .map_err(|e| format!("failed to run `{}`: {}", havenc.display(), e))?;
        havenc_outcome(exit, format!("build script `{}` failed to compile", shown))?;
    }

    // The build's details, passed as environment variables rather than argv: a
    // positional contract rots the moment a field is added, and an environment
    // can be reproduced by hand, so a script can be run standalone under a
    // debugger without a build to drive it.
    let exit = Command::new(&exe)
        .current_dir(&project.root)
        .env("HAVEN_PROJECT_ROOT", &project.root)
        .env("HAVEN_PKG_NAME", &project.project.name)
        .env("HAVEN_PKG_VERSION", project.version_display())
        .env("HAVEN_TARGET_DIR", project.target_dir())
        .env("HAVEN_ARTIFACT", artifact)
        .env("HAVEN_OUTPUT_KIND", output.manifest_kind())
        .env("HAVEN_TARGET_OS", target_os())
        .status()
        .map_err(|e| format!("failed to run build script `{}`: {}", exe.display(), e))?;
    if !exit.success() {
        return Err(match exit.code() {
            Some(code) => format!("build script `{}` exited with status {}", shown, code),
            None => format!("build script `{}` terminated by signal", shown),
        });
    }
    Ok(())
}

/// Whether `exe` needs rebuilding from `src`, by modification time.
///
/// Only the script's *entry* file is consulted: `haven` cannot see which modules
/// it imports without asking `havenc` to tell it, and nothing in the build tracks
/// dependencies at that granularity yet. A script split across several files can
/// therefore go stale - touch the entry, or delete `.haven/build/`, to force a
/// recompile. An unreadable time on either side answers "stale", so a missing
/// executable (the first build) or a filesystem without mtimes recompiles rather
/// than silently running something old.
fn is_stale(src: &Path, exe: &Path) -> bool {
    let time = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    match (time(src), time(exe)) {
        (Some(src_t), Some(exe_t)) => src_t > exe_t,
        _ => true,
    }
}

/// The operating system a build script should package for, in the spelling Rust
/// uses for `target_os`. Reads the *host* today because `haven` has no
/// cross-compilation story; it is passed explicitly all the same, so a script
/// branches on the build's target rather than on where it happens to be running,
/// and keeps working unchanged when one arrives.
fn target_os() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        std::env::consts::OS
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(opts: BuildOpts<'_>, args: &[String]) -> Result<(), String> {
    let project = Project::find_and_load(&cwd()?)?;
    if project.is_library() {
        return Err("cannot `haven run` a library project \
                    (its `kind` has no `bin`)"
            .to_string());
    }
    let bin = build_project(&project, opts, /*force_executable=*/ true)?;

    status(Status::Running, project.bin_name());
    let exit = Command::new(&bin)
        .args(args)
        .status()
        .map_err(|e| format!("failed to run `{}`: {}", bin.display(), e))?;
    if !exit.success() {
        // Surface the program's own exit status rather than masking it.
        return Err(match exit.code() {
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
    status(Status::Compiling, describe(&project));
    let exit = Command::new(&havendoc)
        .arg(&input)
        .arg("--out")
        .arg(&out)
        .status()
        .map_err(|e| format!("failed to run `{}`: {}", havendoc.display(), e))?;
    if !exit.success() {
        return Err("documentation generation failed".to_string());
    }
    // The generated docs are what the user opens next, so name where they landed
    // - but relative to the project root, not as a long absolute path.
    status(Status::Finished, relative_to(&project.root, &out).display());
    Ok(())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Created a new project, file, or directory.
    Created,
    /// Fetching a git dependency into the cache.
    Fetching,
    /// Compiling a project or dependency.
    Compiling,
    /// Finished compiling a project or dependency.
    Finished,
    /// Running a built executable.
    Running,

    Error,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use colored::Colorize;

        let width = 10;

        match self {
            Status::Created   => write!(f, "{:>width$}", "Created".green()),
            Status::Fetching  => write!(f, "{:>width$}", "Fetching".cyan()),
            Status::Compiling => write!(f, "{:>width$}", "Compiling".blue()),
            Status::Finished  => write!(f, "{:>width$}", "Finished".green()),
            Status::Running   => write!(f, "{:>width$}", "Running".green()),
            Status::Error     => write!(f, "{:>width$}", "Error".red()),
        }
    }
}

fn status(verb: Status, detail: impl std::fmt::Display) {
    if verb == Status::Error {
        eprintln!("{} {}", verb, detail);
    } else {
        println!("{} {}", verb, detail);
    }
}

/// How a project is named in progress lines: `name v1.2.3`, or just the name
/// when the manifest omits a version.
fn describe(project: &Project) -> String {
    let ver = project.version_display();
    if ver.is_empty() {
        project.project.name.clone()
    } else {
        format!("{} v{}", project.project.name, ver)
    }
}

/// `path` expressed relative to `base` when it lies inside it, so output paths
/// print as `.haven\doc` rather than a full (possibly `\\?\`-prefixed) path.
/// Falls back to `path` unchanged when it does not.
fn relative_to(base: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(base).unwrap_or(path).to_path_buf()
}

fn cwd() -> Result<PathBuf, String> {
    std::env::current_dir().map_err(|e| format!("cannot determine current directory: {}", e))
}

/// Find `havenc` or `havendoc` next to the current executable, falling back to
/// `PATH`. Exit code 101 is reported as a compiler crash; other failures are
/// ordinary compilation errors.
fn havenc_outcome(exit: std::process::ExitStatus, failed: impl Into<String>) -> Result<(), String> {
    if exit.success() {
        return Ok(());
    }
    Err(match exit.code() {
        Some(101) => "the compiler crashed (internal error) - see the report above".to_string(),
        Some(_) => failed.into(),
        None => "the compiler was terminated by a signal".to_string(),
    })
}

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
