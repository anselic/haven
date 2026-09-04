//! Loading and interpreting `haven.toml`, the per-project manifest.
//!
//! A project is any directory tree with a `haven.toml` at its root. The manifest
//! is deliberately small: a `[project]` table with a name, an optional `entry`,
//! a `kind` list declaring what the project builds to (`bin`, `lib`, `cdylib`,
//! `staticlib`) and an optional `build` script to run afterwards, plus an
//! optional `[dependencies]` table naming the Haven libraries this project
//! consumes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The manifest filename looked for at the project root.
pub const MANIFEST: &str = "haven.toml";

/// A parsed `haven.toml` together with the directory it was found in. Every
/// relative path in the manifest (e.g. `entry`) is resolved against `root`.
#[derive(Debug)]
pub struct Project {
    /// Absolute path to the directory containing `haven.toml`.
    pub root: PathBuf,
    pub project: ProjectTable,
    /// `[dependencies]`, keyed by the name the code imports the library as.
    /// Ordered so a build's dependency order - and therefore its `havenc`
    /// command line - is deterministic.
    pub dependencies: BTreeMap<String, DepSpec>,
    /// The `[[c]]` tables verbatim. Flattened for callers by
    /// [`Project::c_source_files`] and [`Project::link_libs`].
    pub c: Vec<CTable>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub project: ProjectTable,
    #[serde(default)]
    pub dependencies: BTreeMap<String, DepSpec>,
    /// `[[c]]` native-code tables: C sources this package ships and the native
    /// libraries they need linked. Empty when the manifest declares none.
    ///
    /// An *array* of tables (`[[c]]`) rather than a single `[c]` so a package can
    /// group its C by concern (`files = [...]` for one subsystem, another block
    /// for another) if it wants; they are flattened by [`Project::c_source_files`]
    /// and [`Project::link_libs`]. `deny_unknown_fields` above is what makes a
    /// mistyped table name (`[[cc]]`) or key an error instead of silently dropped
    /// - the trap `[dependencies]` used to have, and the reason a `.hvmeta` could
    /// be built with no native code and fail to link with no explanation.
    #[serde(default)]
    pub c: Vec<CTable>,
}

/// One `[[c]]` table: C sources a package ships and libraries they link against.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CTable {
    /// C source files, relative to the manifest directory. Compiled and linked
    /// into any program that (transitively) depends on this package.
    #[serde(default)]
    pub files: Vec<String>,
    /// Native libraries to link, by bare name: `["m"]` becomes `-lm`. Rides in the
    /// package's artifact so a consumer links them without knowing they exist.
    #[serde(default)]
    pub libs: Vec<String>,
}

/// One entry of `[dependencies]`.
///
/// Two forms are understood:
///
/// ```toml
/// [dependencies]
/// example_lib = { path = "../example_lib" }
/// remote_lib  = { git = "https://example.com/remote_lib.git", tag = "v1.2.0" }
/// ```
///
/// An inline *table* rather than a bare string deliberately: it is the shape that
/// can grow a `version`/registry field later without breaking manifests written
/// today. Anything else is captured by [`DepSpec::Other`] so the error can quote
/// what was actually written instead of a serde type mismatch.
///
/// `untagged` picks the variant by which keys are present: `path` -> [`Path`],
/// `git` -> [`Git`], anything else -> [`Other`]. `Other` must stay last, as it
/// matches any value.
///
/// [`Path`]: DepSpec::Path
/// [`Git`]: DepSpec::Git
/// [`Other`]: DepSpec::Other
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DepSpec {
    /// `{ path = "../example_lib" }` - a library on disk, relative to this
    /// manifest's directory.
    Path { path: String },
    /// `{ git = "<url>", rev/tag/branch = "..." }` - a library in a git
    /// repository, fetched into a global cache and then treated exactly like a
    /// path dependency rooted at the checkout. See [`GitDep`] and
    /// [`crate::git::checkout`].
    Git(GitDep),
    /// Any other spelling; rejected by [`Project::dependencies`] with a message
    /// naming the supported forms.
    Other(toml::Value),
}

/// A git dependency: a repository URL and, optionally, which commit to pin to.
///
/// At most one of `rev`/`tag`/`branch` may be given (enforced by
/// [`crate::git::checkout`], not serde). None of them means the default branch's
/// current tip. `rev` and `tag` name an immutable commit and so build
/// reproducibly; `branch` (and the no-ref default) resolve afresh each build and
/// draw a warning.
#[derive(Debug, Deserialize)]
pub struct GitDep {
    /// The repository URL, passed verbatim to `git clone`. Any transport `git`
    /// understands works, including a local path (which is what the tests use).
    pub git: String,
    /// Pin to an exact commit (a full or abbreviated SHA-1).
    #[serde(default)]
    pub rev: Option<String>,
    /// Pin to a tag.
    #[serde(default)]
    pub tag: Option<String>,
    /// Track a branch's tip. Non-reproducible: re-resolved on every build.
    #[serde(default)]
    pub branch: Option<String>,
}

/// A dependency after its manifest has been located, loaded and validated.
pub struct ResolvedDep {
    /// The name this dependency is imported as - the `[dependencies]` key, which
    /// is required to equal the library's own `[project].name`.
    pub name: String,
    /// The dependency's own project, rooted at its directory.
    pub project: Project,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectTable {
    /// Human-readable project name. Also the default binary name (slugified).
    pub name: String,

    /// Version metadata. TOML lets this be a bare number (`0.1`) or a string
    /// (`"0.1.0"`), so it is kept as a raw value and only ever displayed.
    #[serde(default)]
    pub version: Option<toml::Value>,

    /// Entry source file, relative to the project root. Defaults to
    /// `src/main.hv` for a binary project and `src/lib.hv` for a library one.
    #[serde(default)]
    pub entry: Option<String>,

    /// What the project builds to. Omitted means `["bin"]`. See [`Kind`] for the
    /// recognized values and [`Project::validate`] for the legal combinations.
    #[serde(default)]
    pub kind: Option<Vec<Kind>>,

    /// A post-build script: a Haven source file, relative to the project root,
    /// compiled and run once the build's artifact exists. See
    /// [`Project::build_script`] and `haven`'s `run_build_script`.
    #[serde(default)]
    pub build: Option<String>,

    /// Whether this package *provides* the implicit prelude and so must nominate
    /// itself when compiling its own sources (`havenc --prelude <self>`). The
    /// `@!prelude` mark that says so lives in the source and is not visible to the
    /// build tool before `havenc` loads the modules, so the manifest declares it.
    /// Only `std` sets this today. `havenc` still enforces that a nominated
    /// package actually carries the mark, so a manifest that lies is caught there.
    #[serde(default, rename = "provides-prelude")]
    pub provides_prelude: bool,
}

/// A single entry of the manifest's `kind` list.
///
/// `bin` and `lib` are mutually exclusive shapes: a project is either an
/// executable or a library. `cdylib`/`staticlib` refine a `lib` into a
/// natively-compiled artifact for FFI or linking (`--shared` / `--static-lib`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// An executable with a `main` function.
    Bin,
    /// A library consumed by other Haven projects.
    Lib,
    /// A shared/dynamic native library (`.so`/`.dll`/`.dylib`), built `--shared`.
    Cdylib,
    /// A static native library (`.a`/`.lib`), built `--static-lib`.
    Staticlib,
}

/// The single artifact a `haven build` produces, resolved from the `kind` list.
/// A `havenc` invocation emits exactly one of these, which is why conflicting
/// `kind` combinations are rejected up front (see [`Project::validate`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Output {
    Executable,
    Shared,
    Static,
    /// A native Haven library: a `.hvmeta` source-blob artifact for consumption
    /// by other Haven packages, not a linkable native object.
    Lib,
}

impl Output {
    /// Human-readable label for the "Compiling ... (label)" build line.
    pub fn label(self) -> &'static str {
        match self {
            Output::Executable => "executable",
            Output::Shared => "shared library",
            Output::Static => "static library",
            Output::Lib => "library",
        }
    }

    /// The manifest `kind` spelling this output came from, as handed to a build
    /// script in `HAVEN_OUTPUT_KIND`. Deliberately the manifest's vocabulary
    /// rather than [`label`](Self::label)'s prose: a script branching on the
    /// output kind should match against the same word the author wrote in
    /// `haven.toml`, not against a phrase that exists to read well in a
    /// progress line and is therefore free to change.
    pub fn manifest_kind(self) -> &'static str {
        match self {
            Output::Executable => "bin",
            Output::Shared => "cdylib",
            Output::Static => "staticlib",
            Output::Lib => "lib",
        }
    }
}

impl Project {
    /// Walk up from `start` (and its ancestors) looking for a `haven.toml`, load
    /// and parse it. Errors carry a user-facing message, already contextualized.
    pub fn find_and_load(start: &Path) -> Result<Project, String> {
        let mut dir = Some(start);
        while let Some(d) = dir {
            let candidate = d.join(MANIFEST);
            if candidate.is_file() {
                return Self::load(&candidate);
            }
            dir = d.parent();
        }
        Err(format!(
            "no `{}` found in `{}` or any parent directory (run `haven new` to create a project)",
            MANIFEST,
            start.display(),
        ))
    }

    /// Load and parse a specific `haven.toml`.
    fn load(manifest_path: &Path) -> Result<Project, String> {
        let text = std::fs::read_to_string(manifest_path)
            .map_err(|e| format!("cannot read `{}`: {}", manifest_path.display(), e))?;
        let manifest: Manifest = toml::from_str(&text)
            .map_err(|e| format!("invalid `{}`: {}", manifest_path.display(), e))?;
        let root = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Project {
            root,
            project: manifest.project,
            dependencies: manifest.dependencies,
            c: manifest.c,
        })
    }

    /// Locate, load and validate every direct dependency, in manifest order.
    ///
    /// Each entry must name a Haven library (`kind = ["lib"]`, the kind that
    /// emits a `.hvmeta`) whose own `[project].name` equals the key it is bound
    /// to. That last rule is not bureaucracy: the bound name is what anchors the
    /// library's emitted symbols, so binding `example_lib` under some other key
    /// would compile its items under a namespace that disagrees with the
    /// library's own build. `havenc` enforces the same rule; catching it here
    /// just makes the message point at the manifest.
    ///
    /// This returns the **direct** dependencies only; a dependency that declares
    /// its own is walked by the build tool (`build_project` in `main.rs`), which
    /// builds the whole closure and binds every transitive artifact at the leaf.
    /// The graph is knowable only from manifests (a `.hvmeta` records no
    /// dependency list), so the walk resolves path deps directly with no version
    /// arbitration, and a diamond is deduped by package name. Cycle detection
    /// lives in the walk, not here.
    pub fn dependencies(&self) -> Result<Vec<ResolvedDep>, String> {
        let mut out = Vec::new();
        for (name, spec) in &self.dependencies {
            // Resolve the spec to a directory on disk. A git dependency is
            // fetched into the global cache first; from here on it is
            // indistinguishable from a path dependency rooted at its checkout.
            let dir = match spec {
                DepSpec::Path { path } => self.root.join(path),
                DepSpec::Git(git) => crate::git::checkout(name, git)?,
                DepSpec::Other(v) => {
                    return Err(format!(
                        "dependency `{}` is `{}`, which is not a supported form. \
                         Use a path dependency (`{} = {{ path = \"../{}\" }}`) or a \
                         git dependency (`{} = {{ git = \"<url>\", tag = \"...\" }}`); \
                         version and registry dependencies do not exist yet",
                        name, v, name, name, name));
                }
            };

            let manifest = dir.join(MANIFEST);
            if !manifest.is_file() {
                return Err(format!(
                    "dependency `{}` has no `{}` at `{}`",
                    name, MANIFEST, dir.display()));
            }
            // canonicalize so the dependency's own artifact paths and build
            // output do not carry the `..` from the manifest's relative spelling.
            let manifest = std::fs::canonicalize(&manifest).unwrap_or(manifest);
            let dep = Self::load(&manifest)?;
            dep.validate().map_err(|e| format!("dependency `{}`: {}", name, e))?;

            if dep.project.name != *name {
                return Err(format!(
                    "dependency `{}` resolves to a package named `{}`. The key must \
                     be the library's own name, since that is what its symbols are \
                     anchored to - rename the key to `{}`",
                    name, dep.project.name, dep.project.name));
            }
            if !matches!(dep.output_kind(), Output::Lib) {
                return Err(format!(
                    "dependency `{}` is not a Haven library: its `kind` is {:?}, \
                     which builds {} rather than a `.hvmeta`. Only `kind = [\"lib\"]` \
                     packages can be depended on",
                    name, dep.kinds(), dep.output_kind().label()));
            }
            out.push(ResolvedDep { name: name.clone(), project: dep });
        }
        Ok(out)
    }

    /// The declared `kind` list, defaulting to `["bin"]` when the manifest omits
    /// it. Callers should have already run [`validate`](Self::validate).
    pub fn kinds(&self) -> Vec<Kind> {
        self.project
            .kind
            .clone()
            .unwrap_or_else(|| vec![Kind::Bin])
    }

    /// Whether this project is a library (its `kind` list has no `bin`). Governs
    /// the default entry file and whether `haven run` is meaningful.
    pub fn is_library(&self) -> bool {
        !self.kinds().contains(&Kind::Bin)
    }

    /// Reject `kind` combinations `havenc` cannot satisfy in a single build.
    /// A build emits exactly one artifact, so an executable cannot be paired
    /// with a native library, nor a shared library with a static one.
    pub fn validate(&self) -> Result<(), String> {
        let kinds = self.kinds();
        if kinds.is_empty() {
            return Err("`kind` must list at least one output kind \
                        (e.g. `kind = [\"bin\"]`)"
                .to_string());
        }
        let has_bin = kinds.contains(&Kind::Bin);
        let has_shared = kinds.contains(&Kind::Cdylib);
        let has_static = kinds.contains(&Kind::Staticlib);
        if has_shared && has_static {
            return Err("`kind` cannot request both `cdylib` and `staticlib`; \
                        a build produces one native library, not both"
                .to_string());
        }
        if has_bin && (has_shared || has_static) {
            return Err("`kind` cannot combine `bin` with `cdylib`/`staticlib`; \
                        a build is either an executable or a library"
                .to_string());
        }
        Ok(())
    }

    /// The single artifact `haven build` produces for this project's `kind`.
    /// `cdylib` builds shared, `staticlib` builds static, `bin` builds an
    /// executable, and a bare `lib` emits a `.hvmeta` native-library artifact.
    pub fn output_kind(&self) -> Output {
        let kinds = self.kinds();
        if kinds.contains(&Kind::Cdylib) {
            Output::Shared
        } else if kinds.contains(&Kind::Staticlib) {
            Output::Static
        } else if kinds.contains(&Kind::Bin) {
            Output::Executable
        } else {
            // A bare `lib`: emit a `.hvmeta` source-blob artifact for other Haven
            // packages to consume. The build still validates the library (its
            // pre-mono typecheck runs, and no `main` is required).
            Output::Lib
        }
    }

    /// Absolute path to the entry source file. Defaults to `src/main.hv` for a
    /// binary and `src/lib.hv` for a library, unless `entry` overrides it.
    pub fn entry_path(&self) -> PathBuf {
        let default = if self.is_library() { "src/lib.hv" } else { "src/main.hv" };
        let entry = self.project.entry.as_deref().unwrap_or(default);
        self.root.join(entry)
    }

    /// Every `[[c]]` source file, resolved to an absolute path against the
    /// manifest root and flattened across all `[[c]]` tables, in manifest order.
    /// Empty when the package ships no native code.
    pub fn c_source_files(&self) -> Vec<PathBuf> {
        self.c.iter()
            .flat_map(|table| table.files.iter())
            .map(|rel| self.root.join(rel))
            .collect()
    }

    /// Every native library named across all `[[c]]` tables (`-l<name>`), in
    /// manifest order. These ride in the package's artifact so a consumer links
    /// them transitively.
    pub fn link_libs(&self) -> Vec<String> {
        self.c.iter()
            .flat_map(|table| table.libs.iter())
            .cloned()
            .collect()
    }

    /// Absolute path to the post-build script, when the manifest declares one.
    /// Resolved against the project root like `entry`.
    pub fn build_script(&self) -> Option<PathBuf> {
        self.project.build.as_ref().map(|rel| self.root.join(rel))
    }

    /// The build-output directory, `.haven/target/` under the project root.
    pub fn target_dir(&self) -> PathBuf {
        self.root.join(".haven").join("target")
    }

    /// Where a compiled build script lives, `.haven/build/` under the project
    /// root. Kept out of `target/` so the script's own executable is never
    /// mistaken for the project's artifact.
    pub fn build_dir(&self) -> PathBuf {
        self.root.join(".haven").join("build")
    }

    /// The documentation-output directory, `.haven/doc/` under the project root.
    pub fn doc_dir(&self) -> PathBuf {
        self.root.join(".haven").join("doc")
    }

    /// Display string for the manifest version (`"0.1.0"`, `0.1`, ...), or empty
    /// when omitted. Rendered without surrounding quotes for string values.
    pub fn version_display(&self) -> String {
        match &self.project.version {
            Some(toml::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => String::new(),
        }
    }

    /// Binary name derived from the project name: lowercased, with any run of
    /// non-alphanumeric characters collapsed to a single `-`. `"Sample Haven
    /// Project"` becomes `"sample-haven-project"`.
    pub fn bin_name(&self) -> String {
        let mut out = String::new();
        let mut prev_dash = false;
        for ch in self.project.name.chars() {
            if ch.is_alphanumeric() {
                out.extend(ch.to_lowercase());
                prev_dash = false;
            } else if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        }
        let trimmed = out.trim_matches('-');
        if trimmed.is_empty() { "output".to_string() } else { trimmed.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `Project` from raw manifest text, rooted at `/pkg`, so the accessors
    /// can be tested without touching the filesystem. `Manifest`'s fields are the
    /// same the loader fills from `toml::from_str`, so this mirrors `Project::load`.
    fn project_from(toml_src: &str) -> Result<Project, String> {
        let manifest: Manifest = toml::from_str(toml_src).map_err(|e| e.to_string())?;
        Ok(Project {
            root: PathBuf::from("/pkg"),
            project: manifest.project,
            dependencies: manifest.dependencies,
            c: manifest.c,
        })
    }

    const LIB: &str = "[project]\nname = \"std\"\nkind = [\"lib\"]\n";

    #[test]
    fn no_c_table_is_empty() {
        let p = project_from(LIB).unwrap();
        assert!(p.c_source_files().is_empty());
        assert!(p.link_libs().is_empty());
    }

    #[test]
    fn parses_c_files_and_libs() {
        let p = project_from(&format!(
            "{LIB}\n[[c]]\nfiles = [\"c/env.c\", \"c/rt.c\"]\nlibs = [\"m\"]\n"
        )).unwrap();
        // files resolve to absolute paths under the manifest root
        assert_eq!(p.c_source_files(), vec![
            PathBuf::from("/pkg/c/env.c"),
            PathBuf::from("/pkg/c/rt.c"),
        ]);
        assert_eq!(p.link_libs(), vec!["m".to_string()]);
    }

    #[test]
    fn flattens_multiple_c_tables_in_order() {
        let p = project_from(&format!(
            "{LIB}\n[[c]]\nfiles = [\"c/a.c\"]\nlibs = [\"m\"]\n\
             \n[[c]]\nfiles = [\"c/b.c\"]\nlibs = [\"pthread\"]\n"
        )).unwrap();
        assert_eq!(p.c_source_files(), vec![
            PathBuf::from("/pkg/c/a.c"),
            PathBuf::from("/pkg/c/b.c"),
        ]);
        assert_eq!(p.link_libs(), vec!["m".to_string(), "pthread".to_string()]);
    }

    #[test]
    fn a_c_table_may_omit_files_or_libs() {
        let p = project_from(&format!("{LIB}\n[[c]]\nlibs = [\"m\"]\n")).unwrap();
        assert!(p.c_source_files().is_empty());
        assert_eq!(p.link_libs(), vec!["m".to_string()]);
    }

    #[test]
    fn rejects_unknown_top_level_table() {
        // the whole point of `deny_unknown_fields`: a mistyped `[[cc]]` is an error,
        // not a silently dropped native-code block that fails to link later.
        let err = project_from(&format!("{LIB}\n[[cc]]\nfiles = [\"c/rt.c\"]\n")).unwrap_err();
        assert!(err.contains("cc") || err.contains("unknown"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_c_key() {
        let err = project_from(&format!("{LIB}\n[[c]]\nfils = [\"c/rt.c\"]\n")).unwrap_err();
        assert!(err.contains("fils") || err.contains("unknown"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_project_key() {
        let err = project_from("[project]\nnaem = \"std\"\nkind = [\"lib\"]\n").unwrap_err();
        assert!(err.contains("naem") || err.contains("unknown") || err.contains("missing"),
            "got: {err}");
    }
}
