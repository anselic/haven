//! Loading and interpreting `haven.toml`, the per-project manifest.
//!
//! A project is any directory tree with a `haven.toml` at its root. The manifest
//! is deliberately small: a `[project]` table with a name, an optional `entry`,
//! and a `kind` list declaring what the project builds to (`bin`, `lib`,
//! `cdylib`, `staticlib`), plus an optional `[dependencies]` table naming the
//! Haven libraries this project consumes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The manifest filename looked for at the project root.
pub const MANIFEST: &str = "haven.toml";

/// A parsed `haven.toml` together with the directory it was found in. Every
/// relative path in the manifest (e.g. `entry`) is resolved against `root`.
pub struct Project {
    /// Absolute path to the directory containing `haven.toml`.
    pub root: PathBuf,
    pub project: ProjectTable,
    /// `[dependencies]`, keyed by the name the code imports the library as.
    /// Ordered so a build's dependency order - and therefore its `havenc`
    /// command line - is deterministic.
    pub dependencies: BTreeMap<String, DepSpec>,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub project: ProjectTable,
    #[serde(default)]
    pub dependencies: BTreeMap<String, DepSpec>,
}

/// One entry of `[dependencies]`.
///
/// Only the path form is understood in v1:
///
/// ```toml
/// [dependencies]
/// example_lib = { path = "../example_lib" }
/// ```
///
/// An inline *table* rather than a bare string deliberately: it is the shape that
/// can grow a `version`/registry field later without breaking manifests written
/// today. Anything else is captured by [`DepSpec::Other`] so the error can quote
/// what was actually written instead of a serde type mismatch.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DepSpec {
    /// `{ path = "../example_lib" }` - a library on disk, relative to this
    /// manifest's directory.
    Path { path: String },
    /// Any other spelling; rejected by [`Project::dependencies`] with a message
    /// naming the supported form.
    Other(toml::Value),
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
    /// v1 resolves **direct dependencies only**. A dependency that declares
    /// dependencies of its own is rejected rather than walked: the artifact
    /// records no dependency list, so the graph is knowable only from manifests,
    /// and resolving it properly needs the version/diamond arbitration no
    /// resolver exists for yet. The check doubles as cycle protection - a
    /// dependency cycle necessarily has a dependency with dependencies.
    pub fn dependencies(&self) -> Result<Vec<ResolvedDep>, String> {
        let mut out = Vec::new();
        for (name, spec) in &self.dependencies {
            let rel = match spec {
                DepSpec::Path { path } => path,
                DepSpec::Other(v) => {
                    return Err(format!(
                        "dependency `{}` is `{}`, which is not a supported form. \
                         Use a path dependency, e.g. `{} = {{ path = \"../{}\" }}`; \
                         version and registry dependencies do not exist yet",
                        name, v, name, name));
                }
            };

            let dir = self.root.join(rel);
            let manifest = dir.join(MANIFEST);
            if !manifest.is_file() {
                return Err(format!(
                    "dependency `{}` has no `{}` at `{}` (path = \"{}\")",
                    name, MANIFEST, dir.display(), rel));
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
            if !dep.dependencies.is_empty() {
                let mut names: Vec<&str> = dep.dependencies.keys().map(String::as_str).collect();
                names.sort_unstable();
                return Err(format!(
                    "dependency `{}` has dependencies of its own ({}), and transitive \
                     dependencies are not supported yet. Depend on {} directly from \
                     this manifest as well",
                    name, names.join(", "), names.join(" and ")));
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

    /// The build-output directory, `.haven/target/` under the project root.
    pub fn target_dir(&self) -> PathBuf {
        self.root.join(".haven").join("target")
    }

    /// The documentation-output directory, `.haven/doc/` under the project root.
    pub fn doc_dir(&self) -> PathBuf {
        self.root.join(".haven").join("doc")
    }

    /// Binary name derived from the project name: lowercased, with any run of
    /// non-alphanumeric characters collapsed to a single `-`. `"Sample Haven
    /// Project"` becomes `"sample-haven-project"`.
    /// Display string for the manifest version (`"0.1.0"`, `0.1`, ...), or empty
    /// when omitted. Rendered without surrounding quotes for string values.
    pub fn version_display(&self) -> String {
        match &self.project.version {
            Some(toml::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => String::new(),
        }
    }

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
