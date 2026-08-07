//! Loading and interpreting `haven.toml`, the per-project manifest.
//!
//! A project is any directory tree with a `haven.toml` at its root. The manifest
//! is deliberately small: a `[project]` table with a name, an optional `entry`,
//! and a `kind` list declaring what the project builds to (`bin`, `lib`,
//! `cdylib`, `staticlib`).

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
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub project: ProjectTable,
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
}

impl Output {
    /// Human-readable label for the "Compiling ... (label)" build line.
    pub fn label(self) -> &'static str {
        match self {
            Output::Executable => "executable",
            Output::Shared => "shared library",
            Output::Static => "static library",
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
        Ok(Project { root, project: manifest.project })
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
    /// `cdylib` builds shared, `staticlib` (or a bare `lib`) builds static, and
    /// `bin` builds an executable.
    pub fn output_kind(&self) -> Output {
        let kinds = self.kinds();
        if kinds.contains(&Kind::Cdylib) {
            Output::Shared
        } else if kinds.contains(&Kind::Staticlib) {
            Output::Static
        } else if kinds.contains(&Kind::Bin) {
            Output::Executable
        } else {
            // A bare `lib`: compile it to a static archive, which both verifies
            // it (no `main` required) and yields a linkable artifact.
            Output::Static
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
