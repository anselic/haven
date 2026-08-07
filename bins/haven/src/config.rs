//! Loading and interpreting `haven.toml`, the per-project manifest.
//!
//! A project is any directory tree with a `haven.toml` at its root. The manifest
//! is deliberately small: a `[project]` table with a name and a handful of
//! optional knobs mirroring `havenc`'s flags (`entry`, `shared`, `static_lib`).

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
    /// [`Project::DEFAULT_ENTRY`] when omitted.
    #[serde(default)]
    pub entry: Option<String>,

    /// Build a shared library (`.so`/`.dll`/`.dylib`) instead of an executable.
    #[serde(default)]
    pub shared: bool,

    /// Build a static library (`.a`/`.lib`) instead of an executable.
    #[serde(default)]
    pub static_lib: bool,
}

impl Project {
    pub const DEFAULT_ENTRY: &'static str = "src/main.hv";

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

    /// Absolute path to the entry source file.
    pub fn entry_path(&self) -> PathBuf {
        let entry = self.project.entry.as_deref().unwrap_or(Self::DEFAULT_ENTRY);
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
