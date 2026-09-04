//! Fetching git dependencies into a global cache.
//!
//! A `{ git = "<url>", rev/tag/branch = "..." }` dependency has to become a
//! directory on disk before the rest of the build can treat it like any other
//! package. That is all this module does: it clones the repository once per URL,
//! resolves the requested ref to a concrete commit, and checks that commit out
//! into its own directory, returning the path. Everything downstream - loading
//! the manifest, validating name/kind, walking the transitive graph - is shared
//! verbatim with path dependencies (see [`crate::config::Project::dependencies`]).
//!
//! # Cache layout
//!
//! Everything lives under `~/.haven/git/`, shared across all projects:
//!
//! ```text
//! ~/.haven/git/
//!   <url-slug>-<hash>/
//!     db/              a full clone of the repository, one per URL
//!     <commit-sha>/    a checked-out worktree, one per resolved commit
//! ```
//!
//! Keying the checkout by *commit* (not by the ref that named it) is what makes a
//! `rev`/`tag` dependency reuse across projects and skip the network on a warm
//! cache: the commit is resolved from the existing clone with a local
//! `rev-parse`, and if its directory is already there nothing is fetched.
//!
//! # Reproducibility
//!
//! `rev` and `tag` name an immutable commit, so a build is reproducible. `branch`
//! (and the bare-URL default) resolve the branch tip afresh on every build and so
//! are not; [`checkout`] prints a warning for those.

use std::path::{Path, PathBuf};
use std::process::Command;

use colored::Colorize;

use crate::config::GitDep;
use crate::{status, Status};

/// Fetch `dep` and return the directory its checked-out source lives in.
///
/// `name` is the dependency's `[dependencies]` key, used only for error
/// messages. Errors carry a user-facing message, already contextualized.
pub fn checkout(name: &str, dep: &GitDep) -> Result<PathBuf, String> {
    let git_ref = GitRef::from_spec(name, dep)?;
    let url = dep.git.trim();
    if url.is_empty() {
        return Err(format!("dependency `{name}` has an empty `git` URL"));
    }

    let url_dir = cache_root()?.join(url_slug(url));
    let db = url_dir.join("db");

    // 1. Ensure the repository is cloned. `fetched` tracks whether this run has
    //    already touched the network, so a freshly-cloned repo is not immediately
    //    re-fetched.
    let mut fetched = false;
    if !is_git_repo(&db) {
        fs_create_dir_all(&url_dir)?;
        // A stale, non-repository `db` (an interrupted clone) would make the
        // clone below fail with a confusing "already exists"; clear it first.
        if db.exists() {
            let _ = std::fs::remove_dir_all(&db);
        }
        status(Status::Fetching, format_args!("{name} ({url})"));
        run_git(Path::new("."), &["clone", "--quiet", url, &db.to_string_lossy()])
            .map_err(|e| format!("dependency `{name}`: failed to clone `{url}`: {e}"))?;
        fetched = true;
    }

    // 2. A mutable ref (branch, or the default tip) must be re-fetched on a warm
    //    cache to move to the current tip. Immutable refs are resolved locally
    //    first and only fetched if the commit is not already present.
    if git_ref.is_mutable() && !fetched {
        status(Status::Fetching, format_args!("{name} ({url})"));
        fetch(&db, name)?;
        fetched = true;
    }
    if git_ref.is_mutable() {
        warn_unpinned(name, &git_ref);
    }

    // 3. Resolve to a concrete commit, fetching once if a pinned ref turns out
    //    not to be in the clone yet.
    let refspec = git_ref.refspec();
    let commit = match rev_parse(&db, &refspec) {
        Ok(c) => c,
        Err(_) if !fetched => {
            status(Status::Fetching, format_args!("{name} ({url})"));
            fetch(&db, name)?;
            rev_parse(&db, &refspec).map_err(|e| resolve_err(name, &git_ref, &e))?
        }
        Err(e) => return Err(resolve_err(name, &git_ref, &e)),
    };

    // 4. Materialize the commit into its own directory. Reuse it if a previous
    //    build already did (the common warm-cache path).
    let out = url_dir.join(&commit);
    if dir_nonempty(&out) {
        return Ok(out);
    }
    // An empty leftover directory or a worktree registration orphaned by a manual
    // delete would both make `worktree add` fail; clear them first.
    if out.exists() {
        let _ = std::fs::remove_dir(&out);
    }
    run_git(&db, &["worktree", "prune"])
        .map_err(|e| format!("dependency `{name}`: git worktree prune failed: {e}"))?;
    run_git(&db, &["worktree", "add", "--quiet", "--detach", &out.to_string_lossy(), &commit])
        .map_err(|e| format!(
            "dependency `{name}`: failed to check out commit {commit} of `{url}`: {e}"))?;
    Ok(out)
}

/// The requested git ref, normalized from the manifest's `rev`/`tag`/`branch`.
#[derive(Debug)]
enum GitRef {
    /// An exact commit (`rev`).
    Rev(String),
    /// A tag (`tag`).
    Tag(String),
    /// A branch tip (`branch`) - mutable.
    Branch(String),
    /// No ref given: the default branch's tip - mutable.
    DefaultBranch,
}

impl GitRef {
    /// Read the ref out of the spec, rejecting more than one of
    /// `rev`/`tag`/`branch`.
    fn from_spec(name: &str, dep: &GitDep) -> Result<GitRef, String> {
        let given: [(&str, &Option<String>); 3] =
            [("rev", &dep.rev), ("tag", &dep.tag), ("branch", &dep.branch)];
        let set: Vec<&str> = given.iter().filter(|(_, v)| v.is_some()).map(|(k, _)| *k).collect();
        if set.len() > 1 {
            return Err(format!(
                "dependency `{name}` sets {}; a git dependency may pin at most one of \
                 `rev`, `tag`, or `branch`",
                set.join(" and ")));
        }
        Ok(match (&dep.rev, &dep.tag, &dep.branch) {
            (Some(r), _, _) => GitRef::Rev(r.clone()),
            (_, Some(t), _) => GitRef::Tag(t.clone()),
            (_, _, Some(b)) => GitRef::Branch(b.clone()),
            _ => GitRef::DefaultBranch,
        })
    }

    /// Whether the ref can move between builds (and so is non-reproducible).
    fn is_mutable(&self) -> bool {
        matches!(self, GitRef::Branch(_) | GitRef::DefaultBranch)
    }

    /// The revision expression handed to `git rev-parse`. `^{commit}` peels a tag
    /// to the commit it points at; a branch is read from its remote-tracking ref
    /// so it reflects the last fetch rather than a never-created local branch.
    fn refspec(&self) -> String {
        match self {
            GitRef::Rev(r) => format!("{r}^{{commit}}"),
            GitRef::Tag(t) => format!("refs/tags/{t}^{{commit}}"),
            GitRef::Branch(b) => format!("refs/remotes/origin/{b}^{{commit}}"),
            GitRef::DefaultBranch => "HEAD^{commit}".to_string(),
        }
    }

    /// How the ref reads in a diagnostic.
    fn describe(&self) -> String {
        match self {
            GitRef::Rev(r) => format!("rev `{r}`"),
            GitRef::Tag(t) => format!("tag `{t}`"),
            GitRef::Branch(b) => format!("branch `{b}`"),
            GitRef::DefaultBranch => "the default branch".to_string(),
        }
    }
}

/// `git fetch` all branches and tags from `origin`, updating remote-tracking
/// refs so [`GitRef::refspec`] can resolve them.
fn fetch(db: &Path, name: &str) -> Result<(), String> {
    run_git(db, &["fetch", "--quiet", "--tags", "origin"])
        .map_err(|e| format!("dependency `{name}`: git fetch failed: {e}"))?;
    Ok(())
}

/// Resolve a revision expression to a full commit SHA within `db`. Runs offline.
fn rev_parse(db: &Path, refspec: &str) -> Result<String, String> {
    let out = run_git(db, &["rev-parse", "--verify", "--quiet", refspec])?;
    let sha = out.trim().to_string();
    if sha.is_empty() {
        return Err(format!("`{refspec}` did not resolve to a commit"));
    }
    Ok(sha)
}

/// The error for a ref that could not be resolved even after a fetch.
fn resolve_err(name: &str, git_ref: &GitRef, detail: &str) -> String {
    format!(
        "dependency `{name}`: could not find {} in the repository ({detail})",
        git_ref.describe())
}

/// Warn that a build using this ref is not reproducible.
fn warn_unpinned(name: &str, git_ref: &GitRef) {
    let warn = "warning:".yellow().bold();
    eprintln!(
        "{warn} dependency `{name}` tracks {}; this build is not reproducible. \
         Pin a `rev` or `tag` for a reproducible build.",
        git_ref.describe());
}

/// Run `git` in `dir`, returning its stdout on success or a message built from
/// its stderr and exit status on failure.
fn run_git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                "`git` was not found on PATH (a git dependency needs git installed)".to_string()
            } else {
                format!("failed to run git: {e}")
            }
        })?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let msg = stderr.trim();
        Err(if msg.is_empty() { format!("git exited with {}", out.status) } else { msg.to_string() })
    }
}

/// Whether `dir` looks like a git repository (has a `.git` entry, or is itself a
/// bare/worktree repo with a `HEAD`).
fn is_git_repo(dir: &Path) -> bool {
    dir.join(".git").exists() || dir.join("HEAD").exists()
}

/// Whether `dir` exists and contains at least one entry.
fn dir_nonempty(dir: &Path) -> bool {
    std::fs::read_dir(dir).map(|mut d| d.next().is_some()).unwrap_or(false)
}

fn fs_create_dir_all(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create `{}`: {e}", dir.display()))
}

/// The root of the git cache, `~/.haven/git/`, honoring `HAVEN_HOME` for an
/// override (chiefly so tests get an isolated, disposable cache).
fn cache_root() -> Result<PathBuf, String> {
    if let Some(home) = std::env::var_os("HAVEN_HOME") {
        return Ok(PathBuf::from(home).join("git"));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| "cannot locate a home directory for the git cache \
                        (set HOME, USERPROFILE, or HAVEN_HOME)".to_string())?;
    Ok(PathBuf::from(home).join(".haven").join("git"))
}

/// A filesystem-safe, human-readable directory name for a repository URL: the
/// last path segment (minus a trailing `.git`) plus a short hash of the whole
/// URL, so two repositories that share a leaf name never collide.
fn url_slug(url: &str) -> String {
    let tail = url
        .trim_end_matches('/')
        .rsplit(['/', ':', '\\'])
        .next()
        .unwrap_or("repo")
        .trim_end_matches(".git");
    let cleaned: String = tail
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let cleaned = cleaned.trim_matches('-');
    let name = if cleaned.is_empty() { "repo" } else { cleaned };
    format!("{name}-{:016x}", hash_url(url))
}

/// A stable non-cryptographic hash of the URL, only ever used to disambiguate
/// cache directory names (so a weak hash is fine).
fn hash_url(url: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_readable_and_disambiguated() {
        let a = url_slug("https://example.com/foo/bar.git");
        let b = url_slug("https://other.com/baz/bar.git");
        // both keep the readable leaf name...
        assert!(a.starts_with("bar-"), "got {a}");
        assert!(b.starts_with("bar-"), "got {b}");
        // ...but the hash suffix keeps the two distinct.
        assert_ne!(a, b);
    }

    #[test]
    fn slug_is_stable() {
        assert_eq!(
            url_slug("https://example.com/foo/bar.git"),
            url_slug("https://example.com/foo/bar.git"));
    }

    #[test]
    fn slug_handles_local_and_trailing_paths() {
        // a local path with a trailing slash still yields its leaf
        let s = url_slug("/tmp/some/local_repo/");
        assert!(s.starts_with("local_repo-"), "got {s}");
    }

    #[test]
    fn more_than_one_ref_is_rejected() {
        let dep = GitDep {
            git: "x".into(),
            rev: Some("abc".into()),
            tag: Some("v1".into()),
            branch: None,
        };
        let err = GitRef::from_spec("d", &dep).unwrap_err();
        assert!(err.contains("at most one"), "got {err}");
    }

    #[test]
    fn ref_precedence_and_mutability() {
        let rev = GitRef::from_spec("d", &GitDep {
            git: "x".into(), rev: Some("abc".into()), tag: None, branch: None,
        }).unwrap();
        assert!(!rev.is_mutable());
        assert_eq!(rev.refspec(), "abc^{commit}");

        let branch = GitRef::from_spec("d", &GitDep {
            git: "x".into(), rev: None, tag: None, branch: Some("main".into()),
        }).unwrap();
        assert!(branch.is_mutable());
        assert_eq!(branch.refspec(), "refs/remotes/origin/main^{commit}");

        let none = GitRef::from_spec("d", &GitDep {
            git: "x".into(), rev: None, tag: None, branch: None,
        }).unwrap();
        assert!(none.is_mutable());
    }
}
