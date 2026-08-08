//! The `.hvmeta` artifact — a Haven native library's serialized form (v1).
//!
//! A `kind = ["lib"]` package does not drive to LLVM. A monomorphizing,
//! whole-program-source-merging compiler cannot pre-compile a library's generic
//! code, and its leaf re-typechecks the whole merged program anyway, so a typed
//! interface would just be recomputed and discarded there. The minimal *correct*
//! artifact is therefore the package's **own source**, loaded at the leaf exactly
//! the way `std` is loaded today.
//!
//! This works only because naming is package-deterministic: given the package
//! name and a module's source, the leaf re-derives byte-identical slugs
//! (`foo.geo$Point`) to what the lib would emit standalone. So v1 stores no typed
//! AST, no signatures, no `DefId` rewriting — just a header and the source blobs.
//!
//! ## What is *not* here (deliberately)
//!
//! No compiled objects, no serialized interface (`MetaItem`/`MetaType`), no
//! reachability pruning, no target triple. Those belong to a future v2 hybrid
//! format (objects + interface so the leaf can skip re-typechecking). [`HavenMeta`]
//! is meant to grow toward it *additively* — an optional interface section and
//! per-item object blobs alongside the existing `modules` source — not by
//! restructuring what is here.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The on-disk format version. Bump on any breaking layout change; [`read`]
/// rejects a mismatch rather than silently misparsing an older/newer artifact.
/// There is no migration machinery — one producer, one consumer, both `havenc`.
pub const FORMAT_VERSION: u32 = 1;

/// A complete `.hvmeta` artifact: a header plus the package's own source modules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HavenMeta {
    pub header: Header,
    /// This package's OWN source modules only — never `std`/prelude, which the
    /// leaf already embeds and re-resolves against its own copy.
    pub modules: Vec<MetaModule>,
}

/// Identifying metadata for the artifact as a whole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    /// Layout version; see [`FORMAT_VERSION`].
    pub format_version: u32,
    /// The `havenc` that produced this (`env!("CARGO_PKG_VERSION")`), part of the
    /// fingerprint so a compiler upgrade invalidates dependents.
    pub havenc_version: String,
    /// The namespace the leaf must load these modules under, so it re-derives the
    /// same package-anchored slugs the lib would emit standalone.
    pub package_name: String,
    /// A deterministic, location-independent digest of the package (see
    /// [`fingerprint`]); a build tool diffs this to decide whether a dependent
    /// needs rebuilding.
    pub fingerprint: [u8; 32],
}

/// One of the package's own source modules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetaModule {
    /// The module's package-root-relative key, forward-slashed — never an
    /// absolute filesystem path, or the artifact would leak the checkout location
    /// and defeat reproducibility. E.g. `geo.hv`, `dsp/osc.hv`.
    pub key: String,
    /// The module's full `.hv` source text.
    pub source: String,
    /// True for the lib root module (`lib.hv`), which slugs to the bare package
    /// name. Recorded now because it is cheap here and annoying to reconstruct at
    /// the leaf.
    pub is_root: bool,
}

/// A deterministic, location- and order-independent digest of a package.
///
/// Hashes the package name, the producing `havenc` version, and every module's
/// `(key, source)` **sorted by key** — so import order cannot change the result.
/// No absolute paths, no timestamps, no target triple: a source-blob lib is
/// target-independent (compiled fresh per target at the leaf), so the same source
/// from any checkout on any machine fingerprints identically.
///
/// Every field is length-prefixed before hashing, so no two distinct inputs can
/// serialize to the same byte stream (a source containing the delimiter can't
/// be confused with a field boundary).
pub fn fingerprint(package_name: &str, havenc_version: &str, modules: &[MetaModule]) -> [u8; 32] {
    fn feed(h: &mut Sha256, bytes: &[u8]) {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    let mut sorted: Vec<&MetaModule> = modules.iter().collect();
    sorted.sort_by(|a, b| a.key.cmp(&b.key));

    let mut h = Sha256::new();
    feed(&mut h, package_name.as_bytes());
    feed(&mut h, havenc_version.as_bytes());
    feed(&mut h, &(sorted.len() as u64).to_le_bytes());
    for m in sorted {
        feed(&mut h, m.key.as_bytes());
        feed(&mut h, m.source.as_bytes());
    }
    h.finalize().into()
}

/// Why reading a `.hvmeta` failed.
#[derive(Debug)]
pub enum MetaError {
    Io(std::io::Error),
    /// The bytes are not a well-formed artifact (truncated, corrupt, or a wildly
    /// different layout).
    Decode(Box<bincode::ErrorKind>),
    /// The artifact's `format_version` is not the one this build understands.
    VersionMismatch { found: u32, expected: u32 },
}

impl std::fmt::Display for MetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaError::Io(e) => write!(f, "{}", e),
            MetaError::Decode(e) => write!(f, "malformed .hvmeta artifact: {}", e),
            MetaError::VersionMismatch { found, expected } => write!(
                f,
                ".hvmeta format version {} is not supported by this havenc (expects {}); \
                 recompile the library",
                found, expected
            ),
        }
    }
}

impl std::error::Error for MetaError {}

/// Serialize `meta` to `path` (conventionally `<output>.hvmeta`).
pub fn write(path: &Path, meta: &HavenMeta) -> Result<(), MetaError> {
    let bytes = bincode::serialize(meta).map_err(MetaError::Decode)?;
    std::fs::write(path, bytes).map_err(MetaError::Io)
}

/// Read and validate a `.hvmeta` from `path`, rejecting a format-version mismatch.
pub fn read(path: &Path) -> Result<HavenMeta, MetaError> {
    let bytes = std::fs::read(path).map_err(MetaError::Io)?;
    let meta: HavenMeta = bincode::deserialize(&bytes).map_err(MetaError::Decode)?;
    if meta.header.format_version != FORMAT_VERSION {
        return Err(MetaError::VersionMismatch {
            found: meta.header.format_version,
            expected: FORMAT_VERSION,
        });
    }
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(modules: Vec<MetaModule>) -> HavenMeta {
        let fp = fingerprint("foo", "0.1.0", &modules);
        HavenMeta {
            header: Header {
                format_version: FORMAT_VERSION,
                havenc_version: "0.1.0".into(),
                package_name: "foo".into(),
                fingerprint: fp,
            },
            modules,
        }
    }

    fn mods_a() -> Vec<MetaModule> {
        vec![
            MetaModule { key: "lib.hv".into(), source: "pub proc a() i32 { return 1; }".into(), is_root: true },
            MetaModule { key: "geo.hv".into(), source: "pub struct Point { x: i32 }".into(), is_root: false },
        ]
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hvmeta_rt_{}.hvmeta", std::process::id()));
        let meta = sample(mods_a());
        write(&path, &meta).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(meta, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_is_order_independent() {
        let mut reordered = mods_a();
        reordered.reverse();
        assert_eq!(
            fingerprint("foo", "0.1.0", &mods_a()),
            fingerprint("foo", "0.1.0", &reordered),
        );
    }

    #[test]
    fn fingerprint_changes_with_content() {
        let base = fingerprint("foo", "0.1.0", &mods_a());
        let mut changed = mods_a();
        changed[0].source.push_str(" // tweak");
        assert_ne!(base, fingerprint("foo", "0.1.0", &changed));
        // name and compiler version both participate
        assert_ne!(base, fingerprint("bar", "0.1.0", &mods_a()));
        assert_ne!(base, fingerprint("foo", "0.2.0", &mods_a()));
    }

    #[test]
    fn read_rejects_version_mismatch() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hvmeta_ver_{}.hvmeta", std::process::id()));
        let mut meta = sample(mods_a());
        meta.header.format_version = FORMAT_VERSION + 1;
        // write raw (bypassing the version check, which only `read` performs)
        std::fs::write(&path, bincode::serialize(&meta).unwrap()).unwrap();
        assert!(matches!(read(&path), Err(MetaError::VersionMismatch { .. })));
        let _ = std::fs::remove_file(&path);
    }
}
