//! Definition identities.
//!
//! Every top-level item a program declares gets one [`DefId`] here, allocated
//! once while its module is being resolved. A `Def` records where the item came
//! from, what kind it is, and — crucially — how its emitted symbol name is
//! derived. [`Defs::symbol`] is the *only* place a symbol name is constructed.
//!
//! This is the first half of moving definition identity off of mangled strings.
//! Today the AST still carries names as `&str`, and those names still come from
//! `Defs::symbol`; later stages replace the strings in `Type`/`ExprNode` with
//! `DefId`s directly, at which point mangling stops being load-bearing for
//! anything except linking.
//!
//! ## symbol scheme
//!
//! An item that must keep a stable spelling — an `extern`'s C link name, an
//! `@export`ed item, anything in the entry module — is [`Linkage::Fixed`] and is
//! emitted verbatim. Everything else is [`Linkage::Mangled`] and comes out as
//! `<module slug>$<source name>`:
//!
//! ```text
//! std.math$square          std/math.hv
//! helpers.geo$Point        helpers/geo.hv, relative to the entry file
//! m7f3a1c02$Point          a module outside the entry's tree
//! ```
//!
//! The slug is derived from the module's *path*, not from load order. The old
//! scheme was `m{id}_{basename}` with `id` an enqueue index, so adding an
//! unrelated import renamed every symbol in the program — bad for `--shared` and
//! `--static-lib` ABI, and for reading `--emit-ir` diffs.
//!
//! The `$` between slug and name is deliberate: several downstream stages still
//! recover the item name with `rsplit('$')`, and dots inside the slug keep that
//! working while making the module part readable.

use std::collections::HashMap;
use std::path::Path;

use bumpalo::Bump;

use crate::ast::{FileId, Receiver, Span};

/// Identity of one top-level definition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct DefId(pub u32);

/// Identity of one loaded module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefKind { Fn, Extern, Global, Struct, Enum, Trait }

/// How a definition's emitted symbol name is chosen.
///
/// Replaces re-deriving the rule from attributes at three separate call sites,
/// where the three copies had drifted apart (only the fn one checked `extern`).
#[derive(Clone, Copy, Debug)]
pub enum Linkage<'a> {
    /// Emitted under exactly this name: an `extern`'s link symbol, an `@export`ed
    /// item, or anything in the entry module (whose names can't collide with the
    /// mangled imported ones, and whose diagnostics read better unmangled).
    Fixed(&'a str),
    /// Emitted as `<module slug>$<source name>`.
    Mangled,
}

pub struct Def<'a> {
    pub module: ModId,
    pub kind: DefKind,
    /// The name as written in source. For diagnostics and for building the
    /// symbol; never a lookup key once `DefId`s reach the AST.
    pub source_name: &'a str,
    pub is_pub: bool,
    pub linkage: Linkage<'a>,
    pub span: Span,
}

/// One method or associated function reachable through a type.
#[derive(Clone, Copy, Debug)]
pub struct Member<'a> {
    /// Emitted name of the function `lower_methods` desugared this into.
    pub name: &'a str,
    /// Whether it takes `self`, `*self`, or nothing. Recorded at desugaring, so
    /// resolution no longer has to guess by inspecting the first parameter.
    pub receiver: Receiver,
}

/// Methods and associated functions, keyed by `(type, method name)`.
///
/// Replaces reconstructing `format!("{}${}", type_name, method)` and hoping it
/// lands on a real symbol. That only ever worked because `<slug>$Point` plus
/// `$area` happens to equal `<slug>$(Point$area)` — a coincidence of the mangling
/// scheme, and one that does not hold for enums or `@export`ed structs, whose
/// type names are not slug-prefixed while their methods' names are. Those cases
/// were silently unreachable outside the entry module.
///
/// Keyed by *emitted type name* for now. Stage 4 re-keys it to `(DefId, &str)`,
/// which is a type change here and at the three lookup sites, nothing more.
pub type MemberTable<'a> = HashMap<(&'a str, &'a str), Member<'a>>;

pub struct ModInfo {
    pub file: FileId,
    /// Canonical key: an absolute path, `std/...`, or `<prelude>`.
    pub key: String,
    /// Path-derived, load-order-independent symbol prefix, e.g. `std.math`.
    pub slug: String,
    pub is_entry: bool,
}

#[derive(Default)]
pub struct Defs<'a> {
    defs: Vec<Def<'a>>,
    mods: Vec<ModInfo>,
    /// slugs already handed out, so two modules never share one.
    slugs: HashMap<String, ModId>,
    members: MemberTable<'a>,
}

impl<'a> Defs<'a> {
    pub fn new() -> Self { Self::default() }

    /// Register a module and compute its symbol slug. `entry_dir` is the
    /// directory holding the entry file, which user module paths are made
    /// relative to.
    pub fn add_module(&mut self, key: String, file: FileId, is_entry: bool,
                      entry_dir: Option<&Path>) -> ModId {
        let id = ModId(self.mods.len() as u32);
        let mut slug = module_slug(&key, entry_dir);
        // two distinct modules must never share a slug, or their symbols collide
        // at link time. disambiguate with a hash of the full key, which is stable
        // regardless of the order the two were loaded in.
        if self.slugs.contains_key(&slug) {
            slug = format!("{}_{:08x}", slug, fnv1a(&key));
        }
        self.slugs.insert(slug.clone(), id);
        self.mods.push(ModInfo { file, key, slug, is_entry });
        id
    }

    pub fn alloc(&mut self, def: Def<'a>) -> DefId {
        let id = DefId(self.defs.len() as u32);
        self.defs.push(def);
        id
    }

    /// Record a method or associated function on `ty`. Returns the previous
    /// entry, if the same `(type, name)` pair was already claimed.
    pub fn add_member(&mut self, ty: &'a str, name: &'a str, m: Member<'a>) -> Option<Member<'a>> {
        self.members.insert((ty, name), m)
    }

    pub fn members(&self) -> &MemberTable<'a> { &self.members }

    pub fn get(&self, id: DefId) -> &Def<'a> { &self.defs[id.0 as usize] }
    pub fn module(&self, id: ModId) -> &ModInfo { &self.mods[id.0 as usize] }
    pub fn len(&self) -> usize { self.defs.len() }
    pub fn is_empty(&self) -> bool { self.defs.is_empty() }

    /// The name this definition is emitted under. The single source of truth for
    /// mangling; nothing else may construct a symbol name.
    pub fn symbol(&self, id: DefId, arena: &'a Bump) -> &'a str {
        let def = self.get(id);
        match def.linkage {
            Linkage::Fixed(name) => name,
            Linkage::Mangled => {
                let slug = &self.module(def.module).slug;
                arena.alloc_str(&format!("{}${}", slug, def.source_name))
            }
        }
    }

    /// How to name this definition to a human: `std/math::square`. Uses the
    /// module's source key, not its slug, so it reads the way it was written.
    pub fn display(&self, id: DefId) -> String {
        let def = self.get(id);
        let m = self.module(def.module);
        if m.is_entry { def.source_name.to_string() }
        else { format!("{}::{}", m.key, def.source_name) }
    }
}

/// Path-derived symbol prefix for a module.
///
/// * `<prelude>`      -> `std.prelude`
/// * `std/dsp/osc`    -> `std.dsp.osc`
/// * a file under the entry's directory -> that relative path, dotted
/// * anything else (a path escaping the entry's tree, or a key that can't be
///   made relative) -> `m<hash>`, since there is no meaningful readable name and
///   an absolute path would leak the developer's home directory into the binary.
fn module_slug(key: &str, entry_dir: Option<&Path>) -> String {
    if key == "<prelude>" { return "std.prelude".to_string(); }
    if let Some(rest) = key.strip_prefix("std/") {
        return format!("std.{}", sanitize(rest));
    }
    if let Some(dir) = entry_dir {
        if let Ok(rel) = Path::new(key).strip_prefix(dir) {
            let rel = rel.to_string_lossy();
            if !rel.starts_with("..") {
                return sanitize(rel.trim_end_matches(".hv"));
            }
        }
    }
    format!("m{:08x}", fnv1a(key))
}

/// Turn a path fragment into an identifier-safe dotted slug: separators become
/// `.`, everything else that isn't alphanumeric becomes `_`.
fn sanitize(s: &str) -> String {
    s.trim_end_matches(".hv")
        .chars()
        .map(|c| match c {
            '/' | '\\' => '.',
            c if c.is_ascii_alphanumeric() || c == '.' => c,
            _ => '_',
        })
        .collect()
}

/// FNV-1a. Deterministic across runs, machines and Rust versions — unlike
/// `DefaultHasher`, whose output is explicitly not guaranteed stable.
fn fnv1a(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}
