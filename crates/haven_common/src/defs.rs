//! Definition identities and emitted symbol names.
//!
//! Name resolution assigns a [`DefId`] to every top-level item. Compiler tables
//! use that identity until MIL lowering converts definitions to emitted names.
//! [`Defs::symbol`] is the only place that constructs those names.
//!
//! ## symbol scheme
//!
//! Fixed-linkage items keep their source spelling. Other items use
//! `<module slug>$<source name>`:
//!
//! ```text
//! std.math$square          std/math.hv
//! foo$Point                src/main.hv        (the package root of package `foo`)
//! foo.geo$Point            src/geo.hv         (a submodule of package `foo`)
//! foo.dsp.osc$Osc          src/dsp/osc.hv
//! ```
//!
//! Slugs depend only on the package name and relative module path, keeping
//! symbols stable across machines and import order. Diagnostics use
//! [`Def::source_name`] instead of parsing mangled symbols.

use std::collections::HashMap;
use std::path::Path;

use bumpalo::Bump;

use crate::ast::{FileId, GenericArg, GenericParam, Receiver, Span, Type};

/// Names accepted by `@lang(...)`.
pub const LANG_ITEMS: &[&str] = &["delete"];

/// Compiler-known definitions resolved from the prelude.
/// Fields are `None` only when compiling without a prelude.
#[derive(Default)]
pub struct LangItems {
    /// `trait Delete` - marks a type as owning something, and names the
    /// destructor the ownership pass inserts calls to.
    pub delete: Option<DefId>,
}

/// Identity of one top-level definition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct DefId(pub u32);

impl DefId {
    /// Placeholder used before name resolution. [`Defs::get`] rejects it.
    pub const UNRESOLVED: DefId = DefId(u32::MAX);

    pub fn is_resolved(self) -> bool { self != DefId::UNRESOLVED }
}

/// Identity of one loaded module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefKind {
    Fn, Extern, Global, Struct, Enum, Trait,
    /// A `type Name = ...` alias. Has an identity so it can be imported, named
    /// through a qualifier and reported on like any other item - but it is never
    /// emitted: name resolution expands every use and drops the declaration.
    Alias,
    /// The synthetic struct holding one data variant's payload fields. Has no
    /// source declaration of its own: [`Defs::add_payload`] mints one per data
    /// variant, so a payload struct is a type identity like any other rather
    /// than a leaked `format!("{enum}${variant}")` string.
    EnumPayload,
}

/// How a definition's emitted symbol name is chosen.
///
/// Replaces re-deriving the rule from attributes at three separate call sites,
/// where the three copies had drifted apart (only the fn one checked `extern`).
#[derive(Clone, Copy, Debug)]
pub enum Linkage<'a> {
    /// Emitted under exactly this name: an `extern`'s link symbol or an
    /// `@export`ed item (the program's `main` reaches this through the
    /// `@export` the driver injects onto it).
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

/// The type constructor an `extend` block dispatches on.
///
/// Named types use their definition; structural types share a constructor head.
/// A generic instance uses its template definition, so one `extend Vec<T>` can
/// cover every `Vec` instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TyHead {
    /// A struct, enum or enum-payload definition, generic template included.
    Def(DefId),
    Void, Bool,
    Int8, Int16, Int32, Int64,
    Uint8, Uint16, Uint32, Uint64,
    Float32, Float64,
    Str,
    Pointer, Slice, Array, Simd, Function,
}

impl TyHead {
    /// The head `ty` dispatches on, or `None` for a type that cannot carry
    /// methods: a bare type parameter (whose methods come from its bounds, not
    /// from an impl) and an unresolved path (a compiler bug this far along).
    pub fn of(ty: &Type<'_>) -> Option<Self> {
        Some(match ty {
            Type::Named { def, .. } => TyHead::Def(*def),
            Type::Void => TyHead::Void,
            Type::Bool => TyHead::Bool,
            Type::Int8 => TyHead::Int8,
            Type::Int16 => TyHead::Int16,
            Type::Int32 => TyHead::Int32,
            Type::Int64 => TyHead::Int64,
            Type::Uint8 => TyHead::Uint8,
            Type::Uint16 => TyHead::Uint16,
            Type::Uint32 => TyHead::Uint32,
            Type::Uint64 => TyHead::Uint64,
            Type::Float32 => TyHead::Float32,
            Type::Float64 => TyHead::Float64,
            Type::Str => TyHead::Str,
            Type::Pointer(_) => TyHead::Pointer,
            Type::Slice(_) => TyHead::Slice,
            Type::Array(..) => TyHead::Array,
            Type::Simd(..) => TyHead::Simd,
            Type::Function { .. } => TyHead::Function,
            // `!` has no values, so it never dispatches a method and never
            // carries one - like a bare type parameter.
            Type::Param(_) | Type::Path { .. } | Type::Never => return None,
        })
    }

    /// Resolve a built-in type keyword used as a qualified expression path,
    /// such as the `i32` in `i32::from(x)`. Structural types have no keyword.
    pub fn of_builtin(name: &str) -> Option<Self> {
        Some(match name {
            "void" => TyHead::Void,
            "bool" => TyHead::Bool,
            "i8"   => TyHead::Int8,
            "i16"  => TyHead::Int16,
            "i32"  => TyHead::Int32,
            "i64"  => TyHead::Int64,
            "u8"   => TyHead::Uint8,
            "u16"  => TyHead::Uint16,
            "u32"  => TyHead::Uint32,
            "u64"  => TyHead::Uint64,
            "f32"  => TyHead::Float32,
            "f64"  => TyHead::Float64,
            "str"  => TyHead::Str,
            _ => return None,
        })
    }

    /// Whether a written path can name this head — what an associated function
    /// needs, since it has no receiver to be found through: `Point::new()`,
    /// `i32::zero()`.
    ///
    /// The structural heads cannot. A path is a list of identifiers and `[i32]`
    /// is not one, so reaching a slice's associated functions would need a
    /// `<[i32]>::` form the grammar has no production for. A *method* on a
    /// structural target is unaffected — it is found through its receiver.
    pub fn is_nameable(self) -> bool {
        !matches!(self,
            TyHead::Pointer | TyHead::Slice | TyHead::Array
            | TyHead::Simd | TyHead::Function)
    }

    /// An identifier-safe fragment naming this head, for building a desugared
    /// method's symbol. Not injective across definitions (every struct is
    /// `ty`), which is fine: the module resolver uniquifies within a module and
    /// the emitted name is never parsed back.
    pub fn tag(self) -> &'static str {
        match self {
            TyHead::Def(_) => "ty",
            TyHead::Void => "void", TyHead::Bool => "bool",
            TyHead::Int8 => "i8", TyHead::Int16 => "i16", TyHead::Int32 => "i32", TyHead::Int64 => "i64",
            TyHead::Uint8 => "u8", TyHead::Uint16 => "u16", TyHead::Uint32 => "u32", TyHead::Uint64 => "u64",
            TyHead::Float32 => "f32", TyHead::Float64 => "f64",
            TyHead::Str => "str",
            TyHead::Pointer => "ptr", TyHead::Slice => "slice",
            TyHead::Array => "array", TyHead::Simd => "simd",
            TyHead::Function => "proc",
        }
    }
}

/// One method or associated function reachable through a type.
#[derive(Clone, Debug)]
pub struct Member<'a> {
    /// Emitted name of the function `lower_methods` desugared this into. When
    /// [`Self::generics`] is non-empty this names a *template*, and the concrete
    /// instance is minted by monomorphization.
    pub name: &'a str,
    /// Whether it takes `self`, `*self`, or nothing. Recorded at desugaring, so
    /// resolution no longer has to guess by inspecting the first parameter.
    pub receiver: Receiver,
    /// The `extend` target this was declared on, resolved: `i32`, `[T]`,
    /// `Vec<T>`. The head alone is too coarse to dispatch on - every slice
    /// shares one - so a receiver reaches this method only if it *unifies* with
    /// this type, and unification is also what binds `generics` for the call.
    pub self_ty: Type<'a>,
    /// The impl block's own type parameters, inferred from the free names in
    /// `self_ty`. Empty for a target with no parameters (`i32`, `Point`), in
    /// which case unification degenerates to equality.
    pub generics: Vec<GenericParam<'a>>,
    /// Whether another module may reach this method. Module-private unless
    /// declared `pub`, like a top-level function - but methods live in one
    /// global table (so a receiver call resolves without an import), so scoping
    /// cannot enforce it: it is recorded here and checked at each call site. A
    /// trait-impl method is always public: reachability follows the trait, not a
    /// marker (as in Rust), so `lower_methods` folds trait membership into this.
    pub is_pub: bool,
    /// The module that declared this method. A private method is reachable only
    /// from calls in this same module; a cross-module call to it is an error.
    pub module: ModId,
}

/// Methods and associated functions, keyed by `(receiver head, method name)`.
///
/// [`TyHead`] supports methods on primitives and structural types, which have no
/// `DefId`. Only one method may occupy a `(head, name)` slot, so structural impls
/// cannot be specialized.
pub type MemberTable<'a> = HashMap<(TyHead, &'a str), Member<'a>>;

/// What a monomorphized instance came from.
///
/// Recorded by `mono` as it mints each instance, so nothing downstream has to
/// recover the relationship by taking the string apart. Two things needed it:
/// match patterns name the *template* (`std.option$Option::Some`) while the
/// scrutinee's type names the *instance* (`std.option$Option$i32`), because mono
/// has no type information with which to rewrite a bare pattern; and diagnostics
/// want to say `alloc::<Vec2>` rather than `std.alloc$alloc$Vec2`.
#[derive(Clone, Debug)]
pub struct Instance<'a> {
    /// The generic template this specializes.
    pub template: DefId,
    /// How the instance reads back to a human: `alloc::<Vec2>`. Diagnostics only;
    /// never fed back into the compiler.
    pub display: String,
    /// The arguments it was specialized with, in the template's parameter order.
    ///
    /// Unlike `display` these *are* fed back into the compiler: they are what
    /// [`deinstance`] needs to put a flat instance type back into template form,
    /// and dispatch keys on template form. `mono` kept its own copy of this for
    /// its internal use long before `Defs` did; the map lives here now because
    /// the passes that run *after* mono need it too and mono is gone by then.
    pub args: Vec<GenericArg<'a>>,
}

/// Every monomorphized instance in the program, keyed by its own identity.
pub type Instances<'a> = HashMap<DefId, Instance<'a>>;

/// Convert monomorphized instances back to template form for member and
/// conformance lookup. For example, `Vec$i32` becomes `Vec<i32>`.
pub fn deinstance<'a>(instances: &Instances<'a>, ty: &Type<'a>) -> Type<'a> {
    let go = |t: &Type<'a>| deinstance(instances, t);
    let arg = |a: &GenericArg<'a>| match a {
        GenericArg::Type(t) => GenericArg::Type(go(t)),
        other => other.clone(),
    };
    match ty {
        Type::Named { def, args } if args.is_empty() => match instances.get(def) {
            Some(i) => Type::Named { def: i.template, args: i.args.iter().map(arg).collect() },
            None => ty.clone(),
        },
        Type::Named { def, args } =>
            Type::Named { def: *def, args: args.iter().map(arg).collect() },
        Type::Pointer(inner) => Type::Pointer(Box::new(go(inner))),
        Type::Slice(inner) => Type::Slice(Box::new(go(inner))),
        Type::Array(inner, n) => Type::Array(Box::new(go(inner)), n.clone()),
        Type::Simd(inner, n) => Type::Simd(Box::new(go(inner)), n.clone()),
        other => other.clone(),
    }
}

/// Where a module's source came from.
///
/// Recorded at load rather than re-derived from the key, because the key only
/// half-answers the question: `std/...` is recognizable by its prefix, but a
/// dependency module's key (`foo/geo.hv`) looks like nothing in particular, so
/// a test written against the key shape reads "not std" as "the package being
/// compiled" and quietly counts a dependency's modules as the leaf's own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    /// Loaded from the source tree of the package being compiled.
    Own,
    /// The embedded standard library.
    Std,
    /// Loaded out of a dependency's `.hvmeta`.
    Dep,
}

pub struct ModInfo {
    pub file: FileId,
    /// Canonical key: an absolute path for a module of the package being
    /// compiled, `std/...` for an embedded std module (the prelude included -
    /// it is keyed `std/prelude` like any other), and `<dep>/<relative path>`
    /// for one that came out of a dependency's artifact.
    pub key: String,
    /// Path-derived, load-order-independent symbol prefix, e.g. `std.math`.
    pub slug: String,
    pub is_entry: bool,
    /// Whether this module belongs to the package being compiled, and if not,
    /// where it came from instead.
    pub origin: Origin,
}

#[derive(Default)]
pub struct Defs<'a> {
    defs: Vec<Def<'a>>,
    mods: Vec<ModInfo>,
    /// slugs already handed out, so two modules never share one.
    slugs: HashMap<String, ModId>,
    members: MemberTable<'a>,
    instances: Instances<'a>,
    /// `(enum, variant name) -> the synthetic struct holding that variant's
    /// payload fields`. Minted at resolution for declared enums and by `mono`
    /// for each instance it creates, so the mid end can look a payload struct up
    /// instead of rebuilding its name.
    payloads: HashMap<(DefId, &'a str), DefId>,
    /// Reverse index over the definitions whose symbol is fixed - which is every
    /// instance `mono` mints. Lets a stage that only has a symbol in hand (the
    /// alloc check works on MIL callee names) recover the definition and so its
    /// friendly spelling.
    by_symbol: HashMap<&'a str, DefId>,
    /// The prelude module, if this program has one.
    prelude: Option<ModId>,
    /// Compiler-known definitions, resolved from `prelude` at load time.
    lang: LangItems,
}

impl<'a> Defs<'a> {
    pub fn new() -> Self { Self::default() }

    /// Register a module and compute its symbol slug. `package` is the name of
    /// the package this module belongs to, and `root` is that package's root
    /// directory, which non-`std` module paths are made relative to. `is_entry`
    /// marks the package root module (whose slug is the bare package name).
    ///
    /// Fails only when a non-root, non-`std` module resolves outside the package
    /// root — the case the old absolute-path hash silently absorbed, now a
    /// diagnosable error rather than a symbol that bakes in a filesystem path.
    pub fn add_module(&mut self, package: &str, key: String, file: FileId,
                      is_entry: bool, root: Option<&Path>, origin: Origin)
                      -> Result<ModId, String> {
        let id = ModId(self.mods.len() as u32);
        let Some(mut slug) = module_slug(package, &key, root, is_entry) else {
            return Err(format!(
                "module '{}' is outside the root of package '{}'{}. A module must \
                 live under its package root; cross-package imports are not \
                 supported yet",
                key, package,
                root.map(|r| format!(" ('{}')", r.display())).unwrap_or_default()));
        };
        // two distinct modules must never share a slug, or their symbols collide
        // at link time. Package-relative slugs are distinct for distinct relative
        // paths, so this only fires when `sanitize` maps two different paths onto
        // one spelling; disambiguate with a hash of the *package-relative* key,
        // which is stable across machines and independent of load order (never
        // the absolute key, which would bake in a filesystem path).
        if self.slugs.contains_key(&slug) {
            slug = format!("{}_{:08x}", slug, fnv1a(&rel_seed(&key, root)));
        }
        self.slugs.insert(slug.clone(), id);
        self.mods.push(ModInfo { file, key, slug, is_entry, origin });
        Ok(id)
    }

    pub fn alloc(&mut self, def: Def<'a>) -> DefId {
        let id = DefId(self.defs.len() as u32);
        if let Linkage::Fixed(sym) = def.linkage { self.by_symbol.insert(sym, id); }
        self.defs.push(def);
        id
    }

    /// Record a method or associated function on the type headed by `head`.
    /// Returns the previous entry, if the same `(head, name)` pair was already
    /// claimed - which the caller reports as a duplicate, since one slot per
    /// pair is what keeps dispatch unambiguous without overlap checking.
    pub fn add_member(&mut self, head: TyHead, name: &'a str, m: Member<'a>) -> Option<Member<'a>> {
        self.members.insert((head, name), m)
    }

    pub fn members(&self) -> &MemberTable<'a> { &self.members }

    /// Mint the synthetic payload struct for one data variant of `enum_`. Its
    /// symbol is `<enum symbol>$<variant>`, which is why it is `Fixed`: it is
    /// derived from the enum's already-final symbol, not from the enum's source
    /// name plus its own module slug.
    pub fn add_payload(&mut self, enum_: DefId, variant: &'a str, symbol: &'a str) -> DefId {
        if let Some(&existing) = self.payloads.get(&(enum_, variant)) { return existing; }
        let def = self.get(enum_);
        let (module, span) = (def.module, def.span.clone());
        let id = self.alloc(Def {
            module,
            kind: DefKind::EnumPayload,
            source_name: variant,
            is_pub: true,
            linkage: Linkage::Fixed(symbol),
            span,
        });
        self.payloads.insert((enum_, variant), id);
        id
    }

    /// The payload struct of `enum_`'s `variant`, if that variant carries data.
    pub fn payload(&self, enum_: DefId, variant: &str) -> Option<DefId> {
        self.payloads.get(&(enum_, variant)).copied()
    }

    pub fn payloads(&self) -> &HashMap<(DefId, &'a str), DefId> { &self.payloads }

    /// Record that `inst` is `template` specialized to some arguments.
    /// Called by `mono` for every function, struct and enum instance it mints.
    pub fn add_instance(&mut self, inst: DefId, template: DefId, display: String,
                        args: Vec<GenericArg<'a>>) {
        self.instances.insert(inst, Instance { template, display, args });
    }

    pub fn instances(&self) -> &Instances<'a> { &self.instances }

    /// [`deinstance`] against this program's instances.
    pub fn deinstance(&self, ty: &Type<'a>) -> Type<'a> { deinstance(&self.instances, ty) }

    /// The template `id` specializes, if it is a monomorphized instance.
    ///
    /// Load-bearing for matching: a match pattern always names the *template*
    /// (`Option::Some`), because `mono` has no type information with which to
    /// rewrite a bare pattern to the instance it belongs to, while the
    /// scrutinee's type names the *instance* (`Option$i32`). Comparing through
    /// this link is what lets the two meet.
    pub fn template_of(&self, id: DefId) -> Option<DefId> {
        self.instances.get(&id).map(|i| i.template)
    }

    /// The friendliest spelling of an emitted symbol: the turbofish form if it
    /// names an instance, otherwise the symbol unchanged.
    pub fn show_symbol<'s>(&'s self, sym: &'s str) -> &'s str {
        self.by_symbol.get(sym)
            .and_then(|id| self.instances.get(id))
            .map_or(sym, |i| i.display.as_str())
    }

    /// How to name this definition to a human: `std/math::square` for an
    /// imported item, a bare `square` for one in the entry module, and an
    /// instance's turbofish form (`alloc::<Vec2>`) for anything `mono` minted.
    pub fn show(&self, id: DefId) -> String {
        if let Some(i) = self.instances.get(&id) { return i.display.clone(); }
        self.display(id)
    }

    /// Record the prelude and the lang items resolved from it. Called once, by
    /// the module loader; taking both together is what keeps the invariant that
    /// a lang item can only be missing when the prelude itself is.
    pub fn set_lang_items(&mut self, prelude: ModId, items: LangItems) {
        self.prelude = Some(prelude);
        self.lang = items;
    }

    /// The prelude module, or `None` under `--no-prelude`.
    pub fn prelude(&self) -> Option<ModId> { self.prelude }

    /// The compiler-known definitions. Every field is `None` exactly when
    /// [`Self::prelude`] is.
    pub fn lang(&self) -> &LangItems { &self.lang }

    pub fn get(&self, id: DefId) -> &Def<'a> {
        assert!(id.is_resolved(), "an unresolved DefId survived name resolution");
        &self.defs[id.0 as usize]
    }
    pub fn module(&self, id: ModId) -> &ModInfo { &self.mods[id.0 as usize] }
    /// Every loaded module, in load order. Lets a stage that needs the module set
    /// itself — e.g. writing a lib's `.hvmeta` from each own module's key +
    /// source — enumerate them without a private-field accessor per use.
    pub fn modules(&self) -> &[ModInfo] { &self.mods }
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

    /// Every definition's emitted symbol, computed once.
    ///
    /// The mid end hands this to MIL lowering, which is the seam where identity
    /// stops mattering and linkage starts: everything downstream of it works in
    /// symbol names, so nothing downstream needs `Defs` at all.
    pub fn symbols(&self, arena: &'a Bump) -> HashMap<DefId, &'a str> {
        (0..self.defs.len() as u32)
            .map(|i| (DefId(i), self.symbol(DefId(i), arena)))
            .collect()
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

/// Package-anchored symbol prefix for a module: a pure function of the package
/// name and the module's path relative to the package root, so no absolute path
/// ever reaches a symbol.
///
/// * `std/dsp/osc`               -> `std.dsp.osc`   (`std` is the package `std`)
/// * the package root (`is_root`) -> `foo`          (the bare package name)
/// * `foo`'s `src/geo.hv`        -> `foo.geo`       (`<package>.<relpath>`)
///
/// `None` when a non-root, non-`std` key escapes the package root. The old
/// scheme hashed such a key's absolute path (`m<hash>`); that baked the
/// developer's home directory into the binary and is now a caller-diagnosed
/// error instead.
fn module_slug(package: &str, key: &str, root: Option<&Path>, is_root: bool) -> Option<String> {
    // `std` is just the package named `std`, anchored at the embedded tree root.
    if let Some(rest) = key.strip_prefix("std/") {
        return Some(format!("std.{}", sanitize(rest)));
    }
    // the package name is sanitized like a path fragment, so a manifest name
    // with spaces or punctuation still yields a valid identifier prefix.
    let package = sanitize(package);
    // the package root module carries the bare package name, matching Rust's
    // crate-root convention (`crate::thing`, not `crate::main::thing`).
    if is_root {
        return Some(package);
    }
    // a submodule: `<package>.<path relative to the package root>`.
    let rel = Path::new(key).strip_prefix(root?).ok()?;
    let rel = rel.to_string_lossy();
    if rel.starts_with("..") { return None; }
    Some(format!("{}.{}", package, sanitize(rel.trim_end_matches(".hv"))))
}

/// The location-independent seed used to disambiguate a slug collision: a
/// module key's path *within its package* (`std/`'s tail, or the path relative
/// to the package root), never the absolute key. Two keys reach this only when
/// they slug alike, and their in-package paths still differ, so hashing this
/// keeps their symbols apart while staying identical across machines.
fn rel_seed(key: &str, root: Option<&Path>) -> String {
    if let Some(rest) = key.strip_prefix("std/") { return rest.to_string(); }
    if let Some(root) = root {
        if let Ok(rel) = Path::new(key).strip_prefix(root) {
            return rel.to_string_lossy().into_owned();
        }
    }
    key.to_string()
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
