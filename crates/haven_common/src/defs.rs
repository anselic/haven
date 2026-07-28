//! Definition identities.
//!
//! Every top-level item a program declares gets one [`DefId`] here, allocated
//! once while its module is being resolved. A `Def` records where the item came
//! from, what kind it is, and — crucially — how its emitted symbol name is
//! derived. [`Defs::symbol`] is the *only* place a symbol name is constructed.
//!
//! Definition identity is no longer a mangled string. A resolved named type is
//! `Type::Named { def, .. }`, a resolved variant reference is a `NameRef`, and
//! every table that has anything to say about a definition - its fields, its
//! members, its conformances, its instances - is keyed by `DefId`. Nothing
//! between name resolution and MIL lowering compares, constructs or parses a
//! name; mangling is load-bearing only for linking.
//!
//! MIL lowering is the seam. It asks for [`Defs::symbols`] once, and everything
//! downstream of it works in emitted names - which is all the backend wants,
//! since by then monomorphization has flattened every generic and no two types
//! can share a name anyway.
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
//! The `$` between slug and name is a separator only - nothing takes a symbol
//! apart to recover the item name any more. Diagnostics get the source spelling
//! from [`Def::source_name`] instead, which is why an instance now reads as
//! `alloc::<Vec2>` rather than being recovered by stripping everything before
//! the last `$`.

use std::collections::HashMap;
use std::path::Path;

use bumpalo::Bump;

use crate::ast::{FileId, GenericParam, Receiver, Span, Type};

/// The implicit prelude's canonical module key.
///
/// Shared rather than spelled twice because it is also how a *lang item* is
/// identified: a trait the compiler itself knows about (`Delete`) is the one
/// declared under this key, so a user's own `trait Delete` in their own module
/// is an ordinary trait and does not silently acquire destructor semantics.
pub const PRELUDE_KEY: &str = "std/prelude";

/// Identity of one top-level definition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct DefId(pub u32);

impl DefId {
    /// The identity a parser-produced node carries until name resolution fills
    /// it in. Resolution either overwrites every one of these or reports an
    /// unknown-name error, so it must never reach a later stage; [`Defs::get`]
    /// panics on it rather than returning a plausible-looking wrong answer.
    ///
    /// A sentinel rather than an `Option` because the alternative is unwrapping
    /// at every one of the ~40 sites that read a resolved id, all of which would
    /// be `expect`ing the same invariant.
    pub const UNRESOLVED: DefId = DefId(u32::MAX);

    pub fn is_resolved(self) -> bool { self != DefId::UNRESOLVED }
}

/// Identity of one loaded module.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefKind {
    Fn, Extern, Global, Struct, Enum, Trait,
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

/// The type constructor an `extend` block dispatches on.
///
/// A definition is its own head, so a struct and an enum are as distinct as two
/// structs. Everything else in the type grammar is *structural* and has no
/// definition to name: `[i32]`, `*Point` and `[u8; 4]` are built by applying a
/// constructor to other types, and there are infinitely many of them, so they
/// cannot each be given a `DefId`. They share the head of their constructor
/// instead - every slice is `Slice` - and the argument types are recovered by
/// unifying against the impl's written self type (see [`Member::self_ty`]).
///
/// A generic type's head is its *template*: `Vec<i32>` and `Vec<f32>` are both
/// `Def(Vec)`, which is what lets one `extend Vec<T>` answer for every instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TyHead {
    /// A struct, enum or enum-payload definition, generic template included.
    Def(DefId),
    Void, Bool,
    Int8, Int32, Int64,
    Uint8, Uint32, Uint64,
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
            Type::Int32 => TyHead::Int32,
            Type::Int64 => TyHead::Int64,
            Type::Uint8 => TyHead::Uint8,
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
            Type::Param(_) | Type::Path { .. } => return None,
        })
    }

    /// The head a built-in type *keyword* names, for a type-qualified path in
    /// expression position: the `i32` of `i32::from(x)`.
    ///
    /// In type position the parser has already mapped these names to a
    /// [`Type`], so [`Self::of`] suffices; in expression position a path is
    /// just a list of identifiers and nothing has looked at them yet. The list
    /// deliberately mirrors the scalar table in `parse_type` — these are
    /// keywords, so one name must not mean two things depending on where it is
    /// written.
    ///
    /// The structural heads have no entry: `[i32]` is not an identifier, so
    /// there is no path that could name one. Reaching a slice's associated
    /// functions would need a `<[i32]>::` form, which the grammar has no
    /// production for.
    pub fn of_builtin(name: &str) -> Option<Self> {
        Some(match name {
            "void" => TyHead::Void,
            "bool" => TyHead::Bool,
            "i8"   => TyHead::Int8,
            "i32"  => TyHead::Int32,
            "i64"  => TyHead::Int64,
            "u8"   => TyHead::Uint8,
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
            TyHead::Int8 => "i8", TyHead::Int32 => "i32", TyHead::Int64 => "i64",
            TyHead::Uint8 => "u8", TyHead::Uint32 => "u32", TyHead::Uint64 => "u64",
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
}

/// Methods and associated functions, keyed by `(receiver head, method name)`.
///
/// Replaces reconstructing `format!("{}${}", type_name, method)` and hoping it
/// lands on a real symbol. That only ever worked because `<slug>$Point` plus
/// `$area` happens to equal `<slug>$(Point$area)` — a coincidence of the mangling
/// scheme, and one that does not hold for enums or `@export`ed structs, whose
/// type names are not slug-prefixed while their methods' names are. Those cases
/// were silently unreachable outside the entry module.
///
/// Keyed by [`TyHead`] rather than by `DefId` so a method can hang off a type
/// that has no definition to key on - a primitive or a structural type. The
/// consequence is that the key is no longer exact: `extend [i32]` and
/// `extend [f32]` both want `(Slice, ...)`. Rather than make every value a
/// candidate list with an overlap check, one impl per `(head, method)` wins the
/// slot and a second is a duplicate-method error - which forbids specializing
/// `extend [T]` with an `extend [i32]`, exactly as Rust does without the
/// unstable `specialization` feature.
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
pub struct Instance {
    /// The generic template this specializes.
    pub template: DefId,
    /// How the instance reads back to a human: `alloc::<Vec2>`. Diagnostics only;
    /// never fed back into the compiler.
    pub display: String,
}

/// Every monomorphized instance in the program, keyed by its own identity.
pub type Instances = HashMap<DefId, Instance>;

pub struct ModInfo {
    pub file: FileId,
    /// Canonical key: an absolute path, or `std/...` (the prelude included -
    /// it is keyed `std/prelude` like any other embedded std module).
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
    instances: Instances,
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
    pub fn add_instance(&mut self, inst: DefId, template: DefId, display: String) {
        self.instances.insert(inst, Instance { template, display });
    }

    pub fn instances(&self) -> &Instances { &self.instances }

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

    pub fn get(&self, id: DefId) -> &Def<'a> {
        assert!(id.is_resolved(), "an unresolved DefId survived name resolution");
        &self.defs[id.0 as usize]
    }
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

/// Path-derived symbol prefix for a module.
///
/// * `std/dsp/osc`    -> `std.dsp.osc`
/// * a file under the entry's directory -> that relative path, dotted
/// * anything else (a path escaping the entry's tree, or a key that can't be
///   made relative) -> `m<hash>`, since there is no meaningful readable name and
///   an absolute path would leak the developer's home directory into the binary.
fn module_slug(key: &str, entry_dir: Option<&Path>) -> String {
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
