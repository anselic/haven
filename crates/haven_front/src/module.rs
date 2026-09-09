//! Load modules, resolve names, and merge them into one program.
//!
//! Imports are loaded transitively relative to the importing file. Each module
//! receives a package-based symbol prefix, references are resolved through its
//! imports, and the resulting items are merged into a flat AST.
//!
//! ## import forms
//!
//! * `import std/math`            - whole module, symbols visible only as
//!   `math::sinf` (qualified under the last path segment).
//! * `import std/math { sinf }`   - selective, the named symbols visible
//!   unqualified as `sinf`.
//!
//! The `@!prelude` module is imported implicitly into user modules. A dependency
//! may provide it; otherwise the standard prelude is used.
//!
//! ## visibility
//!
//! Top-level items are private unless marked `pub`. `@export` controls linker
//! visibility separately.
//!
//! ## unresolved names
//!
//! Unknown names are diagnosed here. Intrinsics, `Self`, and generic parameters
//! are the only names resolved outside a module symbol table.
//!
//! ## known v1 limitations
//!
//! * Visibility is item-level only; struct fields are always public.

use std::collections::{HashMap, HashSet, VecDeque};
// `Path` is `ast::Path` here - a `::`-separated name. The filesystem one is
// aliased, since this module talks about names far more often than files.
use std::path::{Path as FilePath, PathBuf};

use bumpalo::Bump;
use haven_meta::HavenMeta;

use haven_common::ast::*;
use crate::parse;
use haven_common::defs::{Def, DefKind, DefId, Defs, Linkage, Member, MemberTable, ModId, Origin, TyHead};

use haven_common::diag::{self, Files};
use haven_common::intrinsics::Intrinsic;

use haven_common::defs::LangItems;

/// A loaded module: parsed contents plus the bookkeeping the resolver needs to
/// mangle and rewrite it
struct Module<'a> {
    /// this module's entry in the `Files` table: its spans point at this id, and
    /// its canonical key (absolute path, or `std/...`) and source text are
    /// stored there rather than duplicated here.
    file: FileId,
    /// this module's entry in `Defs`, which owns its symbol slug.
    mid: ModId,
    /// whether this module belongs to the package supplying the prelude. The
    /// predicate for `@!prelude` and `@lang`, which only that package may make
    /// good on - see [`PreludeSource`].
    prelude_pkg: bool,
    /// where this module's source came from. Kept so the post-load `@!prelude`
    /// scan can tell a *leaf* mark (a mistake to name) from a *dependency's* one
    /// (inert - a library is just a library to its consumer).
    origin: Origin,
    /// the `@!name` marks written at this module's file scope. Statements about
    /// the file rather than about any of its declarations, so they are kept
    /// here rather than on an item.
    mod_attrs: Vec<Attribute<'a>>,
    imports: Vec<Import<'a>>,
    /// what each import resolved to (parallel to `imports`), or `None` if it
    /// failed to resolve (error already recorded).
    import_keys: Vec<Option<ImportTarget<'a>>>,
    items: Vec<TopLevel<'a>>,
    /// `extend Target: Trait` conformance records from this module, in source
    /// (pre-mangling) names; remapped to final names after scopes are built.
    impls: Vec<RawImpl<'a>>,
    /// every method this module declares, in source names; likewise remapped
    /// after scopes are built, into `Defs`'s member table.
    methods: Vec<RawMethod<'a>>,
}

/// One entry in a module's symbol table: the final (post-mangling) emitted name
/// plus whether the item is `pub` (importable by other modules). A module always
/// sees all of its own symbols; the `is_pub` flag only gates *cross-module*
/// imports.
#[derive(Clone, Copy)]
struct Sym<'a> {
    /// The symbol it is emitted under.
    name: &'a str,
    /// What it *is*. Two modules may each export a `Buf`; these differ even
    /// when — for an `@export`ed or entry-module item — the names do not.
    def: DefId,
    is_pub: bool,
}

/// The final (post-mangling) name a module exposes per symbol, split by namespace
/// so call sites and type positions look in the right place.
#[derive(Default)]
struct SymTab<'a> {
    /// Callable names: functions and externs. `sym -> (final name, is_pub)`.
    fns: HashMap<&'a str, Sym<'a>>,
    /// Struct type names. `sym -> (final name, is_pub)`.
    structs: HashMap<&'a str, Sym<'a>>,
}

fn is_export(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.value.name == "export")
}

/// Choose fixed or mangled linkage for an item.
///
/// * `extern` - the name *is* the C link symbol.
/// * `@export` - a host looks the symbol up by that name.
///
/// Other items use the module namespace. The driver marks `main` as exported
/// before this runs.
fn linkage_of<'a>(name: &'a str, attrs: &[Attribute], is_extern: bool) -> Linkage<'a> {
    if is_extern || is_export(attrs) {
        Linkage::Fixed(name)
    } else {
        Linkage::Mangled
    }
}

/// One member of a directory namespace: the canonical key of a module file, and
/// the path that names it *inside* the namespace - `["osc"]` for
/// `std/dsp/osc.hv` imported via `import std/dsp`, `["a", "b"]` for a file one
/// directory deeper.
struct DirMember<'a> {
    segments: Vec<&'a str>,
    key: String,
}

/// What an import resolved to.
enum ImportTarget<'a> {
    /// A single module file, under its canonical key.
    Module(String),
    /// A directory. Its files become an implicit namespace: `import std/dsp`
    /// binds the qualifier `dsp`, whose members are the modules below it, so
    /// `dsp::osc::Osc` resolves segment by segment.
    ///
    /// Every member is loaded, not just the ones a path happens to mention -
    /// resolution needs the whole namespace present before it can walk into it.
    Dir(Vec<DirMember<'a>>),
}

/// Every `.hv` file under the on-disk directory `root`, as `(segments below
/// `root`, canonical key)`. Mirrors [`std_dir_members`] for user modules.
fn dir_members<'a>(root: &FilePath, arena: &'a Bump) -> Result<Vec<DirMember<'a>>, String> {
    fn walk<'a>(dir: &FilePath, prefix: &[&'a str], arena: &'a Bump,
                out: &mut Vec<DirMember<'a>>) -> Result<(), String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read module directory '{}': {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("cannot read module directory '{}': {}", dir.display(), e))?;
            let path = entry.path();
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            if path.is_dir() {
                let mut prefix = prefix.to_vec();
                prefix.push(arena.alloc_str(name));
                walk(&path, &prefix, arena, out)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some("hv") {
                let canon = std::fs::canonicalize(&path)
                    .map_err(|_| format!("cannot find module file '{}'", path.display()))?;
                let mut segments = prefix.to_vec();
                segments.push(arena.alloc_str(name));
                out.push(DirMember { segments, key: canon.to_string_lossy().into_owned() });
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, &[], arena, &mut out)?;
    Ok(out)
}

/// The leading path segment that anchors an import at its own package's root
/// rather than at the importing module's directory.
///
/// Reserved, so it always means this and never a package or module of that name.
/// An ordinary import reaches only *downward* - the segments are pushed onto the
/// importing module's own directory - which leaves a nested module unable to name
/// anything above it. `std/...` has always been that anchor for the embedded
/// tree; this is the same thing for every other package.
pub const SELF_SEG: &str = "self";

/// The message for a bare `import self`, which names no module. Shared, because
/// on-disk and dependency-internal resolution both have to reject it.
const BARE_SELF: &str = "`import self` names no module: `self` is the package \
                         root, so it needs a module under it - `import self/math`";

/// Resolve an import to the module - or directory of modules - it names. No file
/// read for `std`; for the relative case canonicalizes (so the path has to
/// exist). `dir` is the importing module's directory, `root` its package's.
fn resolve_target<'a>(imp: &Import, dir: Option<&FilePath>, root: Option<&FilePath>,
                      arena: &'a Bump)
    -> Result<ImportTarget<'a>, String>
{
    if imp.path.first() == Some(&"std") {
        // The compiler embeds no std, so an `import std/...` that reaches here has
        // no std bound: a bound `std` dep is claimed earlier, in `load_and_merge`,
        // before this. So this is the "no std at all" case, reported plainly rather
        // than as a mysterious missing file.
        Err(format!(
            "cannot resolve '{}': no `std` is available. The compiler embeds none - \
             it discovers `std.hvmeta` on disk (via $HAVEN_STD, or beside the \
             `havenc` binary), or you bind one with `--dep std=<path>.hvmeta`",
            imp.path.join("/")))
    } else {
        // `self/...` measures from the package root, everything else from this
        // module's own directory.
        let anchored = imp.path.first() == Some(&SELF_SEG);
        let segs = if anchored { &imp.path[1..] } else { &imp.path[..] };
        if anchored && segs.is_empty() { return Err(BARE_SELF.to_string()); }
        let dir = (if anchored { root } else { dir }).ok_or_else(||
            "cannot resolve a relative import from this module (std/prelude modules may only import `std/...`)".to_string())?;
        let mut base = dir.to_path_buf();
        for seg in segs { base.push(seg); }
        let mut file = base.clone();
        file.set_extension("hv");
        if let Ok(canon) = std::fs::canonicalize(&file) {
            return Ok(ImportTarget::Module(canon.to_string_lossy().into_owned()));
        }
        if base.is_dir() {
            let members = dir_members(&base, arena)?;
            if members.is_empty() {
                return Err(format!("module directory '{}' contains no modules", base.display()));
            }
            return Ok(ImportTarget::Dir(members));
        }
        Err(format!("cannot find module file '{}'", file.display()))
    }
}

/// Load one on-disk module's source + the dir its own relative imports resolve
/// against. Assumes `resolve_target` already succeeded for this key, so the key is
/// the canonical absolute path. (`std/...` never reaches here: a bound `std` is a
/// dep, and an unbound one errors in `resolve_target`.)
fn load_import<'a>(_imp: &Import, key: &str, dir: Option<&FilePath>, arena: &'a Bump) -> Result<(&'a str, Option<PathBuf>), String> {
    let _ = dir; // key is already the canonical absolute path
    let canon = PathBuf::from(key);
    let src = std::fs::read_to_string(&canon)
        .map_err(|e| format!("cannot read module file '{}': {}", canon.display(), e))?;
    Ok((arena.alloc_str(&src), canon.parent().map(|d| d.to_path_buf())))
}

/// The worklist/`seen`/`Files` key for a dependency module: the dep name followed
/// by the module's package-root-relative key from the artifact — `foo` +
/// `dsp/osc.hv` -> `foo/dsp/osc.hv`. Prefixing with the dep name keeps it from
/// ever colliding with the leaf's absolute-path keys or a `std/...` key, and — fed
/// to [`Defs::add_module`] with the dep name as package and root — makes
/// `module_slug` re-derive the dep's own `foo.dsp.osc` slug, byte-identical to
/// what the library emitted standalone.
fn dep_key(pkg: &str, relkey: &str) -> String {
    format!("{}/{}", pkg, relkey)
}

/// Resolve an import *within* a dependency's own module set (its `.hvmeta`
/// source), mirroring the on-disk relative resolution in [`resolve_target`] but
/// against the artifact instead of the filesystem. `segs` is the target module
/// path relative to the dep root: `["geo"]`, `["dsp", "osc"]`, or empty for a
/// bare `import <dep>` (which binds the dep's root module).
///
/// This is the same "load a package's source under a namespace" path std uses,
/// with two differences the caller supplies: the source is the artifact's, and
/// the namespace is the dep's package name rather than `std`.
fn resolve_within_dep<'a>(meta: &HavenMeta, pkg: &str, segs: &[&str], arena: &'a Bump)
    -> Result<ImportTarget<'a>, String>
{
    // bare `import foo` binds the dep's root module (its `is_root` one).
    if segs.is_empty() {
        let root = meta.modules.iter().find(|m| m.is_root).ok_or_else(||
            format!("dependency '{}' has no root module to import", pkg))?;
        return Ok(ImportTarget::Module(dep_key(pkg, &root.key)));
    }
    let rel = segs.join("/");
    let file = format!("{}.hv", rel);
    if meta.modules.iter().any(|m| m.key == file) {
        return Ok(ImportTarget::Module(dep_key(pkg, &file)));
    }
    // not a module file — a directory of them? every member below `rel/` becomes
    // an implicit namespace, exactly as a std/on-disk directory import does.
    let prefix = format!("{}/", rel);
    let members: Vec<DirMember<'a>> = meta.modules.iter().filter_map(|m| {
        let sub = m.key.strip_prefix(&prefix)?;
        let stem = sub.strip_suffix(".hv").unwrap_or(sub);
        let segments = stem.split('/').map(|s| &*arena.alloc_str(s)).collect();
        Some(DirMember { segments, key: dep_key(pkg, &m.key) })
    }).collect();
    if !members.is_empty() {
        return Ok(ImportTarget::Dir(members));
    }
    Err(format!("package '{}' has no module '{}'", pkg, rel))
}

/// A raw (pre-mangling) method record, collected while desugaring an `extend`
/// block. `target`/`name` are as written; `fn_name` is the function
/// `lower_methods` synthesized for it. `load_and_merge` maps all three to final
/// names once scopes exist, and records the result in `Defs`'s member table -
/// which is what makes a method reachable without reconstructing its name from
/// its type's.
struct RawMethod<'a> {
    /// The `extend` target as written (unresolved). Resolved through the
    /// module's scopes in pass 1.5, which is where its head is computed.
    target: Type<'a>,
    /// The target's inferred type parameters, which are also the leading
    /// generics of `fn_name`.
    generics: Vec<GenericParam<'a>>,
    /// The block's `where` clause, as written. Merged onto `generics` in
    /// `load_and_merge`, once inference has said what the parameters are.
    where_bounds: Vec<GenericParam<'a>>,
    name: &'a str,
    fn_name: &'a str,
    receiver: Receiver,
    /// Effective visibility: `pub` as written, or `true` unconditionally when the
    /// method implements a trait (a trait-impl method's reachability follows the
    /// trait, not an explicit marker). See [`Member::is_pub`].
    is_pub: bool,
    /// The trait this was inherited from, when it is a copied default body rather
    /// than a method anyone wrote here. Only a diagnostic needs it: a duplicate
    /// otherwise points at an `extend` block in which the offending method does
    /// not appear.
    default_of: Option<&'a str>,
    span: Span,
}

/// A raw (pre-mangling) `extend Target: Trait` conformance record, collected
/// while desugaring. `target`/`trait_` are as written; `load_and_merge` resolves
/// them through the module's scopes before handing them to the typechecker.
struct RawImpl<'a> {
    target: Type<'a>,
    generics: Vec<GenericParam<'a>>,
    where_bounds: Vec<GenericParam<'a>>,
    trait_: &'a str,
    /// `type Item = Ty;` bindings, still unresolved: resolved in `load_and_merge`
    /// through the module's scopes with the impl's inferred parameters in scope.
    assoc_bindings: Vec<(&'a str, Type<'a>)>,
    /// The names this block defined itself. What a trait's *default* bodies are
    /// checked against: a default is copied in only where the impl was silent.
    defined: Vec<&'a str>,
    span: Span,
}

/// Resolve the trait names in an impl's generic parameters, for the copy of them
/// stored in the member table.
///
/// The member record is built in pass 1.5, which runs *before* the rewriting
/// pass that resolves bounds on ordinary items — so a straight clone would carry
/// `DefId::UNRESOLVED` in every bound, and a dispatch-site bound check would then
/// reject the very impls it should admit. This does the same lookup
/// [`Rewriter::bounds`] does, at the point the copy is taken.
///
/// An unknown trait is left unresolved rather than reported: the desugared
/// function carries the same bound and pass 2 diagnoses it there, once.
fn resolve_bound_defs<'a>(generics: &[GenericParam<'a>], scopes: &Scopes<'a>) -> Vec<GenericParam<'a>> {
    generics.iter().map(|g| match g {
        GenericParam::Type { name, bounds } => GenericParam::Type {
            name,
            bounds: bounds.iter().map(|b| {
                let mut b = b.clone();
                if let Some(sym) = scopes.types.get(b.path.last()) { b.def = sym.def; }
                b
            }).collect(),
        },
        other => other.clone(),
    }).collect()
}

/// Whichever binder a `where` clause is attached to. Only the diagnostic differs:
/// an `extend` block's parameters are *inferred from its target*, so an unbound
/// name means the target does not mention it, while a `proc` writes its
/// parameters out and an unbound name means they simply do not include it.
enum WhereOwner<'x, 'a> {
    Extend(&'x Type<'a>),
    Proc(&'x str),
}

/// Apply a `where` clause to the parameters it names. Unknown parameters are
/// reported once; `reported` suppresses duplicates from desugared methods.
fn apply_where_bounds<'a>(
    generics: &mut [GenericParam<'a>],
    where_bounds: &[GenericParam<'a>],
    owner: WhereOwner<'_, 'a>,
    span: &Span,
    reported: &mut HashSet<(usize, &'a str)>,
    errs: &mut Vec<Error>,
) {
    for wb in where_bounds {
        let GenericParam::Type { name: wname, bounds } = wb else { continue };
        let found = generics.iter_mut().find(|g| matches!(g,
            GenericParam::Type { name, .. } if name == wname));
        match found {
            // the same parameter may be named by more than one clause; the
            // bounds accumulate, exactly as `T: A + B` would.
            Some(GenericParam::Type { bounds: existing, .. }) => {
                for b in bounds {
                    if !existing.iter().any(|e| e.path == b.path) { existing.push(b.clone()); }
                }
            }
            _ => {
                if reported.insert((span.start, wname)) {
                    let (label, note) = match &owner {
                        WhereOwner::Extend(target) => (
                            format!("`extend {}` does not bind '{}'", target, wname),
                            format!(
                                "an `extend` block's type parameters are inferred from its \
                                 target, so only a name appearing inside `{}` can be bounded \
                                 here",
                                target)),
                        WhereOwner::Proc(pname) => (
                            format!("`proc {}` declares no type parameter '{}'", pname, wname),
                            format!(
                                "a `where` clause bounds a parameter the binder already \
                                 declares; write `proc {}<{}>(..)` if it was meant to be one",
                                pname, wname)),
                    };
                    errs.push(Error::new(span.clone(),
                        format!("unbound type parameter '{}' in `where`", wname))
                        .with_label(span.clone(), label)
                        .with_note(note));
                }
            }
        }
    }
}

/// The declared generic parameters of every struct and enum in the program,
/// keyed by definition.
///
/// [`impl_generics`] cannot do without it. In `extend Buf<T, N>` both arguments
/// are bare identifiers, and the grammar offers no way to mark one as a value -
/// so only the *declaration* of `Buf` says which slot is a `const`. Lacking it,
/// `N` came out a type parameter, and every later use of it (as an array length,
/// as a value in a method body) was nonsense.
type TypeParams<'a> = HashMap<DefId, Vec<GenericParam<'a>>>;

/// Resolve a written type path to the definition it names, through one module's
/// scopes.
///
/// A free function rather than a [`Rewriter`] method because pass 1.25 runs
/// before any rewriter exists. It mirrors `Rewriter::type_head`'s lookup minus
/// the diagnostics: a path that resolves to nothing is left alone here, pass 2
/// being where an unknown type is reported.
fn type_def_of<'a>(scopes: &Scopes<'a>, path: &Path<'a>) -> Option<DefId> {
    if let Some(one) = path.as_single() {
        return scopes.types.get(one).map(|s| s.def);
    }
    // `<qualifier...>::Type`, the qualifier possibly several segments deep.
    let segs = &path.segments;
    let mut cur = scopes.quals.get(segs[0])?;
    let mut n = 1;
    while n + 1 < segs.len() {
        match cur.children.get(segs[n]) {
            Some(next) => { cur = next; n += 1; }
            None => break,
        }
    }
    if segs.len() - n != 1 { return None; }
    cur.types.get(segs[n]).map(|s| s.def)
}

/// A type alias, with its body already resolved in the module that declared it.
///
/// Resolving up front rather than at each use is what makes an alias exportable:
/// `pub type Chain = Serial<Gain, OnePole>` has to mean the same thing in a
/// module that imported neither `Serial` nor `Gain`, and it can only do that if
/// the names in its body were looked up where they were written.
struct AliasDef<'a> {
    generics: Vec<GenericParam<'a>>,
    ty: Type<'a>,
    /// where the alias was declared, for an arity error at a use site that has no
    /// better span of its own.
    span: Span,
}

type Aliases<'a> = HashMap<DefId, AliasDef<'a>>;

/// Substitute an alias' parameters throughout its (already resolved) body.
fn subst_alias<'a>(
    ty: &Type<'a>,
    types: &HashMap<&'a str, Type<'a>>,
    consts: &HashMap<&'a str, ConstVal<'a>>,
) -> Type<'a> {
    let cv = |c: &ConstVal<'a>| match c {
        ConstVal::Param(n) => consts.get(n).cloned().unwrap_or_else(|| c.clone()),
        ConstVal::Lit(_) => c.clone(),
    };
    let go = |t: &Type<'a>| subst_alias(t, types, consts);
    match ty {
        Type::Param(n) => types.get(n).cloned().unwrap_or_else(|| ty.clone()),
        Type::Named { def, args } => Type::Named {
            def: *def,
            args: args.iter().map(|a| match a {
                GenericArg::Type(t) => GenericArg::Type(go(t)),
                GenericArg::Const(c) => GenericArg::Const(cv(c)),
            }).collect(),
        },
        Type::Pointer(inner) => Type::Pointer(Box::new(go(inner))),
        Type::Slice(inner) => Type::Slice(Box::new(go(inner))),
        Type::Array(inner, n) => Type::Array(Box::new(go(inner)), cv(n)),
        Type::Simd(inner, n) => Type::Simd(Box::new(go(inner)), cv(n)),
        Type::Function { params, return_type } => Type::Function {
            params: params.iter().map(go).collect(),
            return_type: Box::new(go(return_type)),
        },
        _ => ty.clone(),
    }
}

/// Whether every alias `ty` names has already been resolved, so resolving `ty`
/// itself would expand all of them.
///
/// Walks the type as *written*, since that is all that exists before resolution:
/// each path head is looked up the same way [`type_def_of`] will look it up.
fn alias_deps_ready<'a>(
    ty: &Type<'a>,
    scopes: &Scopes<'a>,
    declared: &HashSet<DefId>,
    resolved: &Aliases<'a>,
) -> bool {
    let ready = |t: &Type<'a>| alias_deps_ready(t, scopes, declared, resolved);
    match ty {
        Type::Path { path, args } => {
            if let Some(def) = type_def_of(scopes, path) {
                if declared.contains(&def) && !resolved.contains_key(&def) { return false; }
            }
            args.iter().all(|a| match a {
                GenericArg::Type(t) => ready(t),
                GenericArg::Const(_) => true,
            })
        }
        Type::Pointer(inner) | Type::Slice(inner)
        | Type::Array(inner, _) | Type::Simd(inner, _) => ready(inner),
        Type::Function { params, return_type } =>
            params.iter().all(&ready) && ready(return_type),
        _ => true,
    }
}

/// Infer an `extend` target's type and const parameters in first-use order.
/// A name is a parameter when it is:
///
///   * a single-segment, argument-less path (`T`, never `geo::Point` or
///     `Vec<T>`),
///   * a *proper subterm* of the target, and
///   * not a type in scope, per `known`.
///
/// Bare targets are always treated as named types, so blanket impls are not
/// supported. `decl` identifies const argument positions when available.
fn impl_generics<'a, 'd>(
    target: &Type<'a>,
    known: &dyn Fn(&str) -> bool,
    decl: &dyn Fn(&Path<'a>) -> Option<&'d [GenericParam<'a>]>,
) -> Vec<GenericParam<'a>> {
    fn push_ty<'a>(n: &'a str, out: &mut Vec<GenericParam<'a>>) {
        if !out.iter().any(|g| matches!(g, GenericParam::Type { name, .. } if *name == n)) {
            out.push(GenericParam::Type { name: n, bounds: Vec::new() });
        }
    }
    fn push_const<'a>(n: &'a str, ty: Type<'a>, out: &mut Vec<GenericParam<'a>>) {
        if !out.iter().any(|g| matches!(g, GenericParam::Const(name, _) if *name == n)) {
            out.push(GenericParam::Const(n, ty));
        }
    }
    /// One argument of a type path, classified against the parameter it fills.
    fn arg<'a, 'd>(
        a: &GenericArg<'a>,
        slot: Option<&GenericParam<'a>>,
        known: &dyn Fn(&str) -> bool,
        decl: &dyn Fn(&Path<'a>) -> Option<&'d [GenericParam<'a>]>,
        out: &mut Vec<GenericParam<'a>>,
    ) {
        // a bare identifier parses as a *type* argument whatever it was meant to
        // be, so a `const` slot is the only thing that can tell `N` from `T`. the
        // declared type rides along with it: `const N: u64` rebuilt as a `u32`
        // would stop the method's `[T; N]` matching the struct's field.
        if let Some(GenericParam::Const(_, cty)) = slot {
            let name = match a {
                GenericArg::Const(ConstVal::Param(n)) => Some(*n),
                GenericArg::Type(Type::Path { path, args }) if args.is_empty() =>
                    path.as_single().filter(|n| !known(n)),
                _ => None,
            };
            if let Some(n) = name {
                push_const(n, cty.clone(), out);
                return;
            }
        }
        match a {
            GenericArg::Type(t) => walk(t, known, decl, out),
            // no slot to consult (an unresolvable head, or a const written where a
            // type belongs - pass 2 and typecheck report those). `u32` matches how
            // the parser types a bare array length.
            GenericArg::Const(ConstVal::Param(n)) => push_const(n, Type::Uint32, out),
            GenericArg::Const(ConstVal::Lit(_)) => {}
        }
    }
    fn walk<'a, 'd>(
        ty: &Type<'a>,
        known: &dyn Fn(&str) -> bool,
        decl: &dyn Fn(&Path<'a>) -> Option<&'d [GenericParam<'a>]>,
        out: &mut Vec<GenericParam<'a>>,
    ) {
        match ty {
            Type::Path { path, args } => {
                match path.as_single() {
                    Some(one) if args.is_empty() && !known(one) => push_ty(one, out),
                    _ => {}
                }
                let params = decl(path);
                for (i, a) in args.iter().enumerate() {
                    arg(a, params.and_then(|p| p.get(i)), known, decl, out);
                }
            }
            Type::Pointer(inner) | Type::Slice(inner) => walk(inner, known, decl, out),
            Type::Array(inner, n) | Type::Simd(inner, n) => {
                walk(inner, known, decl, out);
                // an array length has no declaration to consult, so `u32` again.
                if let ConstVal::Param(n) = n { push_const(n, Type::Uint32, out); }
            }
            Type::Function { params, return_type } => {
                for p in params { walk(p, known, decl, out); }
                walk(return_type, known, decl, out);
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    // descend one level before collecting, so the target itself is never taken
    // for a parameter.
    match target {
        Type::Path { path, args } => {
            let params = decl(path);
            for (i, a) in args.iter().enumerate() {
                arg(a, params.and_then(|p| p.get(i)), known, decl, &mut out);
            }
        }
        other => walk(other, known, decl, &mut out),
    }
    out
}

/// Resolve an `extend` target through a module's scopes, yielding the type in
/// resolved form plus the head it dispatches on.
///
/// `None` means the target names something that doesn't exist, or is a bare type
/// parameter (a blanket impl, which has no head and isn't supported). Either way
/// the caller drops the record silently: the same target is resolved again in
/// pass 2 as the `self` parameter's type, and *that* is where the diagnostic
/// comes from — reporting here as well would say it twice.
fn resolve_extend_target<'a>(
    target: &Type<'a>,
    generics: &[GenericParam<'a>],
    scopes: &Scopes<'a>,
    type_params: &TypeParams<'a>,
    aliases: &Aliases<'a>,
    members: &MemberTable<'a>,
    file: FileId,
) -> Option<(Type<'a>, TyHead)> {
    let mut errs = Vec::new();
    let mut rw = Rewriter {
        scopes, type_params, aliases, members,
        in_default_of: None,
        // this rewriter only resolves a type, which never consults the member
        // table, so the module never gates anything here.
        module: ModId(u32::MAX),
        errs: &mut errs,
        locals: Vec::new(),
        span: Span::new(file, 0, 0),
    };
    let mut resolved = target.clone();
    rw.ty(&mut resolved, &generic_names(generics));
    if !errs.is_empty() { return None; }
    let head = TyHead::of(&resolved)?;
    Some((resolved, head))
}

/// An identifier-safe fragment naming an `extend` target, for the symbol of the
/// functions its methods desugar into. Only has to be stable and mostly
/// distinct: the emitted name is never parsed back, and `lower_methods`
/// uniquifies within the module.
fn target_key(ty: &Type<'_>) -> String {
    match ty {
        Type::Path { path, .. } => path.last().to_string(),
        Type::Pointer(inner) => format!("ptr_{}", target_key(inner)),
        Type::Slice(inner) => format!("slice_{}", target_key(inner)),
        Type::Array(inner, _) => format!("array_{}", target_key(inner)),
        Type::Simd(inner, _) => format!("simd_{}", target_key(inner)),
        Type::Function { .. } => "proc".to_string(),
        // a scalar prints as its own keyword (`i32`, `bool`, `str`), which is
        // already identifier-safe.
        scalar => scalar.to_string(),
    }
}

/// Replace `Self` in a desugared `extend` method with the block's target type.
/// `head` replaces `Self` at the start of expression paths and is absent for
/// structural targets; type positions always receive the full target.
fn self_subst_type<'a>(ty: &mut Type<'a>, target: &Type<'a>, assoc: &[(&'a str, Type<'a>)]) {
    match ty {
        Type::Path { path, args } => {
            if path.segments.as_slice() == ["Self"] {
                *ty = target.clone();
                return;
            }
            // `Self::Item` - an associated type, which has a concrete answer only
            // once an impl has bound it. That is exactly the situation a trait's
            // *default body* is copied into, so `assoc` carries the impl's
            // bindings; it is empty for an ordinary method, where nothing has
            // written a projection the resolver could not already handle.
            if path.segments.len() == 2 && path.segments[0] == "Self" {
                if let Some((_, bound)) = assoc.iter().find(|(n, _)| *n == path.segments[1]) {
                    *ty = bound.clone();
                    return;
                }
            }
            for a in args {
                if let GenericArg::Type(t) = a { self_subst_type(t, target, assoc); }
            }
        }
        Type::Pointer(inner) | Type::Slice(inner) => self_subst_type(inner, target, assoc),
        Type::Array(inner, _) | Type::Simd(inner, _) => self_subst_type(inner, target, assoc),
        Type::Function { params, return_type } => {
            for p in params { self_subst_type(p, target, assoc); }
            self_subst_type(return_type, target, assoc);
        }
        _ => {}
    }
}

fn self_subst_args<'a>(args: &mut [GenericArg<'a>], target: &Type<'a>, assoc: &[(&'a str, Type<'a>)]) {
    for a in args {
        if let GenericArg::Type(t) = a { self_subst_type(t, target, assoc); }
    }
}

/// Replace a leading `Self` segment of an expression/pattern path with the
/// target's segments, so `Self { .. }` becomes `Gain { .. }` and `Self::make()`
/// becomes `Gain::make()`. A no-op when the head isn't `Self` or the target has
/// no path form.
fn self_subst_head<'a>(path: &mut Path<'a>, head: Option<&Path<'a>>) {
    if path.segments.first() == Some(&"Self") {
        if let Some(h) = head {
            let mut segs = h.segments.clone();
            segs.extend_from_slice(&path.segments[1..]);
            path.segments = segs;
        }
    }
}

fn self_subst_expr<'a>(e: &mut Expr<'a>, target: &Type<'a>, head: Option<&Path<'a>>, assoc: &[(&'a str, Type<'a>)]) {
    match &mut e.value {
        ExprNode::Path(nr) => self_subst_head(&mut nr.path, head),
        ExprNode::Struct { name, type_args, fields } => {
            self_subst_head(&mut name.path, head);
            self_subst_args(type_args, target, assoc);
            for (_, v) in fields { self_subst_expr(v, target, head, assoc); }
        }
        ExprNode::FnRef { name, type_args } => {
            self_subst_head(&mut name.path, head);
            self_subst_args(type_args, target, assoc);
        }
        ExprNode::Call { func, type_args, args } => {
            self_subst_expr(func, target, head, assoc);
            self_subst_args(type_args, target, assoc);
            for a in args { self_subst_expr(a, target, head, assoc); }
        }
        ExprNode::Slice(elems) => for el in elems { self_subst_expr(el, target, head, assoc); },
        ExprNode::Repeat { value, .. } => self_subst_expr(value, target, head, assoc),
        ExprNode::Access { base, .. } => self_subst_expr(base, target, head, assoc),
        ExprNode::Index { slice, index } => {
            self_subst_expr(slice, target, head, assoc);
            self_subst_expr(index, target, head, assoc);
        }
        ExprNode::Unary { operand, .. } => self_subst_expr(operand, target, head, assoc),
        ExprNode::Binary { left, right, .. } => {
            self_subst_expr(left, target, head, assoc);
            self_subst_expr(right, target, head, assoc);
        }
        _ => {}
    }
}

fn self_subst_pat<'a>(p: &mut Pattern<'a>, head: Option<&Path<'a>>) {
    match &mut p.value {
        PatternNode::Path(nr) => self_subst_head(&mut nr.path, head),
        PatternNode::Variant { path, fields } => {
            self_subst_head(&mut path.path, head);
            for f in fields { self_subst_pat(f, head); }
        }
        PatternNode::StructVariant { path, fields } => {
            self_subst_head(&mut path.path, head);
            for (_, f) in fields { self_subst_pat(f, head); }
        }
        _ => {}
    }
}

fn self_subst_stmt<'a>(s: &mut Stmt<'a>, target: &Type<'a>, head: Option<&Path<'a>>, assoc: &[(&'a str, Type<'a>)]) {
    match &mut s.value {
        StmtNode::Expr(e) => self_subst_expr(e, target, head, assoc),
        StmtNode::Block(stmts) => for st in stmts { self_subst_stmt(st, target, head, assoc); },
        StmtNode::Declare { ty, value, .. } => {
            if let Some(ty) = ty { self_subst_type(ty, target, assoc); }
            self_subst_expr(value, target, head, assoc);
        }
        StmtNode::Assign { left, value } => {
            self_subst_expr(left, target, head, assoc);
            self_subst_expr(value, target, head, assoc);
        }
        StmtNode::If { condition, then_branch, else_branch } => {
            self_subst_expr(condition, target, head, assoc);
            self_subst_stmt(then_branch, target, head, assoc);
            if let Some(e) = else_branch { self_subst_stmt(e, target, head, assoc); }
        }
        StmtNode::While { condition, body } => {
            self_subst_expr(condition, target, head, assoc);
            self_subst_stmt(body, target, head, assoc);
        }
        StmtNode::Match { scrutinee, arms } => {
            self_subst_expr(scrutinee, target, head, assoc);
            for (pat, body) in arms {
                self_subst_pat(pat, head);
                self_subst_stmt(body, target, head, assoc);
            }
        }
        StmtNode::Return(Some(e)) => self_subst_expr(e, target, head, assoc),
        StmtNode::Return(None) | StmtNode::Continue | StmtNode::Break => {}
    }
}

/// Renumber every node in a duplicated statement, so the copy is a distinct
/// piece of AST rather than an alias of the original.
///
/// Only a trait's default body needs this: it is the one construct copied into
/// more than one place. Without it the second impl's `self` type overwrites the
/// first's in `node_types`, and the first impl's method typechecks against the
/// *other* impl's receiver - which reads as a wildly confusing mismatch pointing
/// at the trait declaration. Mirrors `self_subst_stmt`'s traversal exactly; the
/// two must stay in step.
fn refresh_ids_stmt(s: &mut Stmt<'_>) {
    s.refresh_id();
    match &mut s.value {
        StmtNode::Expr(e) => refresh_ids_expr(e),
        StmtNode::Block(stmts) => for st in stmts { refresh_ids_stmt(st); },
        StmtNode::Declare { value, .. } => refresh_ids_expr(value),
        StmtNode::Assign { left, value } => {
            refresh_ids_expr(left);
            refresh_ids_expr(value);
        }
        StmtNode::If { condition, then_branch, else_branch } => {
            refresh_ids_expr(condition);
            refresh_ids_stmt(then_branch);
            if let Some(e) = else_branch { refresh_ids_stmt(e); }
        }
        StmtNode::While { condition, body } => {
            refresh_ids_expr(condition);
            refresh_ids_stmt(body);
        }
        StmtNode::Match { scrutinee, arms } => {
            refresh_ids_expr(scrutinee);
            for (pat, body) in arms {
                refresh_ids_pat(pat);
                refresh_ids_stmt(body);
            }
        }
        StmtNode::Return(Some(e)) => refresh_ids_expr(e),
        StmtNode::Return(None) | StmtNode::Continue | StmtNode::Break => {}
    }
}

fn refresh_ids_expr(e: &mut Expr<'_>) {
    e.refresh_id();
    match &mut e.value {
        ExprNode::Struct { fields, .. } => for (_, v) in fields { refresh_ids_expr(v); },
        ExprNode::Call { func, args, .. } => {
            refresh_ids_expr(func);
            for a in args { refresh_ids_expr(a); }
        }
        ExprNode::Slice(elems) => for el in elems { refresh_ids_expr(el); },
        ExprNode::Repeat { value, .. } => refresh_ids_expr(value),
        ExprNode::Access { base, .. } => refresh_ids_expr(base),
        ExprNode::Index { slice, index } => {
            refresh_ids_expr(slice);
            refresh_ids_expr(index);
        }
        ExprNode::Unary { operand, .. } => refresh_ids_expr(operand),
        ExprNode::Binary { left, right, .. } => {
            refresh_ids_expr(left);
            refresh_ids_expr(right);
        }
        _ => {}
    }
}

fn refresh_ids_pat(p: &mut Pattern<'_>) {
    p.refresh_id();
    match &mut p.value {
        PatternNode::Variant { fields, .. } => for f in fields { refresh_ids_pat(f); },
        PatternNode::StructVariant { fields, .. } => for (_, f) in fields { refresh_ids_pat(f); },
        _ => {}
    }
}

/// Desugar every `extend` block (and the synthesized ones from inherent struct/
/// enum method bodies) into ordinary top-level functions, in place. A method on
/// `T` becomes a function named `T$method` with a `self` param prepended for a
/// value/pointer receiver (`self: T` / `self: *T`); associated functions get no
/// receiver param. After this runs the module has no `Extend` nodes, so every
/// later stage - name resolution, typecheck, mono, codegen - sees only functions.
///
/// A receiver-call site `recv.method()` can't be rewritten here (it needs the
/// receiver's type); that resolution happens in the typechecker. An associated
/// call `T::method()` is rewritten by name resolution (see `call_name`), keyed on
/// the same `T$method` name minted here.
fn lower_methods<'a>(items: &mut Vec<TopLevel<'a>>, arena: &'a Bump)
    -> (Vec<Error>, Vec<RawImpl<'a>>, Vec<RawMethod<'a>>)
{
    // desugaring itself can no longer fail: what used to be rejected here (a
    // method on a generic type) is now the point of the exercise, and the
    // remaining ways an `extend` can be wrong - an unknown target, a parameter
    // name clash, a duplicate method - are all only decidable once scopes exist.
    let errors = Vec::new();
    let mut impls: Vec<RawImpl<'a>> = Vec::new();
    let mut methods_out: Vec<RawMethod<'a>> = Vec::new();

    let mut synthesized: Vec<TopLevel<'a>> = Vec::new();
    let mut kept: Vec<TopLevel<'a>> = Vec::with_capacity(items.len());
    // synthesized names already handed out in this module, so two `extend`
    // blocks whose targets share a key (`extend [i32]` and `extend [f32]`) don't
    // emit two functions under one symbol. A same-named *method* on both is
    // separately a duplicate-member error in pass 1.5; this only keeps the
    // symbols apart long enough to get there.
    let mut used_names: HashSet<String> = HashSet::new();
    for tl in items.drain(..) {
        let TopLevelNode::Extend { target, trait_, where_bounds, assoc_bindings, methods } = &tl.value else {
            kept.push(tl);
            continue;
        };
        // record the conformance obligation; resolved through the module's
        // scopes in `load_and_merge`, which is also where the target's inferred
        // parameters are worked out - they need a type scope, which does not
        // exist yet at parse time.
        if let Some(tr) = trait_ {
            impls.push(RawImpl {
                target: target.clone(),
                generics: Vec::new(),
                where_bounds: where_bounds.clone(),
                trait_: tr,
                assoc_bindings: assoc_bindings.clone(),
                defined: methods.iter().map(|m| m.value.name).collect(),
                span: tl.span.clone(),
            });
        }
        let key = target_key(target);
        for m in methods {
            let mnode = &m.value;
            let mut params: Vec<(&'a str, Type<'a>)> = Vec::with_capacity(mnode.params.len() + 1);
            // the target type verbatim, still unresolved: `extend` is desugared
            // before name resolution, so the rewriter sees the same `Path` here
            // that it would have seen written out by hand - and rewrites it with
            // this function's (later-patched) generics in scope.
            let self_ty = target.clone();
            match mnode.receiver {
                Receiver::Associated => {}
                Receiver::Value => params.push(("self", self_ty)),
                Receiver::Pointer => params.push(("self", Type::Pointer(Box::new(self_ty)))),
            }
            params.extend(mnode.params.iter().cloned());
            // rewrite every `Self` in the (non-receiver) param types, the return
            // type, and the body to the concrete target, so the synthesized free
            // function mentions no `Self`. `head` lets an expression-position
            // `Self` (`Self { .. }`, `Self::make()`) take the target's path; it is
            // `None` for a non-nominal target, where only type positions apply.
            let head = match target {
                Type::Path { path, .. } => Some(path),
                _ => None,
            };
            for (_, pty) in params.iter_mut() { self_subst_type(pty, target, &[]); }
            let mut return_type = mnode.return_type.clone();
            self_subst_type(&mut return_type, target, &[]);
            let mut body = mnode.body.clone();
            for st in body.iter_mut() { self_subst_stmt(st, target, head, &[]); }
            // the synthesized name only has to be unique within the module - it is
            // never reconstructed by a consumer, since the member record below is
            // what makes this method findable.
            let mut base = format!("{}${}", key, mnode.name);
            for n in 1.. {
                if used_names.insert(base.clone()) { break; }
                base = format!("{}${}${}", key, mnode.name, n);
            }
            let fname: &'a str = arena.alloc_str(&base);
            methods_out.push(RawMethod {
                target: target.clone(),
                // filled in by `load_and_merge` once a type scope exists.
                generics: Vec::new(),
                where_bounds: where_bounds.clone(),
                name: mnode.name,
                fn_name: fname,
                receiver: mnode.receiver,
                // a trait-impl method is public regardless of what was written:
                // its reachability follows the trait. An inherent method is
                // private unless marked `pub`.
                is_pub: mnode.is_pub || trait_.is_some(),
                default_of: None,
                span: tl.span.clone(),
            });
            synthesized.push(Metadata::new(
                TopLevelNode::Function {
                    name: fname,
                    def: DefId::UNRESOLVED,
                    is_pub: mnode.is_pub,
                    attributes: mnode.attributes.clone(),
                    // the impl's own parameters are prepended in `load_and_merge`.
                    generics: mnode.generics.clone(),
                    // merged there too, once those parameters are in place - a
                    // method's clause may bound one of them as readily as one of
                    // its own.
                    where_bounds: mnode.where_bounds.clone(),
                    params,
                    return_type,
                    body,
                },
                m.span.clone(),
            ));
        }
    }
    *items = kept;
    items.extend(synthesized);
    (errors, impls, methods_out)
}

/// Lex + parse one module's source into `(imports, items)`, printing any
/// lex/parse diagnostics. tokens move into `arena` so the parsed AST can borrow
/// them for `'a`. `extend`/method blocks are desugared to functions here, so the
/// returned items are already method-free.
fn parse_module<'a>(file: FileId, src: &'a str, arena: &'a Bump, files: &Files<'a>)
    -> Result<(Vec<Attribute<'a>>, Vec<Import<'a>>, Vec<TopLevel<'a>>,
               Vec<RawImpl<'a>>, Vec<RawMethod<'a>>), ()>
{
    let (tokens, lex_errs) = parse::lex(file, src);
    for e in &lex_errs {
        diag::report("Lex error", &e.reason().to_string(), e.span(), files);
    }
    let tokens = match tokens {
        Some(t) if lex_errs.is_empty() => t,
        _ => return Err(()),
    };
    let tokens: &'a [Metadata<Token<'a>>] = arena.alloc_slice_fill_iter(tokens);

    let (parsed, parse_errs) = parse::parse(file, src.len(), tokens);
    for e in &parse_errs {
        diag::report("Parse error", &e.reason().to_string(), e.span(), files);
    }
    let (mod_attrs, imports, mut items) = match parsed {
        Some(pi) if parse_errs.is_empty() => pi,
        _ => return Err(()),
    };

    // reject attributes the compiler does not act on, before any stage reads the
    // ones it does. Runs here, ahead of `lower_methods`, because an `extend`
    // block's methods still exist as methods at this point - once desugared they
    // are indistinguishable from top-level functions, which happens to be the
    // right target for them anyway, but their spans read better this way.
    let attr_errs = check_attributes(&mod_attrs, &items);
    if !attr_errs.is_empty() {
        for e in &attr_errs {
            diag::report_error("Attribute error", e, files);
        }
        return Err(());
    }

    // desugar `extend`/method blocks into functions before anything else looks at
    // the items.
    let (method_errs, impls, methods) = lower_methods(&mut items, arena);
    if !method_errs.is_empty() {
        for e in &method_errs {
            diag::report_error("Method error", e, files);
        }
        return Err(());
    }

    Ok((mod_attrs, imports, items, impls, methods))
}

/// Check every attribute in one module - the module's own `@!name` marks and
/// the ones on its items - against the table of attributes the compiler acts on
/// ([`ast::KNOWN_ATTRIBUTES`]).
///
/// The parser accepts any `@name` or `@name(value)`, which is what lets one
/// grammar serve every attribute - but it also meant an attribute the compiler
/// had never heard of, or one written somewhere it is never read, compiled
/// silently and did nothing.
fn check_attributes<'a>(mod_attrs: &[Attribute<'a>], items: &[TopLevel<'a>]) -> Vec<Error> {
    let mut errs = Vec::new();
    let check = |attrs: &[Attribute<'a>], target: AttrTarget, errs: &mut Vec<Error>| {
        for a in attrs {
            if let Err((msg, note)) = check_attribute(&a.value, target) {
                errs.push(Error::new(a.span.clone(), msg).with_note(note));
            }
        }
    };
    check(mod_attrs, AttrTarget::Module, &mut errs);
    for tl in items {
        match &tl.value {
            TopLevelNode::Function { attributes, .. } =>
                check(attributes, AttrTarget::Function, &mut errs),
            TopLevelNode::Extern { attributes, .. } =>
                check(attributes, AttrTarget::Extern, &mut errs),
            TopLevelNode::Struct { attributes, .. } =>
                check(attributes, AttrTarget::Struct, &mut errs),
            TopLevelNode::Enum { attributes, .. } =>
                check(attributes, AttrTarget::Enum, &mut errs),
            TopLevelNode::Global { attributes, .. } =>
                check(attributes, AttrTarget::Global, &mut errs),
            TopLevelNode::Trait { attributes, .. } =>
                check(attributes, AttrTarget::Trait, &mut errs),
            TopLevelNode::Alias { attributes, .. } =>
                check(attributes, AttrTarget::Alias, &mut errs),
            // an `extend` block takes no attributes of its own; each of its
            // methods is checked as the function it desugars into.
            TopLevelNode::Extend { methods, .. } => {
                for m in methods {
                    check(&m.value.attributes, AttrTarget::Function, &mut errs);
                }
            }
        }
    }
    errs
}

/// Name-resolution scopes for one module.
#[derive(Default)]
struct Scopes<'a> {
    /// unqualified callable names in scope: functions, externs, globals.
    calls: HashMap<&'a str, Sym<'a>>,
    /// unqualified type names in scope: structs, enums and traits.
    types: HashMap<&'a str, Sym<'a>>,
    /// `qualifier -> that module's exports`, for whole-module imports.
    quals: HashMap<&'a str, QualScope<'a>>,
}

/// What a whole-module import (`import std/math`) makes reachable as
/// `math::sym`. Split by namespace like [`Scopes`] itself, so a module that
/// exports both a `Buf` type and a `Buf` function doesn't have one hide the
/// other — which a single flat map did, structs being inserted last.
///
/// `children` is what makes a qualifier path nest: importing a *directory*
/// binds one qualifier whose children are the modules under it, so `dsp::osc`
/// walks to a scope and `dsp::osc::Osc` reads a name out of it. A plain module
/// import has no children, and a directory that also had a module of its own
/// would fill both halves - nothing forbids it, there is just no syntax for it
/// yet.
#[derive(Default)]
struct QualScope<'a> {
    calls: HashMap<&'a str, Sym<'a>>,
    types: HashMap<&'a str, Sym<'a>>,
    children: HashMap<&'a str, QualScope<'a>>,
}

impl<'a> QualScope<'a> {
    /// Fill this scope with a module's public exports.
    fn fill_from(&mut self, st: &SymTab<'a>) {
        for (&k, v) in &st.fns { if v.is_pub { self.calls.insert(k, *v); } }
        for (&k, v) in &st.structs { if v.is_pub { self.types.insert(k, *v); } }
    }

    /// The descendant named by `segments`, creating empty scopes along the way.
    fn child_at(&mut self, segments: &[&'a str]) -> &mut QualScope<'a> {
        let mut cur = self;
        for seg in segments {
            cur = cur.children.entry(seg).or_default();
        }
        cur
    }
}

/// What [`Rewriter::type_head`] resolved a written type path to: a definition,
/// or an abstract stand-in (a generic parameter, or `Self` in a trait method
/// signature) that only typecheck can give meaning to.
enum TypeHead<'a> {
    Def(DefId),
    Param(&'a str),
}

/// Which half of a [`SymTab`] a lookup means. Lets the re-export pass do the
/// same thing to both namespaces without duplicating the loop - a re-exported
/// name lands in whichever namespace(s) it exists in, exactly as a plain
/// selective import does.
#[derive(Clone, Copy)]
enum Namespace { Fns, Structs }

impl Namespace {
    fn get<'a>(self, st: &SymTab<'a>, sym: &str) -> Option<Sym<'a>> {
        match self {
            Namespace::Fns => st.fns.get(sym).copied(),
            Namespace::Structs => st.structs.get(sym).copied(),
        }
    }

    fn get_mut<'t, 'a>(self, st: &'t mut SymTab<'a>) -> &'t mut HashMap<&'a str, Sym<'a>> {
        match self {
            Namespace::Fns => &mut st.fns,
            Namespace::Structs => &mut st.structs,
        }
    }
}

/// Holds the per-module scopes and error sink while rewriting a module's AST in
/// place.
struct Rewriter<'x, 'a> {
    scopes: &'x Scopes<'a>,
    /// what every struct and enum declares, for `normalize_const_args`.
    type_params: &'x TypeParams<'a>,
    /// every resolved alias, for the expansion in `ty`. Empty while the aliases
    /// themselves are being resolved and one still has unresolved dependencies -
    /// which is why that pass only resolves a body once every alias it names is
    /// already in here.
    aliases: &'x Aliases<'a>,
    /// every method in the program, keyed by `(final type name, method name)`.
    /// what makes `Point::new()` resolve for an imported `Point`.
    members: &'x MemberTable<'a>,
    /// the module whose items are being rewritten. A `Type::method()` call
    /// reaching a private method declared in a *different* module is an error;
    /// this is the "here" that comparison is against.
    module: ModId,
    errs: &'x mut Vec<Error>,
    /// names bound by params/`let`/match-arm patterns in the function being
    /// walked, innermost last. used as a stack: a block records `locals.len()` on
    /// entry and truncates back to it on exit. a name in here shadows a top-level
    /// callable of the same name, so bare/call references to it are left
    /// unrewritten (it's a local value, not the top-level symbol).
    locals: Vec<&'a str>,
    /// span of the innermost item/statement/expression being walked. `Type` and
    /// bare names carry no span of their own, so unresolved-name errors point at
    /// whatever encloses them - which is as precise as the AST allows.
    span: Span,
    /// Set while walking a method copied out of a trait's default body: the
    /// trait's name. Every error raised in there gets a note saying so, because
    /// the span points into the *trait's* module while the names were resolved in
    /// the impl's - so "is it defined and imported into this module?" is asking
    /// about a module the reader is not looking at.
    in_default_of: Option<&'a str>,
}

impl<'x, 'a> Rewriter<'x, 'a> {
    fn error(&mut self, span: &Span, msg: String) {
        let err = Error::new(span.clone(), msg);
        self.push(err);
    }

    /// Record an error the caller has already given its labels and note.
    fn push(&mut self, err: Error) {
        self.errs.push(match self.in_default_of {
            Some(tr) => err.with_note(format!(
                "this is the default body of trait '{}', copied into the impl - \
                 it is resolved where the impl is written, so every name it uses \
                 has to be in scope there too", tr)),
            None => err,
        });
    }

    /// Record an error against the innermost span being walked. For names that
    /// have no span of their own (types, and the leaves inside them).
    fn error_here(&mut self, msg: String) {
        let span = self.span.clone();
        self.push(Error::new(span, msg));
    }

    fn is_local(&self, name: &str) -> bool {
        self.locals.iter().any(|n| *n == name)
    }

    /// Rewrite the enum half of an `Enum::Variant` path to the enum's final name,
    /// returning `true` if the head segment named a type at all.
    ///
    /// Enum names are mangled like struct names, so every reference to a variant
    /// has to be re-spelled - in call position (`E::V(x)`), value position (a unit
    /// variant), struct-literal position (`E::V { .. }`) and match patterns. Miss
    /// any one of them and the enum's name has to stay globally unique, which is
    /// exactly the constraint this lifts.
    fn variant_path(&mut self, r: &mut NameRef<'a>) -> bool {
        let Some(sym) = self.scopes.types.get(r.path.segments[0]) else { return false };
        r.def = sym.def;
        true
    }

    /// Rewrite the trait names in an item's generic bounds (`T: Display`). Traits
    /// are mangled now, so a bound has to name the final trait for it to line up
    /// with the conformance records and the trait table.
    fn bounds(&mut self, generics: &mut [GenericParam<'a>]) {
        for g in generics.iter_mut() {
            let GenericParam::Type { bounds, .. } = g else { continue };
            for b in bounds.iter_mut() {
                match self.scopes.types.get(b.path.last()) {
                    Some(sym) => b.def = sym.def,
                    None => self.error_here(format!("unknown trait '{}'", b)),
                }
            }
        }
    }

    /// Rewrite any enum paths a match pattern names, and collect the locals it
    /// binds. Patterns used to be skipped entirely, which was only safe while enum
    /// names were never mangled.
    fn pattern(&mut self, pat: &mut PatternNode<'a>, out: &mut Vec<&'a str>) {
        match pat {
            PatternNode::Bind(name) => out.push(name),
            PatternNode::Path(path) => { self.variant_path(path); }
            PatternNode::Variant { path, fields } => {
                self.variant_path(path);
                for f in fields { self.pattern(&mut f.value, out); }
            }
            PatternNode::StructVariant { path, fields } => {
                self.variant_path(path);
                for (_, f) in fields { self.pattern(&mut f.value, out); }
            }
            PatternNode::Wildcard | PatternNode::Int(_) => {}
        }
    }

    /// Rewrite struct type names in `ty`, skipping names bound as generic params
    /// of the enclosing function (those are type params, not structs). a
    /// `geo::Point` written for a whole-module import is resolved through the
    /// qualifier map; a bare name through the unqualified type scope. a name that
    /// resolves to neither is an error here - letting it through used to leave the
    /// source spelling in place, which then failed much later as a mismatch
    /// against some *other* module's mangled name (see the `Display`/`String`
    /// prelude bug).
    fn ty(&mut self, ty: &mut Type<'a>, gparams: &HashSet<&str>) {
        match ty {
            Type::Path { path, args } => {
                // a generic type's arguments are themselves types to rewrite;
                // const arguments carry no names to resolve.
                for a in args.iter_mut() {
                    if let GenericArg::Type(t) = a { self.ty(t, gparams); }
                }
                let mut args = std::mem::take(args);
                let last = path.last();
                *ty = match self.type_head(path, gparams) {
                    TypeHead::Def(def) => {
                        self.normalize_const_args(def, &mut args);
                        match self.aliases.get(&def) {
                            Some(_) => self.expand_alias(def, args, last),
                            None => Type::Named { def, args },
                        }
                    }
                    TypeHead::Param(name) => Type::Param(name),
                };
            }
            Type::Pointer(inner)
            | Type::Array(inner, _)
            | Type::Slice(inner)
            | Type::Simd(inner, _) => self.ty(inner, gparams),
            Type::Function { params, return_type } => {
                for p in params { self.ty(p, gparams); }
                self.ty(return_type, gparams);
            }
            _ => {}
        }
    }

    /// Point a struct literal written through an alias at the struct it names.
    ///
    /// `def` is whatever the head resolved to; if that is an alias, expanding it
    /// must land on a named type, since a literal of `type Sample = f32` is not a
    /// thing that can be written. The expansion's own arguments replace the ones
    /// at the literal - `Quad::<i32> { .. }` becomes `Buf::<i32, 4> { .. }` - so
    /// typecheck sees exactly what the spelled-out form would have produced.
    fn expand_literal_head(&mut self, name: &mut NameRef<'a>, type_args: &mut Vec<GenericArg<'a>>) {
        if !self.aliases.contains_key(&name.def) { return }
        let written = name.path.last();
        match self.expand_alias(name.def, std::mem::take(type_args), written) {
            Type::Named { def, args } => {
                name.def = def;
                *type_args = args;
            }
            // an arity error already reported by `expand_alias` lands here too;
            // it left a `Param`, and a second message would only repeat it.
            Type::Param(_) => {}
            other => {
                let span = self.span.clone();
                self.error(&span, format!(
                    "type alias '{}' names '{}', which is not a struct", written, other));
            }
        }
    }

    /// Replace a use of a type alias with the type it stands for.
    ///
    /// An alias is transparent: nothing downstream learns that `Sample` was ever
    /// written, only that the type is `f32`. That is what lets a method on the
    /// aliased type be called through the alias, and what keeps every later stage
    /// free of a variant it would have to see through.
    fn expand_alias(&mut self, def: DefId, args: Vec<GenericArg<'a>>, written: &'a str) -> Type<'a> {
        let al = &self.aliases[&def];
        if args.len() != al.generics.len() {
            let span = self.span.clone();
            self.errs.push(Error::new(span.clone(), format!(
                "type alias '{}' expects {} argument{}, got {}",
                written, al.generics.len(),
                if al.generics.len() == 1 { "" } else { "s" }, args.len()))
                .with_label(span, format!("used with {} here", args.len()))
                .with_label(al.span.clone(), format!("'{}' is declared here", written)));
            return Type::Param(written);
        }
        let mut types: HashMap<&'a str, Type<'a>> = HashMap::new();
        let mut consts: HashMap<&'a str, ConstVal<'a>> = HashMap::new();
        for (gp, ga) in al.generics.iter().zip(&args) {
            match (gp, ga) {
                (GenericParam::Type { name, .. }, GenericArg::Type(t)) => {
                    types.insert(name, t.clone());
                }
                (GenericParam::Const(name, _), GenericArg::Const(c)) => {
                    consts.insert(name, c.clone());
                }
                // a bare identifier naming a const param still arrives as a type
                // when the alias' own binder is what declares it, there being no
                // struct declaration for `normalize_const_args` to consult.
                (GenericParam::Const(name, _), GenericArg::Type(Type::Param(n))) => {
                    consts.insert(name, ConstVal::Param(n));
                }
                (GenericParam::Type { name, .. }, GenericArg::Const(_)) => {
                    let span = self.span.clone();
                    self.error(&span, format!(
                        "type alias '{}': expected a type argument for '{}', got a const value",
                        written, name));
                }
                (GenericParam::Const(name, _), GenericArg::Type(_)) => {
                    let span = self.span.clone();
                    self.error(&span, format!(
                        "type alias '{}': expected a const argument for '{}', got a type",
                        written, name));
                }
            }
        }
        subst_alias(&al.ty, &types, &consts)
    }

    /// Re-tag any argument that fills a `const` slot but parsed as a type.
    ///
    /// `Buf<T, N>` has no way to say which of its arguments is a value - both are
    /// bare identifiers, and the parser has no declaration to consult - so both
    /// arrive as `GenericArg::Type`. By here the head is resolved and the
    /// declaration *is* in hand, and a parameter name sitting in a const slot can
    /// only be a const param. Re-tagging it now is what lets this type be
    /// compared against a `Buf<i32, 4>`, whose `4` is a `GenericArg::Const`:
    /// until it was, an `extend Buf<T, N>` method never matched its own receiver
    /// and every call on one read as a missing field.
    fn normalize_const_args(&self, def: DefId, args: &mut [GenericArg<'a>]) {
        let Some(params) = self.type_params.get(&def) else { return };
        for (gp, ga) in params.iter().zip(args.iter_mut()) {
            if let (GenericParam::Const(..), GenericArg::Type(Type::Param(n))) = (gp, &*ga) {
                *ga = GenericArg::Const(ConstVal::Param(n));
            }
        }
    }

    /// Rewrite a `Self::Item` projection inside a trait method signature to the
    /// bare associated-type name, so the following `ty` pass resolves it to a
    /// `Param("Item")` (the trait scope treats each associated type like an
    /// implicit type parameter; conformance substitutes it per impl).
    ///
    /// Only `Self::<assoc>` is accepted here. `Self::<other>` names an
    /// associated type the trait never declared, and a projection on anything
    /// but `Self` (`T::Item` on a bounded parameter) is not supported yet; both
    /// are reported. Every other path is left for `ty` to resolve normally, so a
    /// module qualifier (`geo::Point`) is untouched.
    fn self_assoc(&mut self, ty: &mut Type<'a>, assoc: &HashSet<&'a str>) {
        match ty {
            Type::Path { path, args } => {
                let segs = &path.segments;
                if segs.len() == 2 && segs[0] == "Self" {
                    if assoc.contains(segs[1]) {
                        if !args.is_empty() {
                            self.error_here(format!(
                                "associated type '{}' takes no arguments", segs[1]));
                        }
                        *path = Path::single(segs[1]);
                    } else {
                        self.error_here(format!(
                            "trait has no associated type '{}'", segs[1]));
                    }
                    return;
                }
                for a in args.iter_mut() {
                    if let GenericArg::Type(t) = a { self.self_assoc(t, assoc); }
                }
            }
            Type::Pointer(inner)
            | Type::Array(inner, _)
            | Type::Slice(inner)
            | Type::Simd(inner, _) => self.self_assoc(inner, assoc),
            Type::Function { params, return_type } => {
                for p in params { self.self_assoc(p, assoc); }
                self.self_assoc(return_type, assoc);
            }
            _ => {}
        }
    }

    /// Walk the leading qualifier segments of `segs`, returning the scope they
    /// name and how many they consumed.
    ///
    /// Takes the *longest* prefix that resolves, so a module named like a type
    /// in an outer namespace doesn't cut the walk short. `None` means the very
    /// first segment isn't a qualifier at all - the caller then treats the path
    /// as type-qualified (`Point::new`) or reports it.
    fn qual_prefix<'s>(&'s self, segs: &[&'a str]) -> Option<(&'s QualScope<'a>, usize)> {
        let mut cur = self.scopes.quals.get(segs[0])?;
        let mut n = 1;
        // stop before the last segment: a path always ends in a name, never in a
        // bare qualifier, so the final segment is never part of the prefix.
        while n + 1 < segs.len() {
            match cur.children.get(segs[n]) {
                Some(next) => { cur = next; n += 1; }
                None => break,
            }
        }
        Some((cur, n))
    }

    /// Report a path whose qualifier prefix resolved but whose remainder didn't
    /// name anything, with the most specific message the shape allows.
    fn qual_miss(&mut self, path: &Path<'a>, names_module: bool, scope_end: usize, what: &str) {
        let segs = &path.segments;
        let qual = segs[..scope_end].join("::");
        let rest = &segs[scope_end..];
        if rest.len() > 2 {
            self.error_here(format!(
                "'{}' has too many `::` segments after the module qualifier '{}'", path, qual));
        } else if names_module {
            // named a nested module where a name was expected - the likely slip
            // after importing a directory.
            self.error_here(format!(
                "'{}::{}' is a module, not a {}; name something inside it, \
                 e.g. `{}::{}::<name>`", qual, rest[0], what, qual, rest[0]));
        } else {
            self.error_here(format!(
                "{} '{}' is not exported by module '{}'", what, rest[0], qual));
        }
    }

    /// What a written type path denotes.
    ///
    /// An unresolvable path is an error here, and falls back to an abstract
    /// `Param` — never to a real identity, so a bad name cannot be mistaken for
    /// some other module's type. Errors are fatal before typecheck runs, so the
    /// fallback exists only to let resolution finish and report the rest.
    fn type_head(&mut self, path: &Path<'a>, gparams: &HashSet<&str>) -> TypeHead<'a> {
        if let Some(one) = path.as_single() {
            // a generic parameter of the enclosing item is a type param, not a
            // named type; `Self` in a trait method signature stands for the
            // implementing type and typecheck substitutes it per impl. Neither
            // resolves through a module scope, and neither has an identity.
            if gparams.contains(one) || one == "Self" { return TypeHead::Param(one); }
            return match self.scopes.types.get(one) {
                Some(sym) => TypeHead::Def(sym.def),
                None => {
                    self.error_here(format!("unknown type '{}'", one));
                    TypeHead::Param(one)
                }
            };
        }
        // `<qualifier...>::Type`. The qualifier may be several segments deep
        // when it came from a directory import (`dsp::osc::Osc`), so walk it
        // rather than assuming exactly one.
        let segs = &path.segments;
        // ...unless the qualifier is a type parameter of the enclosing item, in
        // which case this is an associated-type projection (`A::Item` where
        // `A: Iterator`), not a module path at all. It is not supported yet, and
        // saying so beats sending the reader after an import that would not help:
        // `A` is bound by the signature they are looking at.
        if segs.len() == 2 && gparams.contains(segs[0]) {
            self.error_here(format!(
                "associated type projection '{}::{}' is not supported yet",
                segs[0], segs[1]));
            return TypeHead::Param(path.last());
        }
        let Some((scope, n)) = self.qual_prefix(segs) else {
            self.error_here(format!(
                "unknown module qualifier '{}' (did you `import .../{}`?)", segs[0], segs[0]));
            return TypeHead::Param(path.last());
        };
        let names_module = scope.children.contains_key(path.last());
        if segs.len() - n == 1 {
            if let Some(sym) = scope.types.get(segs[n]) {
                return TypeHead::Def(sym.def);
            }
        }
        self.qual_miss(path, names_module, n, "type");
        TypeHead::Param(path.last())
    }

    /// Resolve a written path used as a value or a callee, to its final emitted
    /// name.
    ///
    /// `Some(name)` means it denotes a single symbol and the caller should replace
    /// the node with `ExprNode::Var(name)`. `None` means it stays an
    /// `ExprNode::Path` - the one shape that survives resolution is an enum
    /// variant, whose head segment has been rewritten to the enum's final name in
    /// place.
    ///
    /// A bare name that resolves to nothing is an error, with three deliberate
    /// exceptions, each an explicit branch below: a local shadowing a top-level
    /// callable, a compiler intrinsic (in no module's symbol table), and a
    /// type-qualified `T::sym` that isn't a known method (left for typecheck's
    /// enum-constructor path).
    fn value_path(&mut self, r: &mut NameRef<'a>, span: &Span, in_call: bool, gparams: &HashSet<&str>) -> Option<&'a str> {
        let path = &mut r.path;
        if let Some(one) = path.as_single() {
            if self.is_local(one) {
                // a param, `let`, or match-arm binding shadows any top-level
                // symbol of the same name. In call position that makes this an
                // indirect call through a value, not a reference to the symbol.
                return Some(one);
            }
            if let Some(sym) = self.scopes.calls.get(one) { return Some(sym.name); }
            if in_call && Intrinsic::lookup(one).is_some() {
                // a compiler intrinsic (`sizeof`, `null`, `__simd_*`): not declared
                // in any module, resolved by the typechecker. leave it untouched.
                return Some(one);
            }
            // A generic parameter of the enclosing item, read in value position.
            // A `const` one is pushed as a local above (whether it was declared in
            // a binder or inferred from an `extend` target) and never reaches
            // here, so what does reach here is a *type* parameter - a name that
            // is declared, just not as anything with a value. Worth saying, rather
            // than claiming the name is unknown when it is two lines up.
            if !in_call && gparams.contains(one) {
                self.push(Error::new(span.clone(), format!(
                    "type parameter '{}' cannot be used as a value", one))
                    .with_label(span.clone(), format!(
                        "'{}' names a type, not a value", one))
                    .with_note("only a `const` parameter stands for a value"));
                return Some(one);
            }
            self.push(if in_call {
                // same wording as the typechecker's own unknown-call diagnostic:
                // this just catches it a stage earlier, before mangling can
                // obscure it.
                Error::new(span.clone(), format!("unknown function '{}'", one))
                    .with_note("is it defined and imported into this module?")
            } else {
                Error::new(span.clone(), format!("unknown value '{}'", one))
            });
            return Some(one);
        }

        // `Type::sym` where `Type` is an unqualified type in scope: an
        // associated-function call `Point::new(...)`, or a data-enum variant
        // `Enum::Variant`. Both are type-qualified, not module-qualified, so
        // they are tried before the qualifier walk.
        //
        // The member lookup is keyed on the type's identity, so this works for
        // an imported type too. The old form reconstructed `Type$sym` and looked
        // it up in the call scope, which could only ever hit for a type declared
        // in this same module: no import form can name `Point$new` (`$` is
        // unlexable), so `import geo { Point }` + `Point::new()` was silently
        // left unresolved.
        let segs: Vec<&'a str> = path.segments.clone();
        if segs.len() == 2 {
            // an associated call through a generic type parameter, `P::new()`:
            // `P` is not a concrete type, so leave the path unresolved for the
            // typechecker to dispatch through `P`'s trait bound (and mono to
            // re-mangle to the concrete `Gain$new`). Checked before the module
            // qualifier walk, which would otherwise read `P` as a module name.
            if gparams.contains(segs[0]) {
                return None;
            }
            // a built-in type's associated function, `i32::from(x)`. Tried
            // first because the built-in names are keywords: `parse_type` maps
            // them before any user type can shadow them, so letting a `struct
            // i32` win here would make one name mean two things depending on
            // whether it is written in type or expression position.
            if let Some(head) = TyHead::of_builtin(segs[0]) {
                return Some(self.builtin_qualified(head, segs[0], segs[1], span));
            }
            if let Some(&ty) = self.scopes.types.get(segs[0]) {
                return self.type_qualified(r, ty.def, segs[1]);
            }
        }

        // otherwise a module-qualified path: `math::square`, `dsp::osc::phase`,
        // or a qualified type followed by one of its members
        // (`dsp::osc::Osc::new`).
        let Some((scope, n)) = self.qual_prefix(&segs) else {
            self.error(span, format!(
                "unknown module qualifier '{}' (did you `import .../{}`?)", segs[0], segs[0]));
            return Some(path.last());
        };
        match segs.len() - n {
            1 => {
                if let Some(sym) = scope.calls.get(segs[n]) { return Some(sym.name); }
                if scope.children.contains_key(segs[n]) {
                    self.error(span, format!(
                        "'{}::{}' is a module, not a value; name something inside it, \
                         e.g. `{}::{}::<name>`",
                        segs[..n].join("::"), segs[n], segs[..n].join("::"), segs[n]));
                } else {
                    self.error(span, format!(
                        "'{}' is not exported by module '{}'", segs[n], segs[..n].join("::")));
                }
                Some(path.last())
            }
            2 => {
                // `<qualifier...>::Type::sym`: resolve the type through the
                // namespace, then take its member or variant exactly as an
                // unqualified `Type::sym` would.
                let Some(&ty) = scope.types.get(segs[n]) else {
                    self.error(span, format!(
                        "type '{}' is not exported by module '{}'", segs[n], segs[..n].join("::")));
                    return Some(path.last());
                };
                self.type_qualified(r, ty.def, segs[n + 1])
            }
            _ => {
                self.error(span, format!(
                    "'{}' has too many `::` segments after the module qualifier '{}'",
                    path, segs[..n].join("::")));
                Some(path.last())
            }
        }
    }

    /// Resolve `sym` against the type `ty`: an associated function or method if
    /// the member table has one, otherwise an enum variant left for typecheck's
    /// constructor path (which is what `None` means to the caller).
    fn type_qualified(&mut self, r: &mut NameRef<'a>, ty: DefId, sym: &'a str) -> Option<&'a str> {
        if let Some(m) = self.members.get(&(TyHead::Def(ty), sym)) {
            self.check_member_visible(m, sym);
            return Some(m.name);
        }
        r.def = ty;
        None
    }

    /// Report a private method reached from another module. A method is
    /// module-private unless `pub` (trait-impl methods are always public - see
    /// [`Member::is_pub`]); a call in the declaring module always sees it. Same
    /// rule top-level functions follow, enforced here rather than by scope
    /// construction because methods live in one global table, not per-module
    /// scopes.
    fn check_member_visible(&mut self, m: &Member<'a>, sym: &str) {
        if !m.is_pub && m.module != self.module {
            self.error_here(format!(
                "method '{}' is private to its module; mark it `pub` to call it \
                 from another module", sym));
        }
    }

    /// Resolve `sym` against a built-in type: `i32::from(x)`, `str::len(s)`.
    ///
    /// Unlike [`Self::type_qualified`] there is no fallback — a primitive has no
    /// variants, so a miss here is an error rather than something left for
    /// typecheck's enum-constructor path. Returning the bare `sym` on failure
    /// keeps the tree well-formed for the rest of the pass; the error already
    /// stops compilation.
    fn builtin_qualified(&mut self, head: TyHead, ty: &str, sym: &'a str, span: &Span) -> &'a str {
        if let Some(m) = self.members.get(&(head, sym)) {
            self.check_member_visible(m, sym);
            return m.name;
        }
        self.push(Error::new(span.clone(),
            format!("no associated function '{}' on built-in type '{}'", sym, ty))
            .with_note(format!("declare one with `extend {} {{ proc {}(...) ... }}`", ty, sym)));
        sym
    }

    fn expr(&mut self, e: &mut Expr<'a>, gparams: &HashSet<&str>) {
        self.span = e.span.clone();
        match &mut e.value {
            ExprNode::Call { func, type_args, args } => {
                if let ExprNode::Path(path) = &mut func.value {
                    let span = func.span.clone();
                    if let Some(name) = self.value_path(path, &span, true, gparams) {
                        func.value = ExprNode::Var(name);
                    }
                } else {
                    self.expr(func, gparams);
                }
                for ga in type_args {
                    if let GenericArg::Type(t) = ga { self.ty(t, gparams); }
                }
                for a in args { self.expr(a, gparams); }
            }
            // a generic fn taken by value: resolve the name like a callee (so an
            // imported/module-qualified generic proc collapses to its mangled
            // template name, which typecheck/mono key on) and resolve the
            // turbofish types. The name lives in `name.path`, rewritten in place.
            ExprNode::FnRef { name, type_args } => {
                let span = e.span.clone();
                if let Some(resolved) = self.value_path(name, &span, true, gparams) {
                    name.path = Path { segments: vec![resolved] };
                }
                for ga in type_args {
                    if let GenericArg::Type(t) = ga { self.ty(t, gparams); }
                }
            }
            ExprNode::Struct { name, type_args, fields } => {
                if let Some(one) = name.path.as_single() {
                    match self.scopes.types.get(one) {
                        Some(sym) => name.def = sym.def,
                        None => self.error_here(format!("unknown struct '{}'", one)),
                    }
                } else if let Some((qual, sym)) = name.path.as_variant() {
                    if self.variant_path(name) {
                        // a struct-style enum variant literal, `Msg::Cc { id, val }`:
                        // type-qualified, not module-qualified. `def` is the enum;
                        // typecheck routes the rest to the variant.
                    } else if let Some(s) = self.scopes.quals.get(qual).and_then(|m| m.types.get(sym)) {
                        name.def = s.def;
                    } else if self.scopes.quals.contains_key(qual) {
                        self.error_here(format!(
                            "type '{}' is not imported from module qualifier '{}'", sym, qual));
                    } else {
                        self.error_here(format!(
                            "unknown module qualifier '{}' (did you `import .../{}`?)", qual, qual));
                    }
                } else {
                    self.error_here(format!(
                        "'{}' has too many `::` segments; only `qualifier::Type` is supported", name));
                }
                for a in type_args.iter_mut() {
                    if let GenericArg::Type(t) = a { self.ty(t, gparams); }
                }
                // a literal may be written through an alias (`Quad::<i32> { .. }`
                // for `type Quad<T> = Buf<T, 4>`), which names a struct just as
                // well as the struct's own name does. Redirect to what it expands
                // to, taking the arguments the alias supplied with it.
                self.expand_literal_head(name, type_args);
                for (_, fe) in fields { self.expr(fe, gparams); }
            }
            ExprNode::Access { base, .. } => self.expr(base, gparams),
            ExprNode::Index { slice, index } => {
                self.expr(slice, gparams);
                self.expr(index, gparams);
            }
            ExprNode::Unary { operand, .. } => self.expr(operand, gparams),
            ExprNode::Binary { left, right, .. } => {
                self.expr(left, gparams);
                self.expr(right, gparams);
            }
            ExprNode::Slice(elems) => for el in elems { self.expr(el, gparams); },
            ExprNode::Repeat { value, .. } => self.expr(value, gparams),
            // a name used as a value: rewrite it to the mangled top-level name,
            // unless a param/local shadows it (then it's a local read, leave it),
            // or it names an enum variant (which stays a `Path`). taking a
            // *generic* fn by value has no type args to monomorphize with; that's
            // handled (or rejected) downstream, not here.
            ExprNode::Path(path) => {
                let span = e.span.clone();
                if let Some(name) = self.value_path(path, &span, false, gparams) {
                    e.value = ExprNode::Var(name);
                }
            }
            // remaining leaves (literals): nothing to rewrite
            _ => {}
        }
    }

    fn stmt(&mut self, s: &mut Stmt<'a>, gparams: &HashSet<&str>) {
        self.span = s.span.clone();
        match &mut s.value {
            StmtNode::Expr(e) => self.expr(e, gparams),
            StmtNode::Block(ss) => {
                let mark = self.locals.len();
                for s in ss { self.stmt(s, gparams); }
                self.locals.truncate(mark); // drop names bound inside the block
            }
            StmtNode::Declare { ty, value, name } => {
                if let Some(ty) = ty { self.ty(ty, gparams); }
                self.expr(value, gparams); // walk the initializer BEFORE binding,
                self.locals.push(name);    // so `let f = f;` sees the outer/top f
            }
            StmtNode::Assign { left, value } => {
                self.expr(left, gparams);
                self.expr(value, gparams);
            }
            StmtNode::If { condition, then_branch, else_branch } => {
                self.expr(condition, gparams);
                self.stmt(then_branch, gparams);
                if let Some(eb) = else_branch { self.stmt(eb, gparams); }
            }
            StmtNode::While { condition, body } => {
                self.expr(condition, gparams);
                self.stmt(body, gparams);
            }
            // an arm's pattern needs two things: its `Enum::Variant` paths rewritten
            // to the enum's final name, and its field bindings in scope while the
            // arm body is walked - otherwise a read of `pitch` in
            // `Msg::Note(pitch, vel) -> ...` looks like an unresolved top-level
            // name. `Rewriter::pattern` does both.
            StmtNode::Match { scrutinee, arms } => {
                self.expr(scrutinee, gparams);
                for (pat, body) in arms {
                    let mark = self.locals.len();
                    let mut binds = Vec::new();
                    self.pattern(&mut pat.value, &mut binds);
                    self.locals.extend(binds);
                    self.stmt(body, gparams);
                    self.locals.truncate(mark);
                }
            }
            StmtNode::Return(Some(e)) => self.expr(e, gparams),
            StmtNode::Return(None) | StmtNode::Continue | StmtNode::Break => {}
        }
    }

    fn toplevel(&mut self, tl: &mut TopLevel<'a>) {
        // baseline span for anything in this item that has none of its own (field
        // types, a trait method's signature); `stmt`/`expr` narrow it as they go.
        self.span = tl.span.clone();
        match &mut tl.value {
            // an alias' own body was resolved before this pass began (it has to
            // be, since a *use* of it in any module expands during this one), and
            // the declaration itself is dropped when modules are merged. Nothing
            // left to rewrite.
            TopLevelNode::Alias { .. } => {}
            TopLevelNode::Function { name, def, generics, params, return_type, body, .. } => {
                let gparams = generic_names(generics);
                let sym = *self.scopes.calls.get(*name)
                    .expect("a module's own callable is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                self.bounds(generics);
                // a `const N: u64` generic param is read as an ordinary value in the
                // body (`i + N`), so it binds like a param. only *type* params go in
                // `gparams`, which is about type positions.
                for g in generics.iter() {
                    if let GenericParam::Const(cname, _) = g { self.locals.push(cname); }
                }
                // params are locals for the whole body; body-level `let`s stack on
                // top. truncate back to 0 so the next function starts clean.
                for (pname, ty) in params {
                    self.ty(ty, &gparams);
                    self.locals.push(pname);
                }
                self.ty(return_type, &gparams);
                for s in body { self.stmt(s, &gparams); }
                self.locals.clear();
            }
            TopLevelNode::Extern { name, def, generics, params, return_type, .. } => {
                let gparams = generic_names(generics);
                let sym = *self.scopes.calls.get(*name)
                    .expect("a module's own callable is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                for (_, ty) in params { self.ty(ty, &gparams); }
                self.ty(return_type, &gparams);
            }
            TopLevelNode::Struct { name, def, generics, fields, .. } => {
                // the struct's own type params shadow struct names when rewriting
                // field types (a field `T` is a param, not a module type).
                let gparams = generic_names(generics);
                let sym = *self.scopes.types.get(*name)
                    .expect("a module's own struct is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                self.bounds(generics);
                for (_, ty) in fields { self.ty(ty, &gparams); }
            }
            TopLevelNode::Global { name, def, ty, value, .. } => {
                let empty = HashSet::new();
                let sym = *self.scopes.calls.get(*name)
                    .expect("a module's own callable is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                self.ty(ty, &empty);
                self.expr(value, &empty);
            }
            // the enum's type name is now mangled like a struct's, so it is
            // rewritten here and every `E::V` reference is re-spelled to match (see
            // `variant_path`). A data-carrying variant's payload field types are
            // types like any other. The enum's own type params shadow module type
            // names (a payload `T` is a param, not a module type).
            TopLevelNode::Enum { name, def, generics, variants, .. } => {
                let gparams = generic_names(generics);
                let sym = *self.scopes.types.get(*name)
                    .expect("a module's own enum is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                self.bounds(generics);
                for (_, _, payload) in variants.iter_mut() {
                    for (_, ty) in payload.iter_mut() { self.ty(ty, &gparams); }
                }
            }
            // a trait's method signatures carry types (params + return) that must
            // be rewritten like any other - a `String` in `proc display(*self)
            // String` resolves to the imported struct's mangled name. `Self` is
            // left untouched (typecheck substitutes it per implementing type).
            // Each associated type (`type Item;`) is in scope for the signatures
            // as an implicit type parameter: `Self::Item` is rewritten to the
            // bare name first, then resolved to `Param("Item")` via `gparams`.
            TopLevelNode::Trait { name, def, assoc_types, methods, .. } => {
                let sym = *self.scopes.types.get(*name)
                    .expect("a module's own trait is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                let mut assoc: HashSet<&'a str> = HashSet::new();
                for a in assoc_types.iter() {
                    if *a == "Self" {
                        self.error_here("an associated type cannot be named 'Self'".into());
                    } else if !assoc.insert(*a) {
                        self.error_here(format!("duplicate associated type '{}'", a));
                    }
                }
                for m in methods.iter_mut() {
                    for (_, ty) in m.params.iter_mut() {
                        self.self_assoc(ty, &assoc);
                        self.ty(ty, &assoc);
                    }
                    self.self_assoc(&mut m.return_type, &assoc);
                    self.ty(&mut m.return_type, &assoc);
                }
            }
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before name resolution"),
        }
    }
}

/// Names introduced by an item's generic parameter list, of either kind. Used as
/// a skip-set when rewriting type positions: these are parameters of the item
/// being walked, not types imported from some module.
///
/// Const params belong here even though they aren't types: the parser can't tell
/// a const argument from a type argument in a turbofish, so `simd_load::<f32, N>`
/// arrives as `GenericArg::Type(Struct("N"))` and is only disambiguated later, in
/// typecheck. Treating `N` as an unknown type here would reject valid code.
fn generic_names<'a>(generics: &[GenericParam<'a>]) -> HashSet<&'a str> {
    generics.iter().map(|g| match g {
        GenericParam::Type { name, .. } => *name,
        GenericParam::Const(name, _) => *name,
    }).collect()
}

/// Allocate a `DefId` for every top-level item in `m` and build the symbol table
/// the module exposes.
///
/// This is where definition identity is minted. The emitted name in each `Sym`
/// now comes from `Defs::symbol` rather than being formatted here, so mangling
/// lives in exactly one function. The names themselves are unchanged in shape
/// (`<prefix>$<name>`); only the prefix's derivation moved, from an enqueue index
/// to the module's path.
fn build_symtab<'a>(m: &Module<'a>, defs: &mut Defs<'a>, arena: &'a Bump,
                    errs: &mut Vec<Error>) -> SymTab<'a> {
    let mut st = SymTab::default();
    let mut seen_types: HashSet<&'a str> = HashSet::new();
    // register the def, then ask `Defs` what it is emitted as.
    let def = |defs: &mut Defs<'a>, kind, name: &'a str, is_pub: bool,
                   linkage, span: &Span| -> DefId {
        defs.alloc(Def {
            module: m.mid, kind, source_name: name, is_pub, linkage, span: *span,
        })
    };

    // Two declarations of the same type name in one module. This used to be
    // caught in typecheck, whose tables were keyed by name so the second insert
    // collided; keyed by identity they no longer do, and the module's own symbol
    // table is where the collision is actually visible - and where the span
    // points at the right module.
    let mut dup: Vec<Error> = Vec::new();
    for tl in &m.items {
        let (kind, name) = match &tl.value {
            TopLevelNode::Struct { name, .. } => ("type", *name),
            TopLevelNode::Enum { name, .. } => ("type", *name),
            TopLevelNode::Trait { name, .. } => ("trait", *name),
            // an alias shares the type namespace, so `type Point = ...` beside a
            // `struct Point` is the same collision as two structs.
            TopLevelNode::Alias { name, .. } => ("type", *name),
            _ => continue,
        };
        if seen_types.contains(name) {
            dup.push(Error::new(tl.span.clone(), format!(
                "Duplicate {} definition '{}'", kind, name)));
        }
        seen_types.insert(name);
    }
    errs.append(&mut dup);

    for tl in &m.items {
        match &tl.value {
            TopLevelNode::Function { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Fn, name, *is_pub,
                    linkage_of(name, attributes, false), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            TopLevelNode::Extern { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Extern, name, *is_pub,
                    linkage_of(name, attributes, true), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            TopLevelNode::Struct { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Struct, name, *is_pub,
                    linkage_of(name, attributes, false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // enums live in the type namespace like structs, and are mangled like
            // them. That is only sound because every `E::V` reference - in call,
            // value, struct-literal *and* pattern position - is rewritten to the
            // enum's final name; leaving any one of those unrewritten is why enum
            // names used to be forced globally unique.
            TopLevelNode::Enum { name, is_pub, attributes, variants, .. } => {
                let id = def(defs, DefKind::Enum, name, *is_pub,
                    linkage_of(name, attributes, false), &tl.span);
                let sym = defs.symbol(id, arena);
                st.structs.insert(name, Sym { name: sym, def: id, is_pub: *is_pub });
                // a data variant's payload is laid out as its own struct, so it
                // needs an identity of its own. Minting it here - rather than
                // where the mid end first needs it - keeps typecheck from having
                // to invent identities, which matters because typecheck runs
                // twice (before and after monomorphization) and would otherwise
                // mint a second set on the second pass.
                for (vname, _, payload) in variants {
                    if payload.is_empty() { continue; }
                    defs.add_payload(id, vname, arena.alloc_str(&format!("{}${}", sym, vname)));
                }
            }
            // an alias is a name in the type namespace like a struct, and is
            // importable and qualifiable like one - it just never reaches codegen,
            // every use having been expanded by then. Its `Sym` name is never
            // emitted; only the identity is load-bearing.
            TopLevelNode::Alias { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Alias, name, *is_pub,
                    linkage_of(name, attributes, false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // globals live in the callable/value namespace (referenced as vars).
            TopLevelNode::Global { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Global, name, *is_pub,
                    linkage_of(name, attributes, false), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // traits live in the type namespace like structs/enums, and are mangled
            // like them. Sound because trait *bounds* (`T: Display`) and the trait
            // side of each conformance record are rewritten too; a trait declared
            // in one module no longer reserves its name program-wide.
            TopLevelNode::Trait { name, is_pub, .. } => {
                let id = def(defs, DefKind::Trait, name, *is_pub,
                    linkage_of(name, &[], false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // `extend` blocks were lowered to functions in `lower_methods`.
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before symtab"),
        }
    }
    st
}

/// A module we've decided to load but haven't parsed yet.
struct Pending<'a> {
    key: String,
    src: &'a str,
    dir: Option<PathBuf>,
    is_entry: bool,
    /// `Some` when this module came from a dependency artifact rather than the
    /// leaf's own source tree: carries the namespace it slugs under and its
    /// position within the dep, for resolving the dep's *internal* imports.
    dep: Option<DepCtx>,
}

/// Where a dependency module sits, so it slugs under the dep's package name and
/// its own relative imports resolve against the dep's artifact, not the disk.
struct DepCtx {
    /// the dependency's package name (also the namespace it was imported as).
    package: String,
    /// this module's directory within the dep, for joining onto its relative
    /// imports: `""` for a root-level module, `"dsp"` for `dsp/osc.hv`.
    reldir: String,
    /// the dep's package-root module, which slugs to the bare package name.
    is_root: bool,
}

/// Enqueue a dependency's whole module set, once. v1 pulls a dep in wholesale on
/// first touch: the artifact already carries all of it and the leaf recompiles
/// every module regardless.
///
/// Called from two places, which is the reason it is a function: lazily, when an
/// `import <name>/...` first names the dep, and eagerly for the package supplying
/// the prelude - which nothing has to import, and which would therefore never be
/// loaded at all.
fn enqueue_dep<'a>(name: &str, meta: &HavenMeta, arena: &'a Bump,
                   worklist: &mut VecDeque<Pending<'a>>,
                   seen: &HashMap<String, usize>,
                   dep_enqueued: &mut HashSet<String>)
{
    if !dep_enqueued.insert(name.to_string()) { return; }
    for dm in &meta.modules {
        let key = dep_key(name, &dm.key);
        if seen.contains_key(&key) { continue; }
        let reldir = dm.key.rsplit_once('/').map_or("", |(d, _)| d).to_string();
        worklist.push_back(Pending {
            key,
            src: arena.alloc_str(&dm.source),
            dir: None,
            is_entry: false,
            dep: Some(DepCtx {
                package: name.to_string(), reldir, is_root: dm.is_root,
            }),
        });
    }
}

/// Where a program's prelude comes from - the module whose `pub` items every
/// other module sees without an import, and the only package permitted to claim
/// lang items.
///
/// Naming the *package* rather than a module is deliberate: which of its modules
/// is the prelude is the package's own business, stated by the `@!prelude` mark
/// in its source. A caller that had to name the module would be back to knowing
/// another package's file layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PreludeSource<'p> {
    /// No prelude (`--no-prelude`): a freestanding build. Nothing is injected and
    /// no package may claim a lang item.
    None,
    /// No `--prelude` flag: *discover* it. The prelude is whichever loaded module
    /// carries the `@!prelude` mark - a bound dependency that advertises one wins
    /// (so a discovered `--dep std=...` needs no second flag), two providers is an
    /// error, and none at all (no user prelude dep, no discovered `std`) is an
    /// error, since the compiler embeds no fallback. The default. Resolved to a
    /// concrete choice at the top of [`load_and_merge`].
    Auto,
    /// A dependency's, named by `--prelude <name>` (the override); the name must
    /// also be bound by a `--dep`, or be the package being compiled (a stdlib's
    /// own build, whose `@!prelude` isn't visible until its modules are loaded).
    /// A nomination makes every *other* `@!prelude` inert, so a prelude-bearing
    /// package can still be consumed as a plain library.
    Package(&'p str),
}

impl<'p> PreludeSource<'p> {
    /// The providing package's name, for a diagnostic about a prelude that
    /// *exists*.
    ///
    /// Only [`PreludeSource::Package`] has one to give. Both callers are reached
    /// only in that case - one is guarded by `prelude != None` (and `Auto` cannot
    /// survive to there), the other by `prelude_mod` being `Some`, which requires
    /// a module that supplies the prelude. The fallback is therefore dead, and is
    /// deliberately not a plausible package name: answering `"std"` here is what
    /// made a no-prelude build report that only 'std' may declare a lang item
    /// *while pointing at std's own source*.
    fn provider(&self) -> &'p str {
        match self { PreludeSource::Package(n) => n, _ => "<no prelude>" }
    }

    /// Why the module in hand may not claim `item` as a lang item.
    ///
    /// Split by variant because the *reason* differs, and the difference is the
    /// whole diagnostic. With a real provider the claim belongs to someone else.
    /// With no prelude at all there is no one it could belong to - and naming a
    /// provider anyway (this used to answer `"std"` for every variant) accuses
    /// the reader of introducing a second stdlib when the file in hand may well
    /// *be* std's, sending them to look for a conflict that does not exist. The
    /// real fault in that case is upstream, in how the build was invoked.
    /// Returned as `(headline, note)`: the reason is several sentences long, and
    /// only the first belongs on the underline beside the attribute.
    fn lang_item_denial(&self, item: &str) -> (String, String) {
        match self {
            PreludeSource::Package(n) => (
                format!("`@{LANG_ATTR}({item})` may only be declared by '{n}'"),
                format!("'{n}' is the package this program takes its prelude from, and a \
                         lang item is one definition for the whole program, so a second \
                         package cannot introduce its own")),
            // `--no-prelude`. (`Auto` cannot reach here - it either resolves to a
            // `Package` or the build stops at "no `std` library found" - but the
            // wording holds if it ever does.)
            PreludeSource::None | PreludeSource::Auto => (
                format!("`@{LANG_ATTR}({item})` needs a prelude, and this build has none"),
                "a lang item is the prelude package's to declare, so with no prelude \
                 there is no package entitled to claim one - not even the package being \
                 compiled. If this package supplies the prelude, build it with \
                 `--prelude <its own name>`; otherwise remove the attribute".to_string()),
        }
    }
}

/// Whether a dependency's artifact advertises a prelude: does any of its modules
/// carry the `@!prelude` mark? Read by discovery to decide, with no nomination
/// flag, which bound package (if any) supplies the prelude.
///
/// Parses each module's *attributes* only, and via the real lexer - a `@!prelude`
/// written in a doc comment (as the embedded prelude's own documentation contains)
/// is not a mark and must not count. Bails at the first hit: this is a cheap
/// yes/no over source that is re-parsed in full later if the package turns out to
/// be the prelude. A module that fails to lex/parse here is skipped, not reported;
/// if it really is broken, loading it for real will say so with proper spans.
fn meta_provides_prelude(meta: &HavenMeta) -> bool {
    let scratch = Bump::new();
    for m in &meta.modules {
        let src: &str = scratch.alloc_str(&m.source);
        let (tokens, lex_errs) = parse::lex(FileId::UNKNOWN, src);
        let Some(tokens) = tokens.filter(|_| lex_errs.is_empty()) else { continue };
        let tokens = &*scratch.alloc_slice_fill_iter(tokens);
        let (parsed, parse_errs) = parse::parse(FileId::UNKNOWN, src.len(), tokens);
        if let Some((mod_attrs, _, _)) = parsed.filter(|_| parse_errs.is_empty()) {
            if mod_attrs.iter().any(|a| a.value.name == PRELUDE_ATTR) { return true; }
        }
    }
    false
}

/// Load `entry` and everything it transitively imports, then merge the lot into
/// one flat program. `prelude` says where the implicit prelude comes from: its
/// `pub` items become visible unqualified in every other module. The prelude
/// module is loaded like any other module of its package either way, so an
/// explicit `import std/prelude` is not a second copy of it.
///
/// `package` names the package being compiled, which anchors every emitted
/// symbol (`<package>.<relpath>$<name>`); `None` defaults it to the entry file's
/// stem, keeping a bare `havenc foo.hv` working. The package is rooted at the
/// entry file's directory, so its submodules must live under that directory. The
/// resolved package name is returned as the final tuple element.
///
/// `deps` maps a dependency name to the parsed `.hvmeta` artifact bound to it on
/// the command line. An `import <name>/<mod>` (or a bare `import <name>`) whose
/// leading segment is a key here resolves against that artifact's source, merged
/// under package name `<name>` — the same "load a package's source under a
/// namespace" path std travels, so the dep's items re-derive the exact
/// package-anchored symbols they would emit standalone. `std` is an ordinary key
/// in this map: binding `--dep std=<pkg>.hvmeta` makes that package *be* std for
/// the build, so `import std/...` resolves to it and the embedded tree is not
/// consulted at all. Absent such a binding, `std/...` falls through to the
/// embedded tree, which no dependency can then shadow or double-load.
///
/// [`PreludeSource::Package`] names one of those deps as the prelude's supplier.
/// It is loaded up front rather than on first import, since a program is not
/// obliged to import the package its prelude comes from.
/// `default_std` names the dependency the *compiler itself* discovered and bound
/// as std (always `"std"`, or `None` when the user bound their own or none was
/// found). It is a *fallback-priority* prelude provider: it supplies the prelude
/// only when no user-bound dependency does, and is otherwise excluded from prelude
/// discovery so it never collides with a prelude the user brought. In every other
/// respect it is an ordinary dep in `deps`.
pub fn load_and_merge<'a>(entry: &FilePath, package: Option<&str>, prelude: PreludeSource<'_>,
                          deps: &HashMap<String, HavenMeta>, default_std: Option<&str>, arena: &'a Bump)
    -> Result<(Vec<TopLevel<'a>>, Files<'a>, Defs<'a>, Vec<ImplDecl<'a>>, String), ()>
{
    let mut worklist: VecDeque<Pending<'a>> = VecDeque::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    // dependencies whose whole module set has already been enqueued, so a program
    // that imports both `foo/a` and `foo/b` pulls `foo` in exactly once.
    let mut dep_enqueued: HashSet<String> = HashSet::new();

    // entry module. canonicalize *first* so its key matches how imports are
    // keyed (imports always canonicalize). otherwise a module that imports the
    // entry back would key it differently, miss in `seen`, and get the entry
    // parsed + merged twice (dup `main`). if canonicalize fails we can't read it
    // anyway, so bail.
    let entry_path = match std::fs::canonicalize(entry) {
        Ok(p) => p,
        Err(e) => {
            diag::report_plain("Error", &format!("cannot read entry file '{}': {}", entry.display(), e));
            return Err(());
        }
    };
    let entry_src = match std::fs::read_to_string(&entry_path) {
        Ok(s) => arena.alloc_str(&s),
        Err(e) => {
            diag::report_plain("Error", &format!("cannot read entry file '{}': {}", entry_path.display(), e));
            return Err(());
        }
    };
    // the package is rooted at the entry file's directory: every non-std module's
    // slug is its path relative to here, so a build is reproducible across
    // machines (the canonical key is absolute, and would otherwise bake the
    // developer's home directory into every symbol).
    let entry_dir: Option<PathBuf> = entry_path.parent().map(|d| d.to_path_buf());

    // resolve the package name once. Absent an explicit one, the entry file's
    // stem stands in, so `havenc foo.hv` names its package `foo` with no flag.
    let package: String = package.map(str::to_string).unwrap_or_else(||
        entry_path.file_stem().map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "pkg".to_string()));

    // Discovery: with no `--prelude`, the prelude is found by the `@!prelude`
    // mark, not named on the command line. A bound dependency that advertises one
    // supplies it - so `--dep std=std.hvmeta` needs no second flag, and the
    // discovered std is not seeded alongside it (which would double-load the
    // prelude under one `std.` slug). Exactly one provider may win; two is the
    // same ambiguity `@lang` rejects, but resolvable, so it asks for `--prelude`
    // rather than guessing. No provider falls back to the discovered std. An
    // explicit `--prelude`/`--no-prelude` skips all of this - that is the override
    // half, and it also lets a prelude-bearing package be consumed as a plain
    // library. The package being *compiled* still nominates itself the flagged way
    // (`--prelude <self>`): its own mark is not visible here, before its modules
    // are loaded.
    let prelude = match prelude {
        PreludeSource::Auto => {
            // The compiler's own discovered std (`default_std`) is fallback-priority:
            // it is excluded from this scan so a user-bound prelude-bearing dep wins
            // outright, and is consulted only when no user dep supplies one. Without
            // this, binding your own prelude package alongside a compiler that found
            // std on disk would report a spurious two-prelude conflict.
            let mut providers: Vec<&str> = deps.iter()
                .filter(|(k, _)| Some(k.as_str()) != default_std)
                .filter_map(|(k, m)| meta_provides_prelude(m).then(|| k.as_str()))
                .collect();
            providers.sort_unstable();
            match providers.as_slice() {
                // no user dep supplies a prelude: use the discovered std if there is
                // one, else there is no prelude to seed (Auto errors at seeding).
                [] => match default_std.filter(|n| deps.contains_key(*n)) {
                    Some(n) => PreludeSource::Package(n),
                    None => PreludeSource::Auto,
                },
                [one] => PreludeSource::Package(one), // discovered, no flag needed
                many => {
                    diag::report_plain("Error", &format!(
                        "more than one dependency supplies a prelude ({}). A program \
                         has one prelude; nominate which with `--prelude <name>`, or \
                         drop all but one", many.join(", ")));
                    return Err(());
                }
            }
        }
        other => other,
    };

    // the prelude's package goes on the worklist ahead of the entry, so it reads
    // first in dumped output and its modules are in hand before anything that
    // might want them. (Order is cosmetic - a slug is derived from the module's
    // path, never from its position here.)
    match prelude {
        PreludeSource::None => {}
        // `Auto` still here means: no user dependency supplies a prelude, and no
        // `std` was discovered to fall back to (the compiler embeds none). A build
        // that wants no prelude says so with `--no-prelude`; reaching this is a
        // missing std, reported as such rather than as a later cascade of
        // unknown-name errors.
        PreludeSource::Auto => {
            diag::report_plain("Error",
                "no `std` library found: the compiler embeds none, and none was \
                 discovered on disk. Set $HAVEN_STD to a `std.hvmeta`, install one \
                 beside the `havenc` binary, bind one with `--dep std=<path>.hvmeta`, \
                 or build freestanding with `--no-prelude`");
            return Err(());
        }
        // the package being compiled supplies its own prelude. Nothing to seed -
        // its modules load from disk anyway - but the case has to exist, or a
        // prelude package could not be built at all: its `@lang` and `@!prelude`
        // would be claims by a package that, during its own build, is nobody's
        // prelude. This is how a stdlib is compiled.
        PreludeSource::Package(name) if name == package => {}
        // a dependency supplies it. Enqueued *eagerly*: the program is not
        // required to import the package it takes its prelude from - that is
        // rather the point - so nothing else would ever pull it in.
        PreludeSource::Package(name) => {
            let Some(meta) = deps.get(name) else {
                diag::report_plain("Error", &format!(
                    "--prelude names '{}', which is neither a dependency of this \
                     build nor the package being compiled. Bind it first, e.g. \
                     `--dep {}=<path>.hvmeta`", name, name));
                return Err(());
            };
            enqueue_dep(name, meta, arena, &mut worklist, &seen, &mut dep_enqueued);
        }
    }

    worklist.push_back(Pending {
        key: entry_path.to_string_lossy().into_owned(),
        src: entry_src,
        dir: entry_path.parent().map(|d| d.to_path_buf()),
        is_entry: true,
        dep: None,
    });

    // every source a diagnostic might point into, indexed by the `FileId` its
    // spans carry. filled in as modules are popped, so a lex/parse error can quote
    // the module that produced it.
    let mut files: Files<'a> = Files::new();
    // definition + module identities. modules register here as they're popped, so
    // a module's symbol slug is fixed before any of its items are named.
    let mut defs: Defs<'a> = Defs::new();
    let mut modules: Vec<Module<'a>> = Vec::new();
    let mut had_error = false;

    while let Some(p) = worklist.pop_front() {
        if seen.contains_key(&p.key) { continue; }
        let id = modules.len();
        seen.insert(p.key.clone(), id);

        // register the source before parsing: its spans carry this id, and any
        // lex/parse diagnostic has to be able to quote it.
        let file = files.add(p.key.clone(), p.src);
        let (mod_attrs, imports, mut items, impls, methods) = match parse_module(file, p.src, arena, &files) {
            Ok(pi) => pi,
            Err(()) => { had_error = true; continue; }
        };

        // `main` is the program's entry point: the C runtime that calls it needs a
        // fixed external symbol. The entry module is otherwise an ordinary package
        // module whose items mangle under the package namespace, so inject
        // `@export` onto its `main` now - before `build_symtab` reads attributes to
        // decide linkage. That keeps `main` `Fixed` (emitted verbatim, AST name
        // left as "main"), which both the linker and the driver's main-existence
        // check rely on. Doing it here rather than in the driver is what makes it
        // take effect: by the time the driver runs, linkage is already assigned and
        // every mangled name rewritten.
        if p.is_entry {
            for tl in &mut items {
                if let TopLevelNode::Function { name: "main", attributes, .. } = &mut tl.value {
                    if !is_export(attributes) {
                        attributes.push(Metadata::new(
                            AttributeNode::new("export", None), Span::unknown()));
                    }
                }
            }
        }

        // the module's symbol slug is derived from `(package, path relative to
        // the package root)` (see `defs::module_slug`), so it no longer shifts
        // when an unrelated import is added or removed - which the old
        // `m{id}_{basename}` prefix did, `id` being an enqueue index - nor when
        // the package is compiled from a different location. A dependency module
        // slugs under the *dep's* package name and its own synthetic root (the dep
        // name), so it re-derives the library's own `foo.geo` symbols rather than
        // the leaf's; the leaf's own modules use the leaf package + entry dir.
        let (slug_pkg, slug_root, is_root): (&str, Option<&FilePath>, bool) = match &p.dep {
            None => (&package, entry_dir.as_deref(), p.is_entry),
            Some(dc) => (&dc.package, Some(FilePath::new(&dc.package)), dc.is_root),
        };
        // where the source came from, which the slug deliberately does not
        // record: a dependency module and one of the leaf's own both slug from
        // `(package, relative path)`, and that is what keeps them
        // location-independent. Anything that needs to know whose source a
        // module *is* - the `.hvmeta` producer, which must ship this package's
        // own modules and no one else's - asks `Origin` instead of trying to
        // read it back out of the key. (`Origin::Std` is now vestigial: the
        // compiler embeds no std, so std arrives as an ordinary `Origin::Dep`.)
        let origin = match &p.dep {
            Some(_) => Origin::Dep,
            None => Origin::Own,
        };
        // whether this module belongs to the package supplying the prelude, and
        // may therefore claim to *be* the prelude or to declare a lang item.
        //
        // A separate question from `Origin`, which records where source came
        // from and is fixed for the build; this depends on what the build was
        // asked for. std is always a dependency now, so a build with no prelude
        // dependency (`--no-prelude`, or the resolved-away `Auto`) has no
        // prelude package at all.
        let prelude_pkg = match (&p.dep, prelude) {
            (Some(dc), PreludeSource::Package(n)) => dc.package == n,
            // the package being compiled is its own prelude: a stdlib's build.
            (None, PreludeSource::Package(n)) => origin == Origin::Own && package == n,
            (None, PreludeSource::Auto | PreludeSource::None) => false,
            (Some(_), _) => false,
        };
        let mid = match defs.add_module(slug_pkg, p.key.clone(), file, is_root, slug_root, origin) {
            Ok(mid) => mid,
            Err(msg) => {
                diag::report("Module error", &msg, &Span::new(file, 0, 0), &files);
                had_error = true;
                continue;
            }
        };

        // resolve + enqueue each import. errors point at the import statement in
        // this module. Resolution order is deliberate: a named external dependency
        // first (`import foo/...`, and `import std/...` too once `std` is *bound*
        // as one - see below), then, for a module that itself came from a dep,
        // that dep's own relative imports, and finally `std/...`-goes-embedded plus
        // the leaf's on-disk imports.
        let mut import_keys = Vec::with_capacity(imports.len());
        for imp in &imports {
            let first = imp.path.first().copied();

            // a named external dependency, `import foo/geo`. On first touch its
            // whole module set is enqueued, then this specific import is resolved
            // against it. (A dep supplying the prelude was already enqueued before
            // the loop, so `enqueue_dep` is a no-op for it here.)
            //
            // `std` is a dependency name like any other *when it is bound as one*
            // (`--dep std=<pkg>.hvmeta`): binding it replaces the embedded stdlib
            // wholesale, so every `import std/...` routes here instead of to the
            // embedded tree, and the embedded tree is never loaded for this build
            // (discovery up top makes that std the prelude too, rather than seeding
            // the embedded one alongside it - that pairing was the double-load
            // trap). Unbound, `deps.contains_key("std")` is false and this branch
            // is skipped, leaving `resolve_target` to serve `std/...` from the
            // embedded tree exactly as before. `self` is never a dep: it is the
            // reserved package-root anchor.
            if let Some(name) = first.filter(|f| *f != SELF_SEG && deps.contains_key(*f)) {
                let meta = &deps[name];
                enqueue_dep(name, meta, arena, &mut worklist, &seen, &mut dep_enqueued);
                match resolve_within_dep(meta, name, &imp.path[1..], arena) {
                    Ok(t) => import_keys.push(Some(t)),
                    Err(msg) => {
                        diag::report("Import error", &msg, &imp.span, &files);
                        had_error = true;
                        import_keys.push(None);
                    }
                }
                continue;
            }

            // a module that came from a dep resolves its *own* relative imports
            // (`import geo`, `import dsp/osc`) against that dep's artifact, not the
            // disk. Its whole module set is already enqueued, so nothing new is
            // pushed here — only the target is resolved for scope building.
            if let Some(dc) = &p.dep {
                // an *unbound* `std` still means the embedded tree, even for a
                // dep's own `import std/...` — so let it fall through to
                // `resolve_target` below. A bound `std` never reaches here: the
                // dependency branch above already claimed it.
                if first != Some("std") {
                    // `self/...` is already package-root relative, so it skips
                    // this module's own directory prefix - the one difference
                    // between the two spellings, here as on disk.
                    let segs: Vec<&str> = if first == Some(SELF_SEG) {
                        imp.path[1..].to_vec()
                    } else {
                        let mut s: Vec<&str> =
                            dc.reldir.split('/').filter(|s| !s.is_empty()).collect();
                        s.extend(imp.path.iter().copied());
                        s
                    };
                    let resolved = if first == Some(SELF_SEG) && segs.is_empty() {
                        Err(BARE_SELF.to_string())
                    } else {
                        resolve_within_dep(&deps[&dc.package], &dc.package, &segs, arena)
                    };
                    match resolved {
                        Ok(t) => import_keys.push(Some(t)),
                        Err(msg) => {
                            diag::report("Import error", &msg, &imp.span, &files);
                            had_error = true;
                            import_keys.push(None);
                        }
                    }
                    continue;
                }
            }

            // `std/...`, or the leaf's own on-disk relative import.
            let target = match resolve_target(imp, p.dir.as_deref(), entry_dir.as_deref(), arena) {
                Ok(t) => t,
                Err(msg) => {
                    diag::report("Import error", &msg, &imp.span, &files);
                    had_error = true;
                    import_keys.push(None);
                    continue;
                }
            };
            // a directory import pulls in every module under it: resolution
            // walks into the namespace, so the whole thing has to be present.
            let keys: Vec<&str> = match &target {
                ImportTarget::Module(k) => vec![k.as_str()],
                ImportTarget::Dir(ms) => ms.iter().map(|m| m.key.as_str()).collect(),
            };
            let mut failed = false;
            for key in keys {
                if seen.contains_key(key) { continue; }
                match load_import(imp, key, p.dir.as_deref(), arena) {
                    Ok((src, dir)) => worklist.push_back(Pending {
                        key: key.to_string(), src, dir, is_entry: false, dep: None,
                    }),
                    Err(msg) => {
                        diag::report("Import error", &msg, &imp.span, &files);
                        had_error = true;
                        failed = true;
                        break;
                    }
                }
            }
            import_keys.push(if failed { None } else { Some(target) });
        }

        modules.push(Module {
            file,
            mid,
            prelude_pkg,
            origin,
            mod_attrs,
            imports,
            import_keys,
            items,
            impls,
            methods,
        });
    }

    if had_error {
        return Err(());
    }

    // symbol table for every module (indexed by module id). also where each
    // top-level item gets its `DefId`.
    let mut errs: Vec<Error> = Vec::new();
    let symtabs: Vec<SymTab> = modules.iter()
        .map(|m| build_symtab(m, &mut defs, arena, &mut errs))
        .collect();
    // the prelude: the module whose `pub` items every other module sees without
    // an import. Found by the `@!prelude` it marks *itself* with, so the loader
    // no longer has to recognize it by the key it happened to be loaded under -
    // which was a rule about the embedded tree's directory layout masquerading
    // as a rule about the language (`std/prelude` vs `std/prelude.hv` vs a
    // package whose prelude is its root `lib.hv`).
    //
    // Marking is also the only workable rule once the prelude arrives as a
    // dependency: what a package calls its files is its own business, and
    // `.hvmeta` ships source with no manifest alongside it, so the mark has to
    // travel *in* the source.
    let mut prelude_mod: Option<usize> = None;
    let mut prelude_errs: Vec<Error> = Vec::new();
    for (i, m) in modules.iter().enumerate() {
        let Some(attr) = m.mod_attrs.iter().find(|a| a.value.name == PRELUDE_ATTR) else { continue };
        if m.prelude_pkg {
            // the package serving as the prelude: exactly one of its modules may
            // claim the role.
            if let Some(prev) = prelude_mod {
                prelude_errs.push(Error::new(attr.span.clone(),
                    format!("a second `@!{}`", PRELUDE_ATTR))
                    .with_label(attr.span.clone(),
                        format!("'{}' already claims it", files.path(modules[prev].file)))
                    .with_note("a program has one prelude or none"));
            } else {
                prelude_mod = Some(i);
            }
        } else if m.origin == Origin::Own && prelude != PreludeSource::None {
            // the *leaf* marked `@!prelude`, but its prelude comes from elsewhere
            // (a dependency that supplies one, or the embedded stdlib by default).
            // A dependency's stray mark is inert - a library is just a library to
            // its consumer - but the leaf's own mark is a mistake worth naming: it
            // does nothing, and a program has one prelude. To make *this* package
            // the prelude, nominate it (`--prelude`), which is also how a stdlib is
            // built.
            prelude_errs.push(Error::new(attr.span.clone(),
                format!("this `@!{}` has no effect", PRELUDE_ATTR))
                .with_label(attr.span.clone(), format!(
                    "'{}' already supplies this program's prelude", prelude.provider()))
                .with_note(format!(
                    "remove it, or build this package as the prelude with `--prelude {}`",
                    package)));
        }
        // a dependency's mark that isn't the chosen prelude stays inert.
    }
    // the prelude's package was enqueued, parsed, and then claimed nothing.
    // Loud, because the quiet version is a program compiled with no prelude in
    // scope at all - and the likeliest cause is a package (a nominated one, or a
    // discovered `std` whose artifact is wrong) that simply is not a prelude.
    // Only `Package` reaches here: `Auto` already errored at seeding when no std
    // was found, and `None` is guarded out just above.
    if prelude != PreludeSource::None && prelude_mod.is_none() && prelude_errs.is_empty() {
        let PreludeSource::Package(name) = prelude else {
            unreachable!("Auto errors at seeding and None is guarded out")
        };
        diag::report_plain("Error", &format!(
            "package '{}' supplies no prelude: none of its modules is marked `@!{}`. \
             Only a package that says it is one can be nominated with `--prelude` (or \
             discovered as `std`)", name, PRELUDE_ATTR));
        return Err(());
    }
    if !prelude_errs.is_empty() {
        for e in &prelude_errs { diag::report_error("Prelude error", e, &files); }
        return Err(());
    }
    // whether the prelude goes into *scope* implicitly is a separate question:
    // under `--no-prelude` a program may still `import std/prelude` by hand, and
    // that must stay an ordinary import rather than silently going implicit
    // everywhere. Its lang items still apply - the compiler knows what `Delete`
    // means however the trait was brought into scope.
    let prelude_id = if prelude == PreludeSource::None { None } else { prelude_mod };

    // lang items: the definitions the compiler itself knows about, found by the
    // `@lang(...)` each is marked with in source. Everything downstream reads
    // `defs.lang()`, so nothing else has to recognize a lang item - by name, by
    // declaring module, or by anything else it might coincidentally match.
    //
    // Marking beats locating. The predecessor rule was "the trait named `Delete`
    // declared in the prelude", which ties a compiler-known definition to where
    // it happens to be filed: the moment a lang item lives outside the prelude
    // module (`Clone`, in `std/clone.hv`) that anchor is gone.
    let mut lang = LangItems::default();
    let mut lang_errs: Vec<Error> = Vec::new();
    for (i, m) in modules.iter().enumerate() {
        for tl in &m.items {
            let TopLevelNode::Trait { name, attributes, .. } = &tl.value else { continue };
            let Some(attr) = attributes.iter().find(|a| a.value.name == LANG_ATTR) else { continue };
            // `check_attributes` already rejected a `@lang` with no value or an
            // unrecognized one, so this names something in `LANG_ITEMS`.
            let item = attr.value.value.as_deref().unwrap_or_default();

            // only the package supplying the prelude may claim a lang item, and
            // claiming one is not a local decision: `@lang(delete)` decides what
            // owning a resource *means* for every type in the program.
            //
            // This is where the rule stops matching `@!prelude`'s, and it is not
            // an oversight. An unnominated prelude package is still perfectly
            // usable as a library right up until it declares a lang item - at
            // which point two `Delete` traits exist and the ownership pass can
            // only insert calls for one of them, silently treating the other
            // package's owners as `Copy`. Ignoring the mark would be the
            // miscompile; erroring says plainly that two stdlibs do not go into
            // one program.
            if !m.prelude_pkg {
                let (msg, note) = prelude.lang_item_denial(item);
                lang_errs.push(Error::new(attr.span.clone(), msg).with_note(note));
                continue;
            }

            // the AST's `def` is filled by name resolution, which has not run
            // yet; the module's own symbol table already has the identity.
            let Some(sym) = symtabs[i].structs.get(*name) else { continue };
            let slot = match item {
                "delete" => &mut lang.delete,
                // reachable only if `LANG_ITEMS` grew without an arm here.
                other => {
                    lang_errs.push(Error::new(attr.span.clone(), format!(
                        "lang item '{}' is accepted by the parser but not wired up \
                         in the compiler", other)));
                    continue;
                }
            };
            if slot.is_some() {
                lang_errs.push(Error::new(attr.span.clone(), format!(
                    "duplicate `@{}({})`: the program already has one", LANG_ATTR, item)));
                continue;
            }
            *slot = Some(sym.def);
        }
    }
    if !lang_errs.is_empty() {
        for e in &lang_errs { diag::report_error("Lang item error", e, &files); }
        return Err(());
    }

    // A prelude that declares no `Delete` is an error rather than a silent skip.
    // The ownership pass treats a missing `Delete` as "nothing in this program
    // owns anything" - correct under `--no-prelude`, but if it could also mean
    // "nothing was found" then a mismatched stdlib would compile to a program
    // with no destructors at all, which is a miscompile and not a build failure.
    // Supplying a prelude and supplying the lang items are therefore one job:
    // a package cannot take over the first and leave the second to someone else.
    if let Some(pid) = prelude_mod {
        if lang.delete.is_none() {
            diag::report_plain("Error", &format!(
                "'{}' supplies this program's prelude but declares no `@{}(delete)` \
                 trait. The compiler needs it to know which types own memory and \
                 whose destructors to insert; without it every value would silently \
                 be treated as `Copy`",
                prelude.provider(), LANG_ATTR));
            return Err(());
        }
        defs.set_lang_items(modules[pid].mid, lang);
    }

    // re-exports: fold every `pub import`'s symbols into the importing module's
    // own export set, so a third module importing it sees them.
    //
    // This has to happen before any module's scopes are built, since a scope is
    // built from other modules' *export sets* - and it runs to a fixpoint rather
    // than in one pass, because a re-export can itself be re-exported (A pulls a
    // symbol from B, which pulled it from C) and module order says nothing about
    // which comes first. The set only grows and is bounded by modules x names,
    // so it terminates; an import cycle just stops adding.
    //
    // Nothing is copied but a visibility flag: the entry keeps the `DefId` and
    // emitted name it already had, which is what makes a re-exported type the
    // *same* type rather than a look-alike.
    let mut symtabs = symtabs;
    loop {
        let mut changed = false;
        for id in 0..modules.len() {
            for (imp, target) in modules[id].imports.iter().zip(&modules[id].import_keys) {
                if !imp.is_pub { continue; }
                let Some(ImportTarget::Module(key)) = target else { continue };
                let Some(syms) = &imp.symbols else { continue };
                let target = seen[key];
                for sym in syms {
                    for ns in [Namespace::Fns, Namespace::Structs] {
                        let Some(entry) = ns.get(&symtabs[target], sym).filter(|e| e.is_pub)
                        else { continue };
                        // a module's own declaration wins over anything it
                        // re-exports, exactly as it wins over a plain import.
                        let dst = ns.get_mut(&mut symtabs[id]);
                        if dst.contains_key(sym) { continue; }
                        dst.insert(sym, entry);
                        changed = true;
                    }
                }
            }
        }
        if !changed { break; }
    }

    // a whole-module `pub import` has no symbols to re-export - it binds a
    // qualifier, and passing a qualifier on to *this* module's importers would
    // need module-level namespaces the resolver doesn't have yet. Reject it
    // outright rather than silently doing nothing.
    for m in &modules {
        for imp in &m.imports {
            if imp.is_pub && imp.symbols.is_none() {
                errs.push(Error::new(imp.span.clone(),
                    format!("`pub import {}` re-exports nothing", imp.path.join("/")))
                    .with_label(imp.span.clone(), format!(
                        "this binds the qualifier '{}' rather than any names",
                        imp.path.last().unwrap()))
                    .with_note(format!(
                        "list the symbols to re-export, e.g. `pub import {} {{ ... }}`",
                        imp.path.join("/"))));
            }
        }
    }

    // pass 1: build every module's name-resolution scopes (owned; values are all
    // `&'a`, so `all_scopes` borrows nothing from `modules`/`symtabs`).
    let mut all_scopes: Vec<Scopes> = Vec::with_capacity(modules.len());
    for (id, m) in modules.iter().enumerate() {
        let mut scopes = Scopes::default();

        // 1. implicit prelude whole-module import (except into the prelude itself).
        //    only `pub` prelude items are pulled in, a private prelude helper
        //    stays local to the prelude.
        if let Some(pid) = prelude_id {
            if id != pid {
                for (&k, v) in &symtabs[pid].fns { if v.is_pub { scopes.calls.insert(k, *v); } }
                for (&k, v) in &symtabs[pid].structs { if v.is_pub { scopes.types.insert(k, *v); } }
            }
        }

        // 2. explicit imports
        let mut from_import_calls: HashSet<&str> = HashSet::new();
        let mut from_import_types: HashSet<&str> = HashSet::new();
        let mut qual_owner: HashMap<&str, String> = HashMap::new();

        for (imp, target) in m.imports.iter().zip(&m.import_keys) {
            let Some(target) = target else { continue };

            // a directory names no symbols of its own, so it can only be
            // imported whole - as a namespace.
            if let (ImportTarget::Dir(_), Some(_)) = (target, &imp.symbols) {
                errs.push(Error::new(imp.span.clone(), format!(
                    "'{}' is a directory of modules, not a module", imp.path.join("/")))
                    .with_label(imp.span.clone(), "a directory names no symbols to import")
                    .with_note(format!(
                        "import it whole (`import {}`) and reach its members through the \
                         qualifier, e.g. `{}::<module>::<name>`",
                        imp.path.join("/"), imp.path.last().unwrap())));
                continue;
            }

            let ImportTarget::Module(key) = target else {
                // a directory import: bind one qualifier whose children are the
                // modules under it, so `dsp::osc::Osc` walks `dsp` -> `osc` ->
                // the name. Members are namespaced by their path below the
                // directory, so a nested directory nests here too.
                let ImportTarget::Dir(members) = target else { unreachable!() };
                let qualifier = *imp.path.last().unwrap();
                let owner = imp.path.join("/");
                if let Some(prev) = qual_owner.get(qualifier) {
                    if *prev != owner {
                        errs.push(Error::new(imp.span.clone(), format!(
                            "qualifier '{}' already refers to a different module", qualifier)));
                    }
                }
                qual_owner.insert(qualifier, owner);
                for member in members {
                    let st = &symtabs[seen[&member.key]];
                    scopes.quals.entry(qualifier).or_default()
                        .child_at(&member.segments)
                        .fill_from(st);
                }
                continue;
            };

            let target_id = seen[key];
            let target = &symtabs[target_id];

            match &imp.symbols {
                None => {
                    // whole module: every symbol visible only as `qualifier::sym`,
                    // qualified under the module's last path segment.
                    let qualifier = *imp.path.last().unwrap();
                    let owner = imp.path.join("/");
                    if let Some(prev) = qual_owner.get(qualifier) {
                        if *prev != owner {
                            errs.push(Error::new(imp.span.clone(), format!(
                                "qualifier '{}' already refers to a different module", qualifier)));
                        }
                    }
                    qual_owner.insert(qualifier, owner);
                    // only `pub` items are importable; private ones are invisible
                    // outside their own module.
                    scopes.quals.entry(qualifier).or_default().fill_from(target);
                }
                Some(syms) => {
                    // selective: the named symbols visible unqualified. a name that
                    // is both a callable and a struct exists in both namespaces, so
                    // it lands in whichever the reference position asks for.
                    for sym in syms {
                        // a symbol may exist in the target but be private; keep the
                        // two cases apart so the diagnostic is actionable.
                        let exists = target.fns.contains_key(sym) || target.structs.contains_key(sym);
                        let mut imported = false;
                        if let Some(f) = target.fns.get(sym).filter(|f| f.is_pub) {
                            imported = true;
                            if from_import_calls.contains(sym) && scopes.calls.get(sym).map(|s| s.def) != Some(f.def) {
                                errs.push(Error::new(imp.span.clone(), format!(
                                    "'{}' is imported from more than one module", sym))
                                    .with_note("qualify it with a whole-module `import` instead"));
                            }
                            scopes.calls.insert(sym, *f);
                            from_import_calls.insert(sym);
                        }
                        if let Some(f) = target.structs.get(sym).filter(|f| f.is_pub) {
                            imported = true;
                            if from_import_types.contains(sym) && scopes.types.get(sym).map(|s| s.def) != Some(f.def) {
                                errs.push(Error::new(imp.span.clone(), format!(
                                    "struct '{}' is imported from more than one module", sym)));
                            }
                            scopes.types.insert(sym, *f);
                            from_import_types.insert(sym);
                        }
                        if !imported {
                            errs.push(if exists {
                                Error::new(imp.span.clone(), format!(
                                    "symbol '{}' of module '{}' is private", sym, imp.path.join("/")))
                                    .with_note("add `pub` to export it")
                            } else {
                                Error::new(imp.span.clone(), format!(
                                    "module '{}' has no exported symbol '{}'",
                                    imp.path.join("/"), sym))
                            });
                        }
                    }
                }
            }
        }

        // 3. this module's own defs win over imports (inserted last). a module
        //    always sees all of its own symbols, `pub` or not.
        for (&k, v) in &symtabs[id].fns { scopes.calls.insert(k, *v); }
        for (&k, v) in &symtabs[id].structs { scopes.types.insert(k, *v); }

        all_scopes.push(scopes);
    }

    // pass 1.25: work out each `extend` target's inferred type parameters and
    // push them onto the functions its methods desugared into. This could not
    // happen at desugaring time: telling the `T` of `extend [T]` from the
    // `Point` of `extend *Point` needs a type scope, and there wasn't one yet.
    // every struct and enum in the program, so an `extend` target's arguments can
    // be classified against what the type actually declares. built up front: the
    // loop below takes `modules` mutably, and a target may name a type from any
    // module, not only its own.
    //
    // keyed through each module's own type scope rather than the `def` field on
    // the node, which pass 2 has yet to fill in - and looked up the same way
    // `type_def_of` will look one up, so the two cannot disagree.
    let mut type_params: TypeParams<'a> = HashMap::new();
    for (id, m) in modules.iter().enumerate() {
        for tl in &m.items {
            let (TopLevelNode::Struct { name, generics, .. }
               | TopLevelNode::Enum { name, generics, .. }) = &tl.value else { continue };
            if let Some(sym) = all_scopes[id].types.get(*name) {
                type_params.insert(sym.def, generics.clone());
            }
        }
    }

    // pass 1.15: give every trait impl the default bodies it did not override.
    //
    // A default is *copied into the impl* as an ordinary method - the same shape
    // `lower_methods` produces for one written by hand - rather than dispatched to
    // at the call site. So conformance, member lookup, monomorphization and
    // codegen all see a perfectly normal method, and a trait is still nothing but
    // a set of signatures by the time anything typechecks.
    //
    // It runs here because this is the first point at which a trait declared in
    // one module can be found from another: `extend Gain: Mono` says only "Mono",
    // and deciding *which* `Mono` takes a resolved type scope. It must also
    // precede pass 1.25, which is what fills in the generics of every method that
    // desugared into a function.
    //
    // The copied body resolves in the **impl's** module, not the trait's - the
    // same rule the `for`-loop desugar already follows for the `Option` it
    // mentions. A default that names anything beyond the trait's own methods on
    // `self` is therefore only portable if that name is in scope where the impl is
    // written; `pub` re-exports from the trait's module are the way to guarantee
    // it.
    let mut trait_defaults: HashMap<DefId, Vec<TraitMethod<'a>>> = HashMap::new();
    let mut inherited: HashMap<&'a str, &'a str> = HashMap::new();
    for (id, m) in modules.iter().enumerate() {
        for tl in &m.items {
            let TopLevelNode::Trait { name, methods, .. } = &tl.value else { continue };
            let with_body: Vec<TraitMethod<'a>> =
                methods.iter().filter(|tm| tm.body.is_some()).cloned().collect();
            if with_body.is_empty() { continue }
            if let Some(sym) = all_scopes[id].types.get(*name) {
                trait_defaults.insert(sym.def, with_body);
            }
        }
    }
    for id in 0..modules.len() {
        if trait_defaults.is_empty() { break }
        // built first, then installed: the collection loop holds a shared borrow
        // of this module's impls and of its type scope, and installing needs a
        // mutable one of its items and its call scope.
        //
        // `inherited` is only for diagnostics - pass 2 reads it to explain a span
        // that lands in the trait's module.
        let mut synthesized: Vec<(&'a str, TopLevel<'a>, RawMethod<'a>)> = Vec::new();
        let mut used: HashSet<String> = HashSet::new();
        for ri in &modules[id].impls {
            let Some(sym) = all_scopes[id].types.get(ri.trait_) else { continue };
            let Some(defaults) = trait_defaults.get(&sym.def) else { continue };
            for tm in defaults {
                if ri.defined.contains(&tm.name) { continue }
                let mut params: Vec<(&'a str, Type<'a>)> =
                    Vec::with_capacity(tm.params.len() + 1);
                let self_ty = ri.target.clone();
                match tm.receiver {
                    Receiver::Associated => {}
                    Receiver::Value => params.push(("self", self_ty)),
                    Receiver::Pointer => params.push(("self", Type::Pointer(Box::new(self_ty)))),
                }
                params.extend(tm.params.iter().cloned());
                let head = match &ri.target {
                    Type::Path { path, .. } => Some(path),
                    _ => None,
                };
                let assoc = ri.assoc_bindings.as_slice();
                for (_, pty) in params.iter_mut() {
                    self_subst_type(pty, &ri.target, assoc);
                }
                let mut return_type = tm.return_type.clone();
                self_subst_type(&mut return_type, &ri.target, assoc);
                let mut body = tm.body.clone().expect("only defaulted methods are collected");
                for st in body.iter_mut() {
                    // the *only* place an AST subtree is duplicated: one trait
                    // body becomes a method of every impl that inherited it.
                    refresh_ids_stmt(st);
                    self_subst_stmt(st, &ri.target, head, assoc);
                }

                // `$default` keeps these out of the way of `lower_methods`' own
                // naming, so an inherent method of the same name collides as a
                // duplicate *member* in pass 1.5 - which says so - rather than as
                // two functions sharing one symbol, which says something else.
                let key = target_key(&ri.target);
                let mut base = format!("{}${}$default", key, tm.name);
                for i in 1.. {
                    if used.insert(base.clone()) { break }
                    base = format!("{}${}$default${}", key, tm.name, i);
                }
                let fname: &'a str = arena.alloc_str(&base);
                synthesized.push((
                    fname,
                    Metadata::new(
                        TopLevelNode::Function {
                            name: fname,
                            def: DefId::UNRESOLVED,
                            // as for a hand-written trait-impl method: the
                            // *function* is private, the *member* is not.
                            is_pub: false,
                            attributes: Vec::new(),
                            // a trait method declares no parameters of its own;
                            // the impl's are prepended by pass 1.25.
                            generics: Vec::new(),
                            where_bounds: Vec::new(),
                            params,
                            return_type,
                            body,
                        },
                        ri.span.clone(),
                    ),
                    RawMethod {
                        target: ri.target.clone(),
                        generics: Vec::new(),
                        where_bounds: ri.where_bounds.clone(),
                        name: tm.name,
                        fn_name: fname,
                        receiver: tm.receiver,
                        // a trait-impl method's reachability follows the trait.
                        is_pub: true,
                        default_of: Some(ri.trait_),
                        span: ri.span.clone(),
                    },
                ));
            }
        }
        for (fname, item, rm) in synthesized {
            // minted here rather than in `build_symtab`, which ran before this
            // function existed. Only the *declaring* module needs to see the
            // name: every other module reaches the method through the member
            // table, which pass 1.5 fills in from the `RawMethod`.
            let did = defs.alloc(Def {
                module: modules[id].mid,
                kind: DefKind::Fn,
                source_name: fname,
                is_pub: false,
                linkage: linkage_of(fname, &[], false),
                span: item.span,
            });
            let sym = Sym { name: defs.symbol(did, arena), def: did, is_pub: false };
            all_scopes[id].calls.insert(fname, sym);
            symtabs[id].fns.insert(fname, sym);
            inherited.insert(fname, rm.default_of.expect("synthesized from a default"));
            modules[id].items.push(item);
            modules[id].methods.push(rm);
        }
    }

    // pass 1.2: resolve every type alias' body, in the module that declared it.
    //
    // Has to happen before anything resolves a type, since a *use* of an alias in
    // any module expands during pass 2 and needs the body ready. Round-based
    // rather than recursive: an alias whose body names another can only be
    // resolved once that one is, so the rounds simply repeat until a round
    // resolves nothing new. Whatever is left over then is a cycle - `type A = B;
    // type B = A;` has no expansion, and looping on it would not find one.
    let mut aliases: Aliases<'a> = HashMap::new();
    // every alias in the program, and (module, def, generics, body as written,
    // span) for each one still to resolve.
    let mut declared: HashSet<DefId> = HashSet::new();
    let mut pending: Vec<(usize, DefId, Vec<GenericParam<'a>>, Type<'a>, Span)> = Vec::new();
    for (id, m) in modules.iter().enumerate() {
        for tl in &m.items {
            let TopLevelNode::Alias { name, generics, ty, .. } = &tl.value else { continue };
            let Some(sym) = all_scopes[id].types.get(*name) else { continue };
            declared.insert(sym.def);
            pending.push((id, sym.def, generics.clone(), ty.clone(), tl.span.clone()));
        }
    }
    let no_members_yet = MemberTable::new();
    while !pending.is_empty() {
        let mut progressed = false;
        let mut deferred = Vec::new();
        for (id, def, generics, ty, span) in pending {
            let scopes = &all_scopes[id];
            if !alias_deps_ready(&ty, scopes, &declared, &aliases) {
                deferred.push((id, def, generics, ty, span));
                continue;
            }
            let mut rw = Rewriter {
                scopes,
                type_params: &type_params,
                aliases: &aliases,
                in_default_of: None,
                // resolving a type consults no members, so an empty table is not
                // a limitation here - see `resolve_extend_target`.
                members: &no_members_yet,
                module: modules[id].mid,
                errs: &mut errs,
                locals: Vec::new(),
                span: span.clone(),
            };
            let mut body = ty;
            rw.ty(&mut body, &generic_names(&generics));
            aliases.insert(def, AliasDef { generics, ty: body, span });
            progressed = true;
        }
        if !progressed {
            for (_, _, _, ty, span) in deferred {
                errs.push(Error::new(span.clone(), "type alias is cyclic".to_string())
                    .with_label(span, format!("expanding it reaches itself through '{}'", ty))
                    .with_note("an alias is expanded, not defined - it cannot name itself, \
                                directly or through another alias"));
            }
            break;
        }
        pending = deferred;
    }

    for (id, m) in modules.iter_mut().enumerate() {
        let scopes = &all_scopes[id];
        let known = |n: &str| scopes.types.contains_key(n);
        let decl = |path: &Path<'a>| type_def_of(scopes, path)
            .and_then(|d| type_params.get(&d))
            .map(Vec::as_slice);
        // one `where` clause is copied onto every method of its block, so an
        // error in it would otherwise be reported once per method.
        let mut reported: HashSet<(usize, &'a str)> = HashSet::new();
        for rm in m.methods.iter_mut() {
            rm.generics = impl_generics(&rm.target, &known, &decl);
            apply_where_bounds(
                &mut rm.generics, &rm.where_bounds, WhereOwner::Extend(&rm.target), &rm.span,
                &mut reported, &mut errs);
        }
        for ri in m.impls.iter_mut() {
            ri.generics = impl_generics(&ri.target, &known, &decl);
            apply_where_bounds(
                &mut ri.generics, &ri.where_bounds, WhereOwner::Extend(&ri.target), &ri.span,
                &mut reported, &mut errs);
        }

        // the impl's parameters lead the method's own, so `self`'s type resolves
        // against them and a turbofish on the desugared function stays in source
        // order (`extend [T] { proc map<U>(...) }` -> `<T, U>`).
        let mut by_fn: HashMap<&'a str, &RawMethod<'a>> =
            m.methods.iter().map(|rm| (rm.fn_name, rm)).collect();
        for tl in m.items.iter_mut() {
            let TopLevelNode::Function { name, generics, where_bounds, .. } = &mut tl.value
                else { continue };
            let span = tl.span.clone();
            let fn_name = *name;
            let own_where = std::mem::take(where_bounds);
            let Some(rm) = by_fn.remove(fn_name) else {
                // a plain top-level `proc`: its binder is written out, so its
                // clause has only that to merge onto.
                apply_where_bounds(
                    generics, &own_where, WhereOwner::Proc(fn_name), &span,
                    &mut reported, &mut errs);
                continue;
            };
            for ig in &rm.generics {
                let ig_name = match ig {
                    GenericParam::Type { name, .. } => *name,
                    GenericParam::Const(name, _) => *name,
                };
                if generics.iter().any(|g| match g {
                    GenericParam::Type { name, .. } => *name == ig_name,
                    GenericParam::Const(name, _) => *name == ig_name,
                }) {
                    errs.push(Error::new(rm.span.clone(), format!(
                        "generic parameter '{}' shadows one from the `extend` target",
                        ig_name))
                        .with_label(rm.span.clone(),
                            format!("method '{}' declares it again here", rm.name))
                        .with_note("rename one of them"));
                }
            }
            let own = std::mem::take(generics);
            generics.extend(rm.generics.iter().cloned());
            generics.extend(own);
            // deliberately last: a method's own clause may name either one of
            // its own parameters or one the `extend` target bound, and only now
            // are both in `generics`.
            apply_where_bounds(
                generics, &own_where, WhereOwner::Proc(rm.name), &span,
                &mut reported, &mut errs);
        }
    }

    // pass 1.5: record every method under its target's *head*, before any module
    // is rewritten. has to precede pass 2 because a call `Point::new()` in module
    // A resolves through a member declared in module B.
    //
    // Resolving the target needs a `Rewriter`, which wants a member table it will
    // never consult (type resolution touches no members) - so it gets an empty
    // one, leaving `defs`'s free to be written to here.
    let no_members = MemberTable::new();
    for (id, m) in modules.iter().enumerate() {
        let scopes = &all_scopes[id];
        for rm in &m.methods {
            // an `extend` on an unknown type is reported when the block's `self`
            // parameter is resolved in pass 2; drop the errors from this
            // speculative resolution so it isn't reported twice, and skip the
            // member rather than key it on a type that doesn't exist.
            let Some((self_ty, head)) =
                resolve_extend_target(&rm.target, &rm.generics, scopes, &type_params, &aliases,
                                      &no_members, m.file)
            else { continue };
            // an associated function is only ever reached by naming its type, so
            // one declared on a target no path can name could never be called.
            // Rejecting it here beats emitting a symbol with no way in.
            if rm.receiver == Receiver::Associated && !head.is_nameable() {
                errs.push(Error::new(rm.span.clone(),
                    format!("associated function '{}' cannot be reached", rm.name))
                    .with_label(rm.span.clone(), format!(
                        "a call would have to name '{}', which is not a path", rm.target))
                    .with_note("only a declared type or a built-in keyword can be written \
                                as a path; give it a `self` parameter so it is found \
                                through its receiver instead"));
            }
            let f = scopes.calls.get(rm.fn_name).map(|s| s.name).unwrap_or(rm.fn_name);
            let prev = defs.add_member(head, rm.name, Member {
                name: f,
                receiver: rm.receiver,
                self_ty,
                generics: resolve_bound_defs(&rm.generics, scopes),
                is_pub: rm.is_pub,
                module: m.mid,
            });
            // one impl per `(head, method)`: see `MemberTable`. Two `extend`
            // blocks reaching the same slot are ambiguous at every call site, so
            // this is an error rather than a silent last-one-wins.
            if prev.is_some() {
                // a default body is not written in the block it lands in, so
                // "second definition" pointing at that block would send the
                // reader looking for a method that is not there.
                let err = match rm.default_of {
                    Some(tr) => Error::new(rm.span.clone(), format!(
                        "method '{}' is already defined for this type", rm.name))
                        .with_label(rm.span.clone(), format!(
                            "this impl inherits '{}' as a default from trait '{}'",
                            rm.name, tr))
                        .with_note("define it in this block to override the default, \
                                    or rename the other one"),
                    None => Error::new(rm.span.clone(), format!(
                        "method '{}' is already defined for this type", rm.name))
                        .with_label(rm.span.clone(), "second definition")
                        .with_note("a second `extend` block cannot add or specialize a method \
                                    (`extend [T]` and `extend [i32]` both claim every slice)"),
                };
                errs.push(err);
            }
        }
    }

    // pass 2: rewrite each module's items in place using its scopes. Also remap
    // each `extend T: Trait` conformance record to final (mangled) names through
    // the same scopes, so the typechecker matches them against the merged program.
    let mut impls: Vec<ImplDecl<'a>> = Vec::new();
    for (id, m) in modules.iter_mut().enumerate() {
        let scopes = &all_scopes[id];
        let mut rw = Rewriter {
            scopes,
            type_params: &type_params,
            aliases: &aliases,
            in_default_of: None,
            members: defs.members(),
            module: m.mid,
            errs: &mut errs,
            locals: Vec::new(),
            span: Span::new(m.file, 0, 0),
        };
        for tl in &mut m.items {
            rw.in_default_of = match &tl.value {
                TopLevelNode::Function { name, .. } => inherited.get(*name).copied(),
                _ => None,
            };
            rw.toplevel(tl);
        }
        rw.in_default_of = None;
        for imp in &m.impls {
            // as above: an `extend T: Trait` naming an unknown `T` or `Trait`
            // has already produced an error through the type/bound paths, so
            // dropping the record here loses no diagnostic.
            let (Some((self_ty, head)), Some(trait_)) = (
                resolve_extend_target(&imp.target, &imp.generics, scopes, &type_params, &aliases,
                                      &no_members, m.file),
                scopes.types.get(imp.trait_),
            ) else { continue };
            // resolve each `type Item = Ty` binding's right-hand side with the
            // impl's inferred parameters in scope, so `type Item = T` in
            // `extend Vec<T>: Iterator` binds to `Param("T")`. A binding to an
            // unknown type is reported here (through `rw`'s error sink) rather
            // than dropped, since nothing else revisits the binding.
            let gp = generic_names(&imp.generics);
            let assoc_bindings = imp.assoc_bindings.iter().map(|(n, ty)| {
                let mut bty = ty.clone();
                rw.ty(&mut bty, &gp);
                (*n, bty)
            }).collect();
            impls.push(ImplDecl {
                self_ty,
                head,
                // a `where` clause's bound names are resolved here for the same
                // reason the member table resolves them (pass 1.5): pass 1.25
                // merged the clause in before pass 2 knew the trait's `def`, so
                // the raw copy still carries `UNRESOLVED`. `implements` keys on
                // that `def`, so without this a conditional impl (`Vec<T>: Display
                // where T: Display`) never satisfies a bound on the whole `Vec<T>`.
                generics: resolve_bound_defs(&imp.generics, scopes),
                trait_: trait_.def,
                assoc_bindings,
                span: imp.span.clone(),
            });
        }
    }

    if !errs.is_empty() {
        for e in &errs { diag::report_error("Import error", e, &files); }
        return Err(());
    }

    // merge: concatenate all modules into one flat program, reconciling the
    // global callable namespace (functions + externs, keyed by final emitted
    // name) as we go. two identical `extern` declarations of the same C symbol
    // are the legit duplicate - de-duped silently. everything else that collides
    // on a final name would become a duplicate LLVM symbol at link time, so we
    // turn it into a source diagnostic here instead:
    //   * two externs, same name, different signature -> mismatched-ABI error
    //   * any other same-name collision (two defs, or a def vs an extern) ->
    //     already-defined error. covers colliding `@export`s and a user fn
    //     shadowing a prelude extern, which the old single-file path caught too.
    // structs live in a separate namespace; they don't participate here.
    let mut out: Vec<TopLevel<'a>> = Vec::new();
    let mut merge_errs: Vec<Error> = Vec::new();
    let mut seen_callables: HashMap<&str, &TopLevel<'a>> = HashMap::new();
    // globals share the value namespace but carry no signature, so they get their
    // own dedup keyed by final name. (a global colliding with a *function* name is
    // left for the downstream duplicate-symbol check.)
    let mut seen_globals: HashSet<&str> = HashSet::new();
    for m in &modules {
        for tl in &m.items {
            let (name, is_extern, params, ret) = match &tl.value {
                TopLevelNode::Function { name, params, return_type, .. } =>
                    (*name, false, params.as_slice(), return_type),
                TopLevelNode::Extern { name, params, return_type, .. } =>
                    (*name, true, params.as_slice(), return_type),
                TopLevelNode::Struct { .. } => { out.push(tl.clone()); continue; }
                TopLevelNode::Enum { .. } => { out.push(tl.clone()); continue; }
                TopLevelNode::Trait { .. } => { out.push(tl.clone()); continue; }
                // every use of an alias has been expanded into what it names, so
                // the declaration has nothing left to say. Dropping it here is
                // what keeps `TopLevelNode::Alias` a front-end-only variant.
                TopLevelNode::Alias { .. } => continue,
                TopLevelNode::Extend { .. } => unreachable!("extend desugared before merge"),
                TopLevelNode::Global { name, .. } => {
                    if !seen_globals.insert(*name) {
                        merge_errs.push(Error::new(tl.span.clone(), format!(
                            "global '{}' is already defined", name)));
                    } else {
                        out.push(tl.clone());
                    }
                    continue;
                }
            };

            if let Some(prev) = seen_callables.get(name) {
                let (prev_extern, prev_params, prev_ret) = match &prev.value {
                    TopLevelNode::Function { params, return_type, .. } =>
                        (false, params.as_slice(), return_type),
                    TopLevelNode::Extern { params, return_type, .. } =>
                        (true, params.as_slice(), return_type),
                    TopLevelNode::Struct { .. } | TopLevelNode::Enum { .. }
                    | TopLevelNode::Trait { .. } | TopLevelNode::Alias { .. } => unreachable!("only callables are recorded"),
                    TopLevelNode::Extend { .. } => unreachable!("extend desugared before merge"),
                    TopLevelNode::Global { .. } => unreachable!("globals are deduped separately"),
                };
                // signatures match on types only (param names are irrelevant to the ABI)
                let sig_eq = params.len() == prev_params.len()
                    && params.iter().zip(prev_params).all(|((_, a), (_, b))| a == b)
                    && ret == prev_ret;

                if is_extern && prev_extern {
                    if sig_eq {
                        continue; // same extern already emitted: legit dedup
                    }
                    merge_errs.push(Error::new(tl.span.clone(), format!(
                        "extern '{}' is redeclared with a different signature", name)));
                } else {
                    merge_errs.push(Error::new(tl.span.clone(), format!(
                        "'{}' is already defined", name)));
                }
                continue;
            }

            seen_callables.insert(name, tl);
            out.push(tl.clone());
        }
    }

    if !merge_errs.is_empty() {
        for e in &merge_errs { diag::report_error("Merge error", e, &files); }
        return Err(());
    }

    // the resolved package name is returned so the driver names artifacts (a
    // lib's `.hvmeta`) under exactly the name that shaped the symbols, with no
    // second, drift-prone re-derivation of the entry-stem default.
    Ok((out, files, defs, impls, package))
}
