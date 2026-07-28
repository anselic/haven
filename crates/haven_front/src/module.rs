//! module loading, name mangling and merging.
//!
//! sits between parsing and typechecking. given an entry file it:
//!
//!   1. transitively loads every imported module (`import std/...` -> an embedded
//!      stdlib source; any other path -> an `.hv` file relative to the
//!      *importing* file's dir);
//!   2. gives each module a unique mangling prefix and renames its top-level defs
//!      (`foo` in module `m1` -> `m1_...$foo`), leaving `extern` link names,
//!      `@export`ed items and the entry `main` alone;
//!   3. rewrites every reference (call targets, struct literals, struct types)
//!      per that module's imports;
//!   4. concatenates all modules into one flat program, no imports left.
//!
//! everything downstream (typecheck, mono, mil, ...) then sees one flat namespace
//! exactly like before modules existed. lifetimes stay trivial: every module's
//! source, tokens and freshly-minted (mangled) names go into one bumpalo arena
//! owned by the caller, and the returned AST borrows it for `'a`.
//!
//! ## import forms
//!
//! * `import std/math`            - whole module, symbols visible only as
//!   `math::sinf` (qualified under the last path segment).
//! * `import std/math { sinf }`   - selective, the named symbols visible
//!   unqualified as `sinf`.
//!
//! the prelude is an implicit import into every user module, with its `pub`
//! symbols visible unqualified (it has no qualifier spelling). it is an
//! *ordinary* std module, loaded from the embedded tree under the key
//! `std/prelude` like any other - so writing `import std/prelude` explicitly (or
//! pulling it in as part of a `import std` directory import) resolves to the
//! same already-loaded module, rather than a second copy of every prelude type
//! under its own identities.
//!
//! ## visibility
//!
//! top-level items are module-private by default; prefix an item with `pub` to
//! make it importable by other modules. privacy is enforced purely at import
//! resolution: a non-`pub` item never enters another module's `SymTab`, so it
//! can't be named through a whole-module (`qual::sym`) or selective import, nor
//! pulled in by the implicit prelude import. a module always sees all of its own
//! items regardless of `pub`. `pub` is orthogonal to the `@export` attribute,
//! which controls LLVM linkage / name mangling, not cross-module visibility.
//!
//! ## unresolved names
//!
//! a name that resolves to nothing is an error *here*, not a silent passthrough.
//! letting one through used to leave the source spelling in the AST, where it
//! then failed far downstream as a mismatch against some other module's mangled
//! name - e.g. a `String` in a prelude trait signature, unresolved because the
//! prelude imports no `String`, reported three stages later as an impl whose
//! signature "does not match the trait".
//!
//! three things legitimately don't resolve through a module scope, and each is an
//! explicit branch rather than a fallthrough: compiler intrinsics (`sizeof`,
//! `__simd_*` - in no symbol table), `Self` in a trait method signature
//! (substituted per impl by typecheck), and generic parameters of the enclosing
//! item - including *const* params, which the parser can't tell from type
//! arguments inside a turbofish.
//!
//! ## namespacing
//!
//! every kind of top-level item is namespaced by its module. structs, enums and
//! traits alike are emitted as `<module slug>$<name>`, so two modules may declare
//! the same type name and a private one reserves nothing program-wide.
//!
//! for enums and traits that only holds because *every* reference to them is
//! rewritten: an `Enum::Variant` path in call, value, struct-literal and match
//! pattern position (see `variant_path`), and a bare `T: Trait` bound (see
//! `bounds`). leaving any one of those unrewritten is why both used to be emitted
//! unmangled, and therefore had to be globally unique.
//!
//! ## known v1 limitations
//!
//! * visibility is item-level only; struct *fields* are always public.

use std::collections::{HashMap, HashSet, VecDeque};
// `Path` is `ast::Path` here - a `::`-separated name. The filesystem one is
// aliased, since this module talks about names far more often than files.
use std::path::{Path as FilePath, PathBuf};

use bumpalo::Bump;

use haven_common::ast::*;
use crate::parse;
use haven_common::defs::{Def, DefKind, DefId, Defs, Linkage, Member, MemberTable, ModId};

use haven_common::diag::{self, Files};
use haven_common::intrinsics::Intrinsic;

/// The whole `crt/std` tree, embedded into the binary at build time. Nested
/// modules just work: `std/dsp/osc` -> `crt/std/dsp/osc.hv`, no per-file wiring.
static STD_DIR: include_dir::Dir<'static> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../std");

/// The implicit prelude's canonical key. It is a plain `std/...` module: this is
/// exactly the key `resolve_target` produces for an explicit `import
/// std/prelude`, so the two share one entry in `seen` and therefore one set of
/// definitions.
const PRELUDE_KEY: &str = "std/prelude";

/// Source of an embedded stdlib module, by its `std/...` import path. The key is
/// `imp.path.join("/")` (always forward slashes), so `std/<rel>` maps to the
/// embedded file `<rel>.hv`.
fn std_source(key: &str) -> Option<&'static str> {
    let rel = key.strip_prefix("std/")?;
    // include_dir keys files by forward-slash path relative to the embedded root,
    // on every host platform.
    let file = format!("{rel}.hv");
    STD_DIR.get_file(&file)?.contents_utf8()
}

/// A loaded module: parsed contents plus the bookkeeping the resolver needs to
/// mangle and rewrite it
struct Module<'a> {
    /// this module's entry in the `Files` table: its spans point at this id, and
    /// its canonical key (absolute path, or `std/...`) and source text are
    /// stored there rather than duplicated here.
    file: FileId,
    /// this module's entry in `Defs`, which owns its symbol slug.
    mid: ModId,
    is_entry: bool,
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

/// Whether an item keeps its source spelling as its emitted symbol, or gets the
/// module slug prefixed. One rule, one place - this used to be re-derived by
/// three near-identical `final_*_name` helpers, of which only the fn one knew
/// about `extern`.
///
/// * `extern` - the name *is* the C link symbol.
/// * `@export` - a host looks the symbol up by that name.
/// * the entry module - its names can't clash with the mangled imported ones,
///   and single-file diagnostics read better unmangled.
fn linkage_of<'a>(m: &Module<'_>, name: &'a str, attrs: &[Attribute], is_extern: bool) -> Linkage<'a> {
    if is_extern || is_export(attrs) || m.is_entry {
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

/// Every `.hv` file under the embedded std directory `rel`, as
/// `(segments below `rel`, std key)`. Recurses, so a nested directory becomes a
/// nested namespace.
fn std_dir_members<'a>(rel: &str, arena: &'a Bump) -> Option<Vec<DirMember<'a>>> {
    fn walk<'a>(d: &include_dir::Dir<'_>, prefix: &[&'a str], arena: &'a Bump,
                out: &mut Vec<DirMember<'a>>) {
        for f in d.files() {
            let Some(stem) = f.path().file_stem().and_then(|s| s.to_str()) else { continue };
            if f.path().extension().and_then(|e| e.to_str()) != Some("hv") { continue }
            let mut segments = prefix.to_vec();
            segments.push(arena.alloc_str(stem));
            let key = format!("std/{}", f.path().with_extension("").to_string_lossy()
                .replace('\\', "/"));
            out.push(DirMember { segments, key });
        }
        for sub in d.dirs() {
            let Some(name) = sub.path().file_name().and_then(|s| s.to_str()) else { continue };
            let mut prefix = prefix.to_vec();
            prefix.push(arena.alloc_str(name));
            walk(sub, &prefix, arena, out);
        }
    }
    let d = STD_DIR.get_dir(rel)?;
    let mut out = Vec::new();
    walk(d, &[], arena, &mut out);
    Some(out)
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

/// Resolve an import to the module - or directory of modules - it names. No file
/// read for `std`; for the relative case canonicalizes (so the path has to
/// exist). `dir` is the importing module's directory.
fn resolve_target<'a>(imp: &Import, dir: Option<&FilePath>, arena: &'a Bump)
    -> Result<ImportTarget<'a>, String>
{
    if imp.path.first() == Some(&"std") {
        let key = imp.path.join("/");
        if std_source(&key).is_some() {
            return Ok(ImportTarget::Module(key));
        }
        // not a file - a directory of them?
        let rel = key.strip_prefix("std/").unwrap_or("");
        if let Some(members) = std_dir_members(rel, arena) {
            if members.is_empty() {
                return Err(format!("std module directory '{}' contains no modules", key));
            }
            return Ok(ImportTarget::Dir(members));
        }
        Err(format!("unknown std module '{}'", key))
    } else {
        let dir = dir.ok_or_else(||
            "cannot resolve a relative import from this module (std/prelude modules may only import `std/...`)".to_string())?;
        let mut base = dir.to_path_buf();
        for seg in &imp.path { base.push(seg); }
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

/// Load one module's source + the dir its own relative imports resolve against.
/// assumes `resolve_target` already succeeded for this key.
fn load_import<'a>(imp: &Import, key: &str, dir: Option<&FilePath>, arena: &'a Bump) -> Result<(&'a str, Option<PathBuf>), String> {
    if imp.path.first() == Some(&"std") {
        Ok((std_source(key).expect("std source vanished after resolve_target"), None))
    } else {
        let _ = dir; // key is already the canonical absolute path
        let canon = PathBuf::from(key);
        let src = std::fs::read_to_string(&canon)
            .map_err(|e| format!("cannot read module file '{}': {}", canon.display(), e))?;
        Ok((arena.alloc_str(&src), canon.parent().map(|d| d.to_path_buf())))
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
/// A raw (pre-mangling) method record, collected while desugaring an `extend`
/// block. `target`/`name` are as written; `fn_name` is the function
/// `lower_methods` synthesized for it. `load_and_merge` maps all three to final
/// names once scopes exist, and records the result in `Defs`'s member table -
/// which is what makes a method reachable without reconstructing its name from
/// its type's.
struct RawMethod<'a> {
    target: &'a str,
    name: &'a str,
    fn_name: &'a str,
    receiver: Receiver,
}

/// A raw (pre-mangling) `extend Target: Trait` conformance record, collected
/// while desugaring. `target`/`trait_` are the source names; `load_and_merge`
/// remaps them to their final (mangled) forms through the module's scopes before
/// handing them to the typechecker.
struct RawImpl<'a> {
    target: &'a str,
    trait_: &'a str,
    span: Span,
}

fn lower_methods<'a>(items: &mut Vec<TopLevel<'a>>, arena: &'a Bump)
    -> (Vec<Error>, Vec<RawImpl<'a>>, Vec<RawMethod<'a>>)
{
    let mut errors = Vec::new();
    let mut impls: Vec<RawImpl<'a>> = Vec::new();
    let mut methods_out: Vec<RawMethod<'a>> = Vec::new();

    // Types declared generic in this module. A method's `self` type would need the
    // type's own params in scope (`*Vec<T>`), which Stage 1 doesn't handle - reject
    // with a clear message rather than emit a function with an unbound `Param`.
    let mut generic_types: HashSet<&str> = HashSet::new();
    for tl in items.iter() {
        match &tl.value {
            TopLevelNode::Struct { name, generics, .. } if !generics.is_empty() => { generic_types.insert(*name); }
            TopLevelNode::Enum { name, generics, .. } if !generics.is_empty() => { generic_types.insert(*name); }
            _ => {}
        }
    }

    let mut synthesized: Vec<TopLevel<'a>> = Vec::new();
    let mut kept: Vec<TopLevel<'a>> = Vec::with_capacity(items.len());
    for tl in items.drain(..) {
        let TopLevelNode::Extend { target, trait_, methods } = &tl.value else {
            kept.push(tl);
            continue;
        };
        let target: &'a str = target;
        // record the conformance obligation; remapped to final names later.
        if let Some(tr) = trait_ {
            impls.push(RawImpl { target, trait_: tr, span: tl.span.clone() });
        }
        if generic_types.contains(target) {
            errors.push(Error::new(tl.span.clone(), format!(
                "methods on generic type '{}' are not supported yet (Stage 1 supports methods on non-generic types only)",
                target)));
            continue;
        }
        for m in methods {
            let mnode = &m.value;
            let mut params: Vec<(&'a str, Type<'a>)> = Vec::with_capacity(mnode.params.len() + 1);
            // an unresolved path, exactly as if the author had written the target
            // type's name - `extend` runs before name resolution, so a `Struct`
            // here would be skipped by the rewriter and keep the source spelling.
            let self_ty = Type::path(Path::single(target));
            match mnode.receiver {
                Receiver::Associated => {}
                Receiver::Value => params.push(("self", self_ty)),
                Receiver::Pointer => params.push(("self", Type::Pointer(Box::new(self_ty)))),
            }
            params.extend(mnode.params.iter().cloned());
            // the synthesized name only has to be unique within the module - it is
            // never reconstructed by a consumer, since the member record below is
            // what makes this method findable.
            let fname: &'a str = arena.alloc_str(&format!("{}${}", target, mnode.name));
            methods_out.push(RawMethod {
                target, name: mnode.name, fn_name: fname, receiver: mnode.receiver,
            });
            synthesized.push(Metadata::new(
                TopLevelNode::Function {
                    name: fname,
                    def: DefId::UNRESOLVED,
                    is_pub: mnode.is_pub,
                    attributes: mnode.attributes.clone(),
                    generics: mnode.generics.clone(),
                    params,
                    return_type: mnode.return_type.clone(),
                    body: mnode.body.clone(),
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
    -> Result<(Vec<Import<'a>>, Vec<TopLevel<'a>>, Vec<RawImpl<'a>>, Vec<RawMethod<'a>>), ()>
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
    let (imports, mut items) = match parsed {
        Some(pi) if parse_errs.is_empty() => pi,
        _ => return Err(()),
    };

    // desugar `extend`/method blocks into functions before anything else looks at
    // the items.
    let (method_errs, impls, methods) = lower_methods(&mut items, arena);
    if !method_errs.is_empty() {
        for e in &method_errs {
            diag::report_error("Method error", e, files);
        }
        return Err(());
    }

    Ok((imports, items, impls, methods))
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
    /// every method in the program, keyed by `(final type name, method name)`.
    /// what makes `Point::new()` resolve for an imported `Point`.
    members: &'x MemberTable<'a>,
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
}

impl<'x, 'a> Rewriter<'x, 'a> {
    fn error(&mut self, span: &Span, msg: String) {
        self.errs.push(Error::new(span.clone(), msg));
    }

    /// Record an error against the innermost span being walked. For names that
    /// have no span of their own (types, and the leaves inside them).
    fn error_here(&mut self, msg: String) {
        let span = self.span.clone();
        self.errs.push(Error::new(span, msg));
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
                let args = std::mem::take(args);
                *ty = match self.type_head(path, gparams) {
                    TypeHead::Def(def) => Type::Named { def, args },
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

    /// Resolve a name in call position to its final emitted name. A bare name
    /// that resolves to nothing is an error, with three deliberate exceptions,
    /// each an explicit branch below rather than a fallthrough: a local shadowing
    /// a top-level callable, a compiler intrinsic (in no module's symbol table),
    /// and a type-qualified `T::sym` that isn't a known method (left for
    /// typecheck's enum-constructor path).
    /// Resolve a written path used as a value or as a callee.
    ///
    /// `Some(name)` means it denotes a single symbol and the caller should replace
    /// the node with `ExprNode::Var(name)`. `None` means it stays an
    /// `ExprNode::Path` - the one shape that survives resolution is an enum
    /// variant, whose head segment has been rewritten to the enum's final name in
    /// place.
    fn value_path(&mut self, r: &mut NameRef<'a>, span: &Span, in_call: bool) -> Option<&'a str> {
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
            self.error(span, if in_call {
                // same wording as the typechecker's own unknown-call diagnostic:
                // this just catches it a stage earlier, before mangling can
                // obscure it.
                format!("unknown function '{}', is it defined and imported into this module?", one)
            } else {
                format!("unknown value '{}'", one)
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
        if let Some(m) = self.members.get(&(ty, sym)) { return Some(m.name); }
        r.def = ty;
        None
    }

    fn expr(&mut self, e: &mut Expr<'a>, gparams: &HashSet<&str>) {
        self.span = e.span.clone();
        match &mut e.value {
            ExprNode::Call { func, type_args, args } => {
                if let ExprNode::Path(path) = &mut func.value {
                    let span = func.span.clone();
                    if let Some(name) = self.value_path(path, &span, true) {
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
                for a in type_args {
                    if let GenericArg::Type(t) = a { self.ty(t, gparams); }
                }
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
            // a name used as a value: rewrite it to the mangled top-level name,
            // unless a param/local shadows it (then it's a local read, leave it),
            // or it names an enum variant (which stays a `Path`). taking a
            // *generic* fn by value has no type args to monomorphize with; that's
            // handled (or rejected) downstream, not here.
            ExprNode::Path(path) => {
                let span = e.span.clone();
                if let Some(name) = self.value_path(path, &span, false) {
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
                self.ty(ty, gparams);
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
            StmtNode::Return(e) => self.expr(e, gparams),
            StmtNode::Continue | StmtNode::Break => {}
        }
    }

    fn toplevel(&mut self, tl: &mut TopLevel<'a>) {
        // baseline span for anything in this item that has none of its own (field
        // types, a trait method's signature); `stmt`/`expr` narrow it as they go.
        self.span = tl.span.clone();
        match &mut tl.value {
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
            TopLevelNode::Trait { name, def, methods, .. } => {
                let empty = HashSet::new();
                let sym = *self.scopes.types.get(*name)
                    .expect("a module's own trait is always in its own scope");
                *name = sym.name;
                *def = sym.def;
                for m in methods.iter_mut() {
                    for (_, ty) in m.params.iter_mut() { self.ty(ty, &empty); }
                    self.ty(&mut m.return_type, &empty);
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
                    linkage_of(m, name, attributes, false), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            TopLevelNode::Extern { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Extern, name, *is_pub,
                    linkage_of(m, name, attributes, true), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            TopLevelNode::Struct { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Struct, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // enums live in the type namespace like structs, and are mangled like
            // them. That is only sound because every `E::V` reference - in call,
            // value, struct-literal *and* pattern position - is rewritten to the
            // enum's final name; leaving any one of those unrewritten is why enum
            // names used to be forced globally unique.
            TopLevelNode::Enum { name, is_pub, attributes, variants, .. } => {
                let id = def(defs, DefKind::Enum, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
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
            // globals live in the callable/value namespace (referenced as vars).
            TopLevelNode::Global { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Global, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // traits live in the type namespace like structs/enums, and are mangled
            // like them. Sound because trait *bounds* (`T: Display`) and the trait
            // side of each conformance record are rewritten too; a trait declared
            // in one module no longer reserves its name program-wide.
            TopLevelNode::Trait { name, is_pub, .. } => {
                let id = def(defs, DefKind::Trait, name, *is_pub,
                    linkage_of(m, name, &[], false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), def: id, is_pub: *is_pub });
            }
            // `extend` blocks were lowered to functions in `lower_methods`.
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before symtab"),
        }
    }
    st
}

/// Load `entry` and everything it transitively imports, then merge the lot into
/// one flat program. `inject_prelude` makes `std/prelude`'s `pub` items visible
/// unqualified in every other module; the module itself is loaded like any other
/// std module either way, so an explicit `import std/prelude` is not a second
/// copy of it.
pub fn load_and_merge<'a>(entry: &FilePath, inject_prelude: bool, arena: &'a Bump)
    -> Result<(Vec<TopLevel<'a>>, Files<'a>, Defs<'a>, Vec<ImplDecl>), ()>
{
    // a module we've decided to load but haven't parsed yet.
    struct Pending<'a> {
        key: String,
        src: &'a str,
        dir: Option<PathBuf>,
        is_entry: bool,
    }

    let mut worklist: VecDeque<Pending<'a>> = VecDeque::new();

    // prelude first (id 0, if enabled) so it reads nicely in dumped output and
    // becomes the implicit whole-module import of every user module. Seeding the
    // worklist under its real `std/...` key is what makes a later explicit
    // `import std/prelude` hit `seen` and reuse this module.
    if inject_prelude {
        let src = std_source(PRELUDE_KEY)
            .expect("embedded std tree has no prelude.hv");
        worklist.push_back(Pending { key: PRELUDE_KEY.into(), src, dir: None, is_entry: false });
    }

    // entry module. canonicalize *first* so its key matches how imports are
    // keyed (imports always canonicalize). otherwise a module that imports the
    // entry back would key it differently, miss in `seen`, and get the entry
    // parsed + merged twice (dup `main`). if canonicalize fails we can't read it
    // anyway, so bail.
    let entry_path = match std::fs::canonicalize(entry) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: cannot read entry file '{}': {}", entry.display(), e);
            return Err(());
        }
    };
    let entry_src = match std::fs::read_to_string(&entry_path) {
        Ok(s) => arena.alloc_str(&s),
        Err(e) => {
            eprintln!("Error: cannot read entry file '{}': {}", entry_path.display(), e);
            return Err(());
        }
    };
    // user module symbol slugs are relative to the entry file's directory, so a
    // build is reproducible across machines (the canonical key is absolute, and
    // would otherwise bake the developer's home directory into every symbol).
    let entry_dir: Option<PathBuf> = entry_path.parent().map(|d| d.to_path_buf());

    worklist.push_back(Pending {
        key: entry_path.to_string_lossy().into_owned(),
        src: entry_src,
        dir: entry_path.parent().map(|d| d.to_path_buf()),
        is_entry: true,
    });

    // every source a diagnostic might point into, indexed by the `FileId` its
    // spans carry. filled in as modules are popped, so a lex/parse error can quote
    // the module that produced it.
    let mut files: Files<'a> = Files::new();
    // definition + module identities. modules register here as they're popped, so
    // a module's symbol slug is fixed before any of its items are named.
    let mut defs: Defs<'a> = Defs::new();
    let mut modules: Vec<Module<'a>> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut had_error = false;

    while let Some(p) = worklist.pop_front() {
        if seen.contains_key(&p.key) { continue; }
        let id = modules.len();
        seen.insert(p.key.clone(), id);

        // register the source before parsing: its spans carry this id, and any
        // lex/parse diagnostic has to be able to quote it.
        let file = files.add(p.key.clone(), p.src);
        let (imports, items, impls, methods) = match parse_module(file, p.src, arena, &files) {
            Ok(pi) => pi,
            Err(()) => { had_error = true; continue; }
        };

        // the module's symbol slug is derived from its *path* (see
        // `defs::module_slug`), so it no longer shifts when an unrelated import
        // is added or removed - which the old `m{id}_{basename}` prefix did, `id`
        // being an enqueue index.
        let mid = defs.add_module(p.key.clone(), file, p.is_entry, entry_dir.as_deref());

        // resolve + enqueue each import. errors point at the import statement in
        // this module.
        let mut import_keys = Vec::with_capacity(imports.len());
        for imp in &imports {
            let target = match resolve_target(imp, p.dir.as_deref(), arena) {
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
                        key: key.to_string(), src, dir, is_entry: false,
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
            is_entry: p.is_entry,
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
    // looked up rather than assumed to be 0: with `inject_prelude` off, a program
    // may still `import std/prelude` by hand, and that must stay an ordinary
    // qualified/selective import instead of silently going implicit everywhere.
    let prelude_id = if inject_prelude { seen.get(PRELUDE_KEY).copied() } else { None };

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
                errs.push(Error::new(imp.span.clone(), format!(
                    "`pub import {}` re-exports nothing: a whole-module import binds \
                     the qualifier '{}' rather than any names. List the symbols to \
                     re-export, e.g. `pub import {} {{ ... }}`",
                    imp.path.join("/"), imp.path.last().unwrap(), imp.path.join("/"))));
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
                    "'{}' is a directory of modules, not a module: import it whole \
                     (`import {}`) and reach its members through the qualifier, \
                     e.g. `{}::<module>::<name>`",
                    imp.path.join("/"), imp.path.join("/"), imp.path.last().unwrap())));
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
                                    "'{}' is imported from more than one module; qualify it with a \
                                     whole-module `import` instead", sym)));
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
                            errs.push(Error::new(imp.span.clone(), if exists {
                                format!("symbol '{}' of module '{}' is private; add `pub` to export it",
                                    sym, imp.path.join("/"))
                            } else {
                                format!("module '{}' has no exported symbol '{}'", imp.path.join("/"), sym)
                            }));
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

    // pass 2: rewrite each module's items in place using its scopes. Also remap
    // each `extend T: Trait` conformance record to final (mangled) names through
    // the same scopes, so the typechecker matches them against the merged program.
    // pass 1.5: record every method under its type's *final* name, before any
    // module is rewritten. has to precede pass 2 because a call `Point::new()` in
    // module A resolves through a member declared in module B.
    for (id, m) in modules.iter().enumerate() {
        let scopes = &all_scopes[id];
        for rm in &m.methods {
            // an `extend` on an unknown type is reported when the block's `self`
            // parameter is resolved; skip the member rather than inventing an
            // identity for a type that doesn't exist.
            let Some(ty) = scopes.types.get(rm.target) else { continue };
            let f = scopes.calls.get(rm.fn_name).map(|s| s.name).unwrap_or(rm.fn_name);
            defs.add_member(ty.def, rm.name, Member { name: f, receiver: rm.receiver });
        }
    }

    let mut impls: Vec<ImplDecl> = Vec::new();
    for (id, m) in modules.iter_mut().enumerate() {
        let scopes = &all_scopes[id];
        let mut rw = Rewriter {
            scopes,
            members: defs.members(),
            errs: &mut errs,
            locals: Vec::new(),
            span: Span::new(m.file, 0, 0),
        };
        for tl in &mut m.items {
            rw.toplevel(tl);
        }
        for imp in &m.impls {
            // as above: an `extend T: Trait` naming an unknown `T` or `Trait`
            // has already produced an error through the type/bound paths, so
            // dropping the record here loses no diagnostic.
            let (Some(target), Some(trait_)) =
                (scopes.types.get(imp.target), scopes.types.get(imp.trait_)) else { continue };
            impls.push(ImplDecl {
                target: target.def,
                trait_: trait_.def,
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
                    TopLevelNode::Struct { .. } | TopLevelNode::Enum { .. } | TopLevelNode::Trait { .. } => unreachable!("only callables are recorded"),
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

    Ok((out, files, defs, impls))
}
