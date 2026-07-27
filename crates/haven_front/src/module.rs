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
//! symbols visible unqualified (it has no qualifier spelling).
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
use std::path::{Path, PathBuf};

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
    /// its canonical key (absolute path, or `std/...`, or `<prelude>`) and source
    /// text are stored there rather than duplicated here.
    file: FileId,
    /// this module's entry in `Defs`, which owns its symbol slug.
    mid: ModId,
    is_entry: bool,
    imports: Vec<Import<'a>>,
    /// canonical key of each import (parallel to `imports`), or `None` if it
    /// failed to resolve (error already recorded).
    import_keys: Vec<Option<String>>,
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
    name: &'a str,
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

/// Resolve an import to its canonical key. no file read for `std`; for the
/// relative case canonicalizes the path (so the file has to exist). `dir` is the
/// importing module's directory.
fn resolve_key(imp: &Import, dir: Option<&Path>) -> Result<String, String> {
    if imp.path.first() == Some(&"std") {
        let key = imp.path.join("/");
        if std_source(&key).is_none() {
            return Err(format!("unknown std module '{}'", key));
        }
        Ok(key)
    } else {
        let dir = dir.ok_or_else(||
            "cannot resolve a relative import from this module (std/prelude modules may only import `std/...`)".to_string())?;
        let mut p = dir.to_path_buf();
        for seg in &imp.path { p.push(seg); }
        p.set_extension("hv");
        let canon = std::fs::canonicalize(&p)
            .map_err(|_| format!("cannot find module file '{}'", p.display()))?;
        Ok(canon.to_string_lossy().into_owned())
    }
}

/// Load an import's source + the dir its own relative imports resolve against.
/// assumes `resolve_key` already succeeded for this import.
fn load_import<'a>(imp: &Import, key: &str, dir: Option<&Path>, arena: &'a Bump) -> Result<(&'a str, Option<PathBuf>), String> {
    if imp.path.first() == Some(&"std") {
        Ok((std_source(key).expect("std source vanished after resolve_key"), None))
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
            let self_ty = Type::Struct { name: target, args: Vec::new() };
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
    /// unqualified callable names in scope -> final emitted name.
    calls: HashMap<&'a str, &'a str>,
    /// unqualified struct type names in scope -> final emitted name.
    types: HashMap<&'a str, &'a str>,
    /// `qualifier -> (symbol -> final name)` for selective imports.
    quals: HashMap<&'a str, HashMap<&'a str, &'a str>>,
}

/// Holds the per-module scopes and error sink while rewriting a module's AST in
/// place.
struct Rewriter<'x, 'a> {
    scopes: &'x Scopes<'a>,
    /// needed to mint rewritten `Enum::Variant` paths, whose enum half changes.
    arena: &'a Bump,
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
    /// returning `true` if `qual` named a type at all.
    ///
    /// Enum names are mangled like struct names, so every reference to a variant
    /// has to be re-spelled - in call position (`E::V(x)`), value position (a unit
    /// variant), struct-literal position (`E::V { .. }`) and match patterns. Miss
    /// any one of them and the enum's name has to stay globally unique, which is
    /// exactly the constraint this lifts.
    fn variant_path(&mut self, name: &mut &'a str, qual: &str, sym: &str) -> bool {
        let Some(&ty) = self.scopes.types.get(qual) else { return false };
        if ty != qual {
            *name = self.arena.alloc_str(&format!("{}::{}", ty, sym));
        }
        true
    }

    /// Rewrite the trait names in an item's generic bounds (`T: Display`). Traits
    /// are mangled now, so a bound has to name the final trait for it to line up
    /// with the conformance records and the trait table.
    fn bounds(&mut self, generics: &mut [GenericParam<'a>]) {
        for g in generics.iter_mut() {
            let GenericParam::Type { bounds, .. } = g else { continue };
            for b in bounds.iter_mut() {
                match self.scopes.types.get(*b) {
                    Some(&f) => *b = f,
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
            PatternNode::Path(path) => {
                if let Some((qual, sym)) = path.split_once("::") {
                    let (qual, sym) = (qual.to_string(), sym.to_string());
                    self.variant_path(path, &qual, &sym);
                }
            }
            PatternNode::Variant { path, fields } => {
                if let Some((qual, sym)) = path.split_once("::") {
                    let (qual, sym) = (qual.to_string(), sym.to_string());
                    self.variant_path(path, &qual, &sym);
                }
                for f in fields { self.pattern(&mut f.value, out); }
            }
            PatternNode::StructVariant { path, fields } => {
                if let Some((qual, sym)) = path.split_once("::") {
                    let (qual, sym) = (qual.to_string(), sym.to_string());
                    self.variant_path(path, &qual, &sym);
                }
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
            Type::Struct { name, args } => {
                let nm: &str = *name;
                // `Self` in a trait method signature stands for the implementing
                // type; typecheck substitutes it per impl, so it never resolves
                // through a module scope.
                if !gparams.contains(nm) && nm != "Self" {
                    if let Some((qual, sym)) = nm.split_once("::") {
                        match self.scopes.quals.get(qual).map(|m| m.get(sym)) {
                            Some(Some(&f)) => *name = f,
                            Some(None) => self.error_here(format!(
                                "type '{}' is not imported from module qualifier '{}'", sym, qual)),
                            None => self.error_here(format!(
                                "unknown module qualifier '{}' (did you `import .../{}`?)", qual, qual)),
                        }
                    } else if let Some(&f) = self.scopes.types.get(nm) {
                        *name = f;
                    } else {
                        self.error_here(format!("unknown type '{}'", nm));
                    }
                }
                // a generic struct's type arguments are themselves types to
                // rewrite; const arguments carry no names to resolve.
                for a in args {
                    if let GenericArg::Type(t) = a { self.ty(t, gparams); }
                }
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

    /// Resolve a name in call position to its final emitted name. A bare name
    /// that resolves to nothing is an error, with three deliberate exceptions,
    /// each an explicit branch below rather than a fallthrough: a local shadowing
    /// a top-level callable, a compiler intrinsic (in no module's symbol table),
    /// and a type-qualified `T::sym` that isn't a known method (left for
    /// typecheck's enum-constructor path).
    fn call_name(&mut self, name: &mut &'a str, span: &Span) {
        let full: &str = *name;
        if let Some((qual, sym)) = full.split_once("::") {
            // `Type::sym` where `Type` is a known type: either an associated-function
            // call `Point::new(...)` or a data-enum constructor `Enum::Variant(...)`.
            // Both are type-qualified, not module-qualified.
            //
            // The member lookup is keyed on the type's *final* name, so this works
            // for an imported type too. The old form reconstructed `Type$sym` and
            // looked it up in the call scope, which could only ever hit for a type
            // declared in this same module: no import form can name `Point$new`
            // (`$` is unlexable), so `import geo { Point }` + `Point::new()` was
            // silently left unresolved.
            if let Some(&ty) = self.scopes.types.get(qual) {
                match self.members.get(&(ty, sym)) {
                    Some(m) => *name = m.name,
                    // otherwise a data-enum constructor `Enum::Variant(...)`: only
                    // the enum half is rewritten, and typecheck's constructor path
                    // handles the rest.
                    None => { self.variant_path(name, qual, sym); }
                }
                return;
            }
        }
        if let Some((qual, sym)) = full.split_once("::") {
            match self.scopes.quals.get(qual) {
                Some(map) => match map.get(sym) {
                    Some(&f) => *name = f,
                    None => self.error(span, format!(
                        "'{}' is not imported from module qualifier '{}'", sym, qual)),
                },
                None => self.error(span, format!(
                    "unknown module qualifier '{}' (did you `import .../{}`?)", qual, qual)),
            }
        } else if self.is_local(full) {
            // shadowed by a param/local: this is an indirect call through a value,
            // not a reference to the top-level symbol. leave it untouched.
        } else if let Some(&f) = self.scopes.calls.get(full) {
            *name = f;
        } else if Intrinsic::lookup(full).is_some() {
            // a compiler intrinsic (`sizeof`, `null`, `__simd_*`): not declared in
            // any module, resolved by the typechecker. leave it untouched.
        } else {
            // same wording as the typechecker's own unknown-call diagnostic: this
            // just catches it a stage earlier, before mangling can obscure it.
            self.error(span, format!(
                "unknown function '{}', is it defined and imported into this module?", full));
        }
    }

    fn expr(&mut self, e: &mut Expr<'a>, gparams: &HashSet<&str>) {
        self.span = e.span.clone();
        match &mut e.value {
            ExprNode::Call { func, type_args, args } => {
                if let ExprNode::Var(name) = &mut func.value {
                    self.call_name(name, &func.span);
                } else {
                    self.expr(func, gparams);
                }
                for ga in type_args {
                    if let GenericArg::Type(t) = ga { self.ty(t, gparams); }
                }
                for a in args { self.expr(a, gparams); }
            }
            ExprNode::Struct { name, type_args, fields } => {
                let nm: &str = *name;
                if let Some((qual, sym)) = nm.split_once("::") {
                    if self.variant_path(name, qual, sym) {
                        // a struct-style enum variant literal, `Msg::Cc { id, val }`:
                        // type-qualified, not module-qualified. the enum half is
                        // rewritten above; typecheck routes the rest to the variant.
                    } else if let Some(&f) = self.scopes.quals.get(qual).and_then(|m| m.get(sym)) {
                        *name = f;
                    } else if self.scopes.quals.contains_key(qual) {
                        self.error_here(format!(
                            "type '{}' is not imported from module qualifier '{}'", sym, qual));
                    } else {
                        self.error_here(format!(
                            "unknown module qualifier '{}' (did you `import .../{}`?)", qual, qual));
                    }
                } else if let Some(&f) = self.scopes.types.get(nm) {
                    *name = f;
                } else {
                    self.error_here(format!("unknown struct '{}'", nm));
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
            // a bare name used as a value (fn-as-value): rewrite it to the mangled
            // top-level name, unless a param/local shadows it (then it's a local
            // read, leave it). taking a *generic* fn by value has no type args to
            // monomorphize with; that's handled (or rejected) downstream, not here.
            ExprNode::Var(name) => {
                let full: &str = *name;
                if self.is_local(full) {
                    // a param, `let`, or match-arm binding shadows any top-level
                    // symbol of the same name.
                } else if let Some(&f) = self.scopes.calls.get(full) {
                    *name = f;
                } else if let Some((qual, sym)) = full.split_once("::") {
                    if self.variant_path(name, qual, sym) {
                        // `Enum::Variant` read as a value (a unit variant), or an
                        // associated fn taken by value: the enum half is rewritten
                        // above, the rest resolved in typecheck.
                    } else if let Some(&f) = self.scopes.quals.get(qual).and_then(|m| m.get(sym)) {
                        // a module-qualified value, `math::square`. previously not
                        // rewritten at all, which failed later as an unknown name.
                        *name = f;
                    } else if self.scopes.quals.contains_key(qual) {
                        self.error_here(format!(
                            "'{}' is not imported from module qualifier '{}'", sym, qual));
                    } else {
                        self.error_here(format!(
                            "unknown module qualifier '{}' (did you `import .../{}`?)", qual, qual));
                    }
                } else {
                    self.error_here(format!("unknown value '{}'", full));
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
            TopLevelNode::Function { name, generics, params, return_type, body, .. } => {
                let gparams = generic_names(generics);
                *name = self.scopes.calls.get(*name).copied()
                    .expect("a module's own callable is always in its own scope");
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
            TopLevelNode::Extern { name, generics, params, return_type, .. } => {
                let gparams = generic_names(generics);
                *name = self.scopes.calls.get(*name).copied()
                    .expect("a module's own callable is always in its own scope");
                for (_, ty) in params { self.ty(ty, &gparams); }
                self.ty(return_type, &gparams);
            }
            TopLevelNode::Struct { name, generics, fields, .. } => {
                // the struct's own type params shadow struct names when rewriting
                // field types (a field `T` is a param, not a module type).
                let gparams = generic_names(generics);
                *name = self.scopes.types.get(*name).copied()
                    .expect("a module's own struct is always in its own scope");
                self.bounds(generics);
                for (_, ty) in fields { self.ty(ty, &gparams); }
            }
            TopLevelNode::Global { name, ty, value, .. } => {
                let empty = HashSet::new();
                *name = self.scopes.calls.get(*name).copied()
                    .expect("a module's own callable is always in its own scope");
                self.ty(ty, &empty);
                self.expr(value, &empty);
            }
            // the enum's type name is now mangled like a struct's, so it is
            // rewritten here and every `E::V` reference is re-spelled to match (see
            // `variant_path`). A data-carrying variant's payload field types are
            // types like any other. The enum's own type params shadow module type
            // names (a payload `T` is a param, not a module type).
            TopLevelNode::Enum { name, generics, variants, .. } => {
                let gparams = generic_names(generics);
                *name = self.scopes.types.get(*name).copied()
                    .expect("a module's own enum is always in its own scope");
                self.bounds(generics);
                for (_, _, payload) in variants.iter_mut() {
                    for (_, ty) in payload.iter_mut() { self.ty(ty, &gparams); }
                }
            }
            // a trait's method signatures carry types (params + return) that must
            // be rewritten like any other - a `String` in `proc display(*self)
            // String` resolves to the imported struct's mangled name. `Self` is
            // left untouched (typecheck substitutes it per implementing type).
            TopLevelNode::Trait { name, methods, .. } => {
                let empty = HashSet::new();
                *name = self.scopes.types.get(*name).copied()
                    .expect("a module's own trait is always in its own scope");
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
fn build_symtab<'a>(m: &Module<'a>, defs: &mut Defs<'a>, arena: &'a Bump) -> SymTab<'a> {
    let mut st = SymTab::default();
    // register the def, then ask `Defs` what it is emitted as.
    let def = |defs: &mut Defs<'a>, kind, name: &'a str, is_pub: bool,
                   linkage, span: &Span| -> DefId {
        defs.alloc(Def {
            module: m.mid, kind, source_name: name, is_pub, linkage, span: *span,
        })
    };

    for tl in &m.items {
        match &tl.value {
            TopLevelNode::Function { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Fn, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), is_pub: *is_pub });
            }
            TopLevelNode::Extern { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Extern, name, *is_pub,
                    linkage_of(m, name, attributes, true), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), is_pub: *is_pub });
            }
            TopLevelNode::Struct { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Struct, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), is_pub: *is_pub });
            }
            // enums live in the type namespace like structs, and are mangled like
            // them. That is only sound because every `E::V` reference - in call,
            // value, struct-literal *and* pattern position - is rewritten to the
            // enum's final name; leaving any one of those unrewritten is why enum
            // names used to be forced globally unique.
            TopLevelNode::Enum { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Enum, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), is_pub: *is_pub });
            }
            // globals live in the callable/value namespace (referenced as vars).
            TopLevelNode::Global { name, is_pub, attributes, .. } => {
                let id = def(defs, DefKind::Global, name, *is_pub,
                    linkage_of(m, name, attributes, false), &tl.span);
                st.fns.insert(name, Sym { name: defs.symbol(id, arena), is_pub: *is_pub });
            }
            // traits live in the type namespace like structs/enums, and are mangled
            // like them. Sound because trait *bounds* (`T: Display`) and the trait
            // side of each conformance record are rewritten too; a trait declared
            // in one module no longer reserves its name program-wide.
            TopLevelNode::Trait { name, is_pub, .. } => {
                let id = def(defs, DefKind::Trait, name, *is_pub,
                    linkage_of(m, name, &[], false), &tl.span);
                st.structs.insert(name, Sym { name: defs.symbol(id, arena), is_pub: *is_pub });
            }
            // `extend` blocks were lowered to functions in `lower_methods`.
            TopLevelNode::Extend { .. } => unreachable!("extend desugared before symtab"),
        }
    }
    st
}

pub fn load_and_merge<'a>(entry: &Path, prelude_src: Option<&'a str>, arena: &'a Bump)
    -> Result<(Vec<TopLevel<'a>>, Files<'a>, Defs<'a>, Vec<ImplDecl<'a>>), ()>
{
    // a module we've decided to load but haven't parsed yet.
    struct Pending<'a> {
        key: String,
        src: &'a str,
        dir: Option<PathBuf>,
        is_entry: bool,
    }

    let mut worklist: VecDeque<Pending<'a>> = VecDeque::new();

    // prelude first (id 0, if present) so it reads nicely in dumped output and
    // becomes the implicit whole-module import of every user module.
    let has_prelude = prelude_src.is_some();
    if let Some(psrc) = prelude_src {
        worklist.push_back(Pending { key: "<prelude>".into(), src: psrc, dir: None, is_entry: false });
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
            match resolve_key(imp, p.dir.as_deref()) {
                Ok(key) => {
                    if !seen.contains_key(&key) {
                        match load_import(imp, &key, p.dir.as_deref(), arena) {
                            Ok((src, dir)) => worklist.push_back(Pending {
                                key: key.clone(), src, dir, is_entry: false,
                            }),
                            Err(msg) => {
                                diag::report("Import error", &msg, &imp.span, &files);
                                had_error = true;
                                import_keys.push(None);
                                continue;
                            }
                        }
                    }
                    import_keys.push(Some(key));
                }
                Err(msg) => {
                    diag::report("Import error", &msg, &imp.span, &files);
                    had_error = true;
                    import_keys.push(None);
                }
            }
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
    let symtabs: Vec<SymTab> = modules.iter()
        .map(|m| build_symtab(m, &mut defs, arena))
        .collect();
    let prelude_id = if has_prelude { Some(0usize) } else { None };

    let mut errs: Vec<Error> = Vec::new();

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
                for (&k, v) in &symtabs[pid].fns { if v.is_pub { scopes.calls.insert(k, v.name); } }
                for (&k, v) in &symtabs[pid].structs { if v.is_pub { scopes.types.insert(k, v.name); } }
            }
        }

        // 2. explicit imports
        let mut from_import_calls: HashSet<&str> = HashSet::new();
        let mut from_import_types: HashSet<&str> = HashSet::new();
        let mut qual_owner: HashMap<&str, usize> = HashMap::new();

        for (imp, key) in m.imports.iter().zip(&m.import_keys) {
            let Some(key) = key else { continue };
            let target_id = seen[key];
            let target = &symtabs[target_id];

            match &imp.symbols {
                None => {
                    // whole module: every symbol visible only as `qualifier::sym`,
                    // qualified under the module's last path segment.
                    let qualifier = *imp.path.last().unwrap();
                    if let Some(&prev) = qual_owner.get(qualifier) {
                        if prev != target_id {
                            errs.push(Error::new(imp.span.clone(), format!(
                                "qualifier '{}' already refers to a different module", qualifier)));
                        }
                    }
                    qual_owner.insert(qualifier, target_id);
                    let map = scopes.quals.entry(qualifier).or_default();
                    // only `pub` items are importable; private ones are invisible
                    // outside their own module.
                    for (&k, v) in &target.fns { if v.is_pub { map.insert(k, v.name); } }
                    for (&k, v) in &target.structs { if v.is_pub { map.insert(k, v.name); } }
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
                            if from_import_calls.contains(sym) && scopes.calls.get(sym).copied() != Some(f.name) {
                                errs.push(Error::new(imp.span.clone(), format!(
                                    "'{}' is imported from more than one module; qualify it with a \
                                     whole-module `import` instead", sym)));
                            }
                            scopes.calls.insert(sym, f.name);
                            from_import_calls.insert(sym);
                        }
                        if let Some(f) = target.structs.get(sym).filter(|f| f.is_pub) {
                            imported = true;
                            if from_import_types.contains(sym) && scopes.types.get(sym).copied() != Some(f.name) {
                                errs.push(Error::new(imp.span.clone(), format!(
                                    "struct '{}' is imported from more than one module", sym)));
                            }
                            scopes.types.insert(sym, f.name);
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
        for (&k, v) in &symtabs[id].fns { scopes.calls.insert(k, v.name); }
        for (&k, v) in &symtabs[id].structs { scopes.types.insert(k, v.name); }

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
            let ty = scopes.types.get(rm.target).copied().unwrap_or(rm.target);
            let f = scopes.calls.get(rm.fn_name).copied().unwrap_or(rm.fn_name);
            defs.add_member(ty, rm.name, Member { name: f, receiver: rm.receiver });
        }
    }

    let mut impls: Vec<ImplDecl<'a>> = Vec::new();
    for (id, m) in modules.iter_mut().enumerate() {
        let scopes = &all_scopes[id];
        let mut rw = Rewriter {
            scopes,
            arena,
            members: defs.members(),
            errs: &mut errs,
            locals: Vec::new(),
            span: Span::new(m.file, 0, 0),
        };
        for tl in &mut m.items {
            rw.toplevel(tl);
        }
        for imp in &m.impls {
            impls.push(ImplDecl {
                target: scopes.types.get(imp.target).copied().unwrap_or(imp.target),
                trait_: scopes.types.get(imp.trait_).copied().unwrap_or(imp.trait_),
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
