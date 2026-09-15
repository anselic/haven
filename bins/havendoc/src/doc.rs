//! `havendoc`: turn `.hv` source into renderer-independent documentation, then
//! emit Markdown or a static HTML site.
//!
//! Signatures come from the parsed AST. Since the lexer discards comments, doc
//! bodies are read from the source and matched to items by span.
//!
//! Markdown output layout, given `havendoc std -o docs`:
//! ```text
//! docs/
//!   index.md
//!   std/alloc.md
//!   std/dsp/osc.md
//!   ...
//! ```
//! `--format html` emits the same module structure as `.html` pages plus a
//! shared `assets/style.css`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bumpalo::Bump;
use pulldown_cmark::{Event, Options, Parser as MarkdownParser, Tag, html};

use crate::{DocArgs, OutputFormat};
use haven_common::ast::{Method, Receiver, TopLevel, TopLevelNode, Type};
use haven_common::diag::Files;
use haven_front::parse;

/// Entry point for the `doc` subcommand. Returns `Err(())` if nothing could be
/// documented (bad path, or every file failed to parse); individual per-file
/// failures are reported and skipped.
pub fn generate(args: &DocArgs) -> Result<(), ()> {
    let files = collect_hv_files(&args.inputs);
    if files.is_empty() {
        eprintln!("havendoc: no .hv files found in {:?}", args.inputs);
        return Err(());
    }

    let package = files
        .first()
        .and_then(|source| source.package.as_deref())
        .filter(|name| {
            files
                .iter()
                .all(|source| source.package.as_deref() == Some(*name))
        })
        .map(str::to_owned);
    let mut modules = Vec::new();
    for source in &files {
        match extract_module(source) {
            Ok(module) => modules.push(module),
            Err(()) => {
                eprintln!(
                    "havendoc: skipping {} (failed to parse)",
                    source.file.display()
                );
            }
        }
    }

    if modules.is_empty() {
        eprintln!("havendoc: no pages generated");
        return Err(());
    }

    modules.sort_by(|a, b| a.title.cmp(&b.title));
    let docs = Documentation { package, modules };
    match args.format {
        OutputFormat::Markdown => render_markdown(&docs, &args.out)?,
        OutputFormat::Html => render_html(&docs, &args.out)?,
    }

    println!(
        "havendoc: wrote {} page(s) to {}",
        docs.modules.len(),
        args.out.display()
    );
    Ok(())
}

/// Renderer-independent documentation extracted from one invocation. This is
/// the boundary future HTML or JSON renderers consume.
#[derive(Debug, PartialEq, Eq)]
struct Documentation {
    /// Present when every input module belongs to the same Vestry package.
    package: Option<String>,
    modules: Vec<ModuleDoc>,
}

#[derive(Debug, PartialEq, Eq)]
struct ModuleDoc {
    title: String,
    docs: Option<String>,
    items: Vec<ItemDoc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ItemKind {
    Function,
    Extern,
    Struct,
    Constant,
    Enum,
    Trait,
    Alias,
    Extension,
}

impl ItemKind {
    fn label(self) -> &'static str {
        match self {
            ItemKind::Function => "function",
            ItemKind::Extern => "extern",
            ItemKind::Struct => "struct",
            ItemKind::Constant => "constant",
            ItemKind::Enum => "enum",
            ItemKind::Trait => "trait",
            ItemKind::Alias => "alias",
            ItemKind::Extension => "extension",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ItemDoc {
    kind: ItemKind,
    name: String,
    /// Haven declaration without its implementation body. Extension groups do
    /// not have a single declaration, so their signature is absent.
    signature: Option<String>,
    docs: Option<String>,
    methods: Vec<MethodDoc>,
}

#[derive(Debug, PartialEq, Eq)]
struct MethodDoc {
    name: String,
    signature: String,
    docs: Option<String>,
}

struct SourceFile {
    title: String,
    file: PathBuf,
    package: Option<String>,
}

/// Recursively gather `.hv` files from each input and assign their module
/// titles. A directory containing `vestry.toml` is a package: its manifest name
/// becomes the title's first component and paths are relative to its `src/`
/// directory. Other directory inputs keep their directory name, while a single
/// file uses just its stem.
fn collect_hv_files(inputs: &[PathBuf]) -> Vec<SourceFile> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_dir() {
            match package_source(input) {
                Ok(Some((name, src))) => walk_dir(&src, &src, Some(&name), &mut out),
                Ok(None) => {
                    let base = input.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
                    walk_dir(&base, input, None, &mut out);
                }
                Err(()) => {}
            }
        } else if is_hv(input) {
            let base = input.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
            push_source(&base, None, input, &mut out);
        } else {
            eprintln!("havendoc: skipping {} (not a .hv file)", input.display());
        }
    }
    out
}

/// If `dir` is a Vestry package, return its name and source directory. A present
/// but invalid manifest is diagnosed and treated as an invalid input rather
/// than silently falling back to ordinary-directory behavior.
fn package_source(dir: &Path) -> Result<Option<(String, PathBuf)>, ()> {
    let manifest_path = dir.join("vestry.toml");
    if !manifest_path.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&manifest_path).map_err(|e| {
        eprintln!(
            "havendoc: cannot read {}: {}",
            manifest_path.display(),
            e
        );
    })?;
    let manifest: toml::Value = toml::from_str(&text).map_err(|e| {
        eprintln!("havendoc: invalid {}: {}", manifest_path.display(), e);
    })?;
    let name = manifest
        .get("project")
        .and_then(|project| project.get("name"))
        .and_then(toml::Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            eprintln!(
                "havendoc: {} has no non-empty `project.name`",
                manifest_path.display()
            );
        })?;
    if name == "." || name == ".." || name.contains(['/', '\\']) {
        eprintln!(
            "havendoc: package name {:?} in {} cannot be used as a documentation path",
            name,
            manifest_path.display()
        );
        return Err(());
    }

    let src = dir.join("src");
    if !src.is_dir() {
        eprintln!(
            "havendoc: package {} has no source directory at {}",
            name,
            src.display()
        );
        return Err(());
    }
    Ok(Some((name.to_string(), src)))
}

fn walk_dir(base: &Path, dir: &Path, prefix: Option<&str>, out: &mut Vec<SourceFile>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("havendoc: cannot read {}: {}", dir.display(), e);
            return;
        }
    };
    // Collect + sort so output is deterministic across filesystems.
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk_dir(base, &path, prefix, out);
        } else if is_hv(&path) {
            push_source(base, prefix, &path, out);
        }
    }
}

fn push_source(base: &Path, prefix: Option<&str>, file: &Path, out: &mut Vec<SourceFile>) {
    let relative = module_title(base, file);
    let title = match prefix {
        Some(prefix) => format!("{}/{}", prefix, relative),
        None => relative,
    };
    out.push(SourceFile {
        title,
        file: file.to_path_buf(),
        package: prefix.map(str::to_owned),
    });
}

fn is_hv(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()) == Some("hv")
}

/// Strip `strip_base` off `file` and drop the extension to form a module title:
/// `std/dsp/osc.hv` with base `` -> `std/dsp/osc`; `foo/bar.hv` with base `foo`
/// -> `bar`. Falls back to the full path when `file` isn't under `strip_base`.
fn module_title(strip_base: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(strip_base).unwrap_or(file);
    let rel = rel.with_extension("");
    rel.to_string_lossy().replace('\\', "/")
}

fn title_to_rel_path(title: &str) -> PathBuf {
    PathBuf::from(format!("{}.md", title))
}

// ---------------------------------------------------------------------------
// Documentation extraction
// ---------------------------------------------------------------------------

fn extract_module(source: &SourceFile) -> Result<ModuleDoc, ()> {
    let src = std::fs::read_to_string(&source.file).map_err(|e| {
        eprintln!("havendoc: cannot read {}: {}", source.file.display(), e);
    })?;

    // Parse into real signatures. A fresh arena per file is fine: everything we
    // keep (the rendered strings) is owned by the time it drops.
    let arena = Bump::new();
    let src_ref: &str = arena.alloc_str(&src);

    // havendoc renders one file at a time and never reports a span-carrying
    // diagnostic (a lex/parse failure just fails the page), so a one-entry file
    // table is all the spans need to be well-formed.
    let mut files = Files::new();
    let key = files.add(source.file.to_string_lossy().into_owned(), src_ref);

    let (tokens, lex_errs) = parse::lex(key, src_ref);
    let tokens = match tokens {
        Some(t) if lex_errs.is_empty() => t,
        _ => return Err(()),
    };
    let tokens = arena.alloc_slice_fill_iter(tokens);
    let (parsed, parse_errs) = parse::parse(key, src_ref.len(), tokens);
    let (_mod_attrs, _imports, items) = match parsed {
        Some(pi) if parse_errs.is_empty() => pi,
        _ => return Err(()),
    };

    let lines = LineIndex::new(&src);

    // `extend`/inherent-method blocks were parsed into `Extend` items. Group each
    // block's *visible* methods (public, or any method of a trait impl) under the
    // type name they extend, so they render as a section of that type rather than
    // as standalone blocks. A target with no declared type in this file (a
    // builtin, or an imported type) keeps its own group, rendered after the
    // declared items.
    let mut method_groups: BTreeMap<String, Vec<&Method>> = BTreeMap::new();
    for item in &items {
        if let TopLevelNode::Extend { methods, trait_, target, .. } = &item.value {
            let key = extend_target_key(target);
            let group = method_groups.entry(key).or_default();
            for m in methods {
                if method_visible(&m.value, trait_.is_some()) {
                    group.push(m);
                }
            }
        }
    }

    let mut documented_items = Vec::new();
    for item in &items {
        match &item.value {
            // extend blocks are rendered as method sections under their type, not
            // as items in their own right; handled above/below.
            TopLevelNode::Extend { .. } => continue,
            // docs describe a module's public surface; private (non-`pub`) items
            // are implementation details and are omitted entirely.
            node if !item_is_pub(node) => continue,
            node => {
                // a type carries its methods directly beneath its declaration.
                let methods = if let TopLevelNode::Struct { name, .. }
                    | TopLevelNode::Enum { name, .. } = node
                {
                    method_groups.remove(*name).unwrap_or_default()
                } else {
                    Vec::new()
                };
                documented_items.push(extract_item(item, &src, &lines, &methods));
            }
        }
    }

    // methods extending a type not declared here (`extend i32`, or an imported
    // type). Only emitted when a group actually has visible methods.
    for (target, methods) in &method_groups {
        if methods.is_empty() {
            continue;
        }
        documented_items.push(ItemDoc {
            kind: ItemKind::Extension,
            name: target.clone(),
            signature: None,
            docs: None,
            methods: methods
                .iter()
                .map(|method| extract_method(method, &lines))
                .collect(),
        });
    }

    Ok(ModuleDoc {
        title: source.title.clone(),
        docs: lines.module_doc(),
        items: documented_items,
    })
}

fn extract_item(
    item: &TopLevel,
    src: &str,
    lines: &LineIndex,
    methods: &[&Method],
) -> ItemDoc {
    ItemDoc {
        kind: item_kind(&item.value),
        name: item_name(&item.value).to_string(),
        signature: Some(signature(
            &item.value,
            src,
            item.span.start,
            item.span.end,
        )),
        docs: lines.doc_above(item.span.start),
        methods: methods
            .iter()
            .map(|method| extract_method(method, lines))
            .collect(),
    }
}

fn extract_method(method: &Method, lines: &LineIndex) -> MethodDoc {
    MethodDoc {
        name: method.value.name.to_string(),
        signature: method_signature(&method.value),
        docs: lines.doc_above(method.span.start),
    }
}

fn item_kind(node: &TopLevelNode) -> ItemKind {
    match node {
        TopLevelNode::Function { .. } => ItemKind::Function,
        TopLevelNode::Extern { .. } => ItemKind::Extern,
        TopLevelNode::Struct { .. } => ItemKind::Struct,
        TopLevelNode::Global { .. } => ItemKind::Constant,
        TopLevelNode::Enum { .. } => ItemKind::Enum,
        TopLevelNode::Trait { .. } => ItemKind::Trait,
        TopLevelNode::Alias { .. } => ItemKind::Alias,
        TopLevelNode::Extend { .. } => ItemKind::Extension,
    }
}

/// The group key for an `extend` target: the name a reader looks it up under. A
/// named type keys on its bare name (`Vec<T>` -> `Vec`) so its methods attach to
/// the struct/enum declared under that name; everything else keys on its written
/// form (`i32`, `f32`, `str`, `*T`, `[T]`), so each primitive and structural
/// target gets its own section instead of collapsing together.
fn extend_target_key(target: &Type) -> String {
    match target {
        Type::Path { path, .. } => path.last().to_string(),
        other => other.to_string(),
    }
}

/// Whether a method is part of the module's public surface: declared `pub`, or a
/// method of a trait impl (whose reachability follows the trait, so it carries no
/// explicit `pub` yet is public). Mirrors the effective visibility the compiler
/// records in `Member::is_pub`.
fn method_visible(m: &haven_common::ast::MethodNode, in_trait_impl: bool) -> bool {
    m.is_pub || in_trait_impl
}

/// Render a method's signature without its body: `[attrs] pub proc name<gens>(recv,
/// params) Ret`. The receiver renders as `self`/`*self`; an associated function
/// has none.
fn method_signature(m: &haven_common::ast::MethodNode) -> String {
    let mut s = attr_prefix(&m.attributes);
    if m.is_pub {
        s.push_str("pub ");
    }
    let recv = match m.receiver {
        Receiver::Associated => String::new(),
        Receiver::Value => "self".to_string(),
        Receiver::Pointer => "*self".to_string(),
    };
    let sep = if !recv.is_empty() && !m.params.is_empty() { ", " } else { "" };
    s.push_str(&format!(
        "proc {}{}({}{}{}) {}",
        m.name,
        fmt_generics(&m.generics),
        recv,
        sep,
        fmt_params(&m.params),
        m.return_type
    ));
    s
}

/// Whether a top-level item is `pub` (part of the module's public API).
fn item_is_pub(node: &TopLevelNode) -> bool {
    match node {
        TopLevelNode::Function { is_pub, .. }
        | TopLevelNode::Extern { is_pub, .. }
        | TopLevelNode::Struct { is_pub, .. }
        | TopLevelNode::Global { is_pub, .. }
        | TopLevelNode::Enum { is_pub, .. }
        | TopLevelNode::Trait { is_pub, .. }
        | TopLevelNode::Alias { is_pub, .. } => *is_pub,
        // `extend` blocks are rendered as method sections under their target type
        // (see `render_file`), not as items here.
        TopLevelNode::Extend { .. } => false,
    }
}

fn item_name<'a>(node: &TopLevelNode<'a>) -> &'a str {
    match node {
        TopLevelNode::Function { name, .. }
        | TopLevelNode::Extern { name, .. }
        | TopLevelNode::Struct { name, .. }
        | TopLevelNode::Global { name, .. }
        | TopLevelNode::Enum { name, .. }
        | TopLevelNode::Trait { name, .. }
        | TopLevelNode::Alias { name, .. } => name,
        // an `extend` target is a type, not a name: `[T]` and `*Point` have no
        // identifier to report. Group those under their constructor, and a named
        // target under the name it extends, which is what a reader looks for.
        TopLevelNode::Extend { target, .. } => match target {
            Type::Path { path, .. } => path.last(),
            Type::Slice(_) => "[]",
            Type::Array(..) => "[;]",
            Type::Pointer(_) => "*",
            Type::Simd(..) => "simd",
            Type::Function { .. } => "proc",
            _ => "builtin",
        },
    }
}

/// Render a declaration's signature without its body. Types, generics and
/// attributes all reuse the AST `Display` impls so the output tracks the real
/// grammar. `start`/`end` bound the item in `src`, used to recover per-field
/// trailing doc comments on structs.
fn signature(node: &TopLevelNode, src: &str, start: usize, end: usize) -> String {
    match node {
        TopLevelNode::Function { name, attributes, generics, params, return_type, .. } => {
            let mut s = attr_prefix(attributes);
            s.push_str(&format!(
                "proc {}{}({}) {}",
                name,
                fmt_generics(generics),
                fmt_params(params),
                return_type
            ));
            s
        }
        TopLevelNode::Extern { name, attributes, generics, params, return_type, .. } => {
            let mut s = attr_prefix(attributes);
            s.push_str(&format!(
                "extern {}{}({}) {};",
                name,
                fmt_generics(generics),
                fmt_params(params),
                return_type
            ));
            s
        }
        TopLevelNode::Struct { name, attributes, generics, fields, .. } => {
            let mut s = attr_prefix(attributes);
            s.push_str(&format!("struct {}{} {{\n", name, fmt_generics(generics)));
            let field_docs = struct_field_docs(src, start, end);
            for (fname, fty) in fields {
                let doc = field_docs
                    .iter()
                    .find(|(n, _)| n == fname)
                    .map(|(_, d)| format!("  /// {}", d))
                    .unwrap_or_default();
                s.push_str(&format!("    {}: {},{}\n", fname, fty, doc));
            }
            s.push('}');
            s
        }
        TopLevelNode::Global { name, attributes, ty, value, .. } => {
            let mut s = attr_prefix(attributes);
            s.push_str(&format!("const {}: {} = {};", name, ty, value.value));
            s
        }
        TopLevelNode::Enum { name, attributes, variants, .. } => {
            let mut s = attr_prefix(attributes);
            s.push_str(&format!("enum {} {{\n", name));
            for (vname, val, payload) in variants {
                // tuple variants carry synthesized names "0", "1", ...; render them
                // positionally as `(T, U)`. Struct-style variants keep real field
                // names and render as `{ id: T, val: U }`.
                let is_tuple = payload.first()
                    .is_some_and(|(n, _)| n.bytes().all(|b| b.is_ascii_digit()));
                let payload_str = if payload.is_empty() {
                    String::new()
                } else if is_tuple {
                    format!("({})", payload.iter().map(|(_, ty)| ty.to_string()).collect::<Vec<_>>().join(", "))
                } else {
                    format!(" {{ {} }}", payload.iter().map(|(n, ty)| format!("{}: {}", n, ty)).collect::<Vec<_>>().join(", "))
                };
                match val {
                    Some(v) => s.push_str(&format!("    {}{} = {},\n", vname, payload_str, v)),
                    None => s.push_str(&format!("    {}{},\n", vname, payload_str)),
                }
            }
            s.push('}');
            s
        }
        // a trait renders its method-signature block via the Display impl.
        TopLevelNode::Trait { .. } => node.to_string(),
        // never rendered (extends are filtered out by `item_is_pub`), but the
        // match must stay exhaustive; fall back to the Display impl.
        TopLevelNode::Extend { .. } | TopLevelNode::Alias { .. } => node.to_string(),
    }
}

fn attr_prefix(attributes: &[haven_common::ast::Attribute]) -> String {
    if attributes.is_empty() {
        String::new()
    } else {
        attributes
            .iter()
            .map(|a| a.value.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }
}

fn fmt_generics(generics: &[haven_common::ast::GenericParam]) -> String {
    if generics.is_empty() {
        String::new()
    } else {
        format!(
            "<{}>",
            generics.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", ")
        )
    }
}

fn fmt_params(params: &[(&str, Type)]) -> String {
    params
        .iter()
        .map(|(n, t)| format!("{}: {}", n, t))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Recover `field: ... /// doc` trailing comments from a struct's source text.
/// The AST drops these (they're padding), and struct fields carry no per-field
/// span, so we scan the item's byte range and match by leading field name.
fn struct_field_docs(src: &str, start: usize, end: usize) -> Vec<(String, String)> {
    let slice = &src[start.min(src.len())..end.min(src.len())];
    let mut out = Vec::new();
    for line in slice.lines() {
        let Some((code, comment)) = line.split_once("///") else {
            continue;
        };
        // `name: type,` -> field name is the last ident before the first `:`
        // (last, so a `struct Foo {` prefix on the same line doesn't fool us).
        let Some((before_colon, _)) = code.split_once(':') else {
            continue;
        };
        let Some(fname) = before_colon.rsplit([' ', '\t', '{', ',', '(']).find(|s| !s.is_empty())
        else {
            continue;
        };
        if !is_ident(fname) {
            continue;
        }
        out.push((fname.to_string(), comment.trim().to_string()));
    }
    out
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ---------------------------------------------------------------------------
// Line/byte bookkeeping for doc-comment recovery
// ---------------------------------------------------------------------------

/// Maps byte offsets to lines so doc comments (stripped before the AST) can be
/// read back out of the raw source.
struct LineIndex {
    /// (byte offset of line start, line text without the trailing newline).
    lines: Vec<(usize, String)>,
}

impl LineIndex {
    fn new(src: &str) -> Self {
        let mut lines = Vec::new();
        let mut offset = 0;
        for line in src.split_inclusive('\n') {
            let text = line.strip_suffix('\n').unwrap_or(line);
            let text = text.strip_suffix('\r').unwrap_or(text);
            lines.push((offset, text.to_string()));
            offset += line.len();
        }
        LineIndex { lines }
    }

    /// Index of the line containing `byte`.
    fn line_of(&self, byte: usize) -> usize {
        match self.lines.binary_search_by(|(start, _)| start.cmp(&byte)) {
            Ok(i) => i,
            // `Err(i)` is the first line starting after `byte`; the owner is the
            // one before it.
            Err(i) => i.saturating_sub(1),
        }
    }

    /// Contiguous `///` block directly above the item at `byte` (no blank gap),
    /// as rendered Markdown, or `None` if there isn't one.
    fn doc_above(&self, byte: usize) -> Option<String> {
        let item_line = self.line_of(byte);
        let mut collected: Vec<&str> = Vec::new();
        let mut i = item_line;
        while i > 0 {
            i -= 1;
            let text = self.lines[i].1.trim_start();
            if let Some(rest) = doc_body(text) {
                collected.push(rest);
            } else {
                break;
            }
        }
        if collected.is_empty() {
            return None;
        }
        collected.reverse();
        Some(collected.join("\n"))
    }

    /// A leading `///` block at the top of the file, but only when it's set off
    /// from the first item by a blank line (so it reads as module-level prose,
    /// not the first item's doc).
    fn module_doc(&self) -> Option<String> {
        let mut i = 0;
        let mut collected: Vec<&str> = Vec::new();
        while i < self.lines.len() {
            let text = self.lines[i].1.trim_start();
            match doc_body(text) {
                Some(rest) => {
                    collected.push(rest);
                    i += 1;
                }
                None => break,
            }
        }
        if collected.is_empty() {
            return None;
        }
        // Require a blank separator (or EOF) after the block; otherwise it's
        // glued to the first item and `doc_above` will claim it there.
        let separated = i >= self.lines.len() || self.lines[i].1.trim().is_empty();
        if separated {
            Some(collected.join("\n"))
        } else {
            None
        }
    }
}

/// If `line` (already left-trimmed) is a `///` doc comment, return its body with
/// the marker and one optional leading space stripped. `//` (non-doc) and code
/// return `None`.
fn doc_body(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("///")?;
    // `////`-style dividers aren't doc text; treat as ordinary comments.
    if rest.starts_with('/') {
        return None;
    }
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

fn render_markdown(docs: &Documentation, out_dir: &Path) -> Result<(), ()> {
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("havendoc: cannot create {}: {}", out_dir.display(), e);
        return Err(());
    }

    let mut pages = Vec::with_capacity(docs.modules.len());
    for module in &docs.modules {
        let relative = title_to_rel_path(&module.title);
        let path = out_dir.join(&relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                return Err(());
            }
        write_file(&path, render_module_markdown(module).as_bytes())?;
        pages.push((module.title.clone(), relative));
    }

    let tree = build_tree(&pages);
    write_landing(out_dir, docs.package.as_deref(), &pages)?;
    write_module_indexes(out_dir, &tree)
}

fn render_module_markdown(module: &ModuleDoc) -> String {
    let mut markdown = format!("# `{}`\n\n", module.title);
    if let Some(docs) = &module.docs {
        markdown.push_str(docs);
        markdown.push_str("\n\n");
    }
    if module.items.is_empty() {
        markdown.push_str("_This module exposes no documented items._\n");
        return markdown;
    }

    for item in &module.items {
        markdown.push_str(&format!("## `{}`\n\n", item.name));
        if item.kind == ItemKind::Extension {
            markdown.push_str("_Methods on this type, declared in this module._\n\n");
        } else if let Some(signature) = &item.signature {
            markdown.push_str("```hv\n");
            markdown.push_str(signature);
            markdown.push_str("\n```\n\n");
        }
        if let Some(docs) = &item.docs {
            markdown.push_str(docs);
            markdown.push_str("\n\n");
        }
        render_methods_markdown(&mut markdown, &item.methods);
    }
    markdown
}

fn render_methods_markdown(markdown: &mut String, methods: &[MethodDoc]) {
    if methods.is_empty() {
        return;
    }
    markdown.push_str("### Methods\n\n");
    for method in methods {
        markdown.push_str(&format!("#### `{}`\n\n", method.name));
        markdown.push_str("```hv\n");
        markdown.push_str(&method.signature);
        markdown.push_str("\n```\n\n");
        if let Some(docs) = &method.docs {
            markdown.push_str(docs);
            markdown.push_str("\n\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Static HTML renderer
// ---------------------------------------------------------------------------

fn render_html(docs: &Documentation, out_dir: &Path) -> Result<(), ()> {
    if let Err(e) = std::fs::create_dir_all(out_dir.join("assets")) {
        eprintln!("havendoc: cannot create {}: {}", out_dir.display(), e);
        return Err(());
    }
    write_file(&out_dir.join("assets/style.css"), HTML_STYLE.as_bytes())?;

    let mut pages = Vec::with_capacity(docs.modules.len());
    for module in &docs.modules {
        let relative = title_to_html_path(&module.title);
        let path = out_dir.join(&relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                return Err(());
            }
        let body = render_module_html(module, &relative);
        let page = html_shell(docs, &module.title, &relative, &body);
        write_file(&path, page.as_bytes())?;
        pages.push((module.title.clone(), relative));
    }

    let landing_path = Path::new("index.html");
    let landing = render_html_landing(docs, &pages);
    write_file(
        &out_dir.join(landing_path),
        html_shell(docs, docs.package.as_deref().unwrap_or("Haven API"), landing_path, &landing)
            .as_bytes(),
    )?;

    let tree = build_tree(&pages);
    write_html_module_indexes(docs, out_dir, &tree)
}

fn title_to_html_path(title: &str) -> PathBuf {
    PathBuf::from(format!("{}.html", title))
}

fn render_module_html(module: &ModuleDoc, current: &Path) -> String {
    let mut output = render_breadcrumbs(&module.title, current);
    output.push_str("<header class=\"page-heading\"><p class=\"eyebrow\">Module</p><h1><code>");
    output.push_str(&escape_html(&module.title));
    output.push_str("</code></h1></header>");
    if let Some(docs) = &module.docs {
        output.push_str("<div class=\"prose module-docs\">");
        output.push_str(&markdown_to_html(docs));
        output.push_str("</div>");
    }
    if module.items.is_empty() {
        output.push_str("<p class=\"empty\">This module exposes no documented items.</p>");
        return output;
    }

    output.push_str("<div class=\"items\">");
    for item in &module.items {
        let anchor = item_anchor(item);
        output.push_str("<section class=\"item\" id=\"");
        output.push_str(&anchor);
        output.push_str("\"><header class=\"item-heading\"><span class=\"kind\">");
        output.push_str(item.kind.label());
        output.push_str("</span><h2><a href=\"#");
        output.push_str(&anchor);
        output.push_str("\"><code>");
        output.push_str(&escape_html(&item.name));
        output.push_str("</code></a></h2></header>");
        if item.kind == ItemKind::Extension {
            output.push_str("<p class=\"muted\">Methods on this type declared in this module.</p>");
        } else if let Some(signature) = &item.signature {
            output.push_str(&code_block(signature));
        }
        if let Some(docs) = &item.docs {
            output.push_str("<div class=\"prose\">");
            output.push_str(&markdown_to_html(docs));
            output.push_str("</div>");
        }
        if !item.methods.is_empty() {
            output.push_str("<div class=\"methods\"><h3>Methods</h3>");
            for method in &item.methods {
                let method_anchor = format!("{}-method-{}", anchor, slug(&method.name));
                output.push_str("<article class=\"method\" id=\"");
                output.push_str(&method_anchor);
                output.push_str("\"><h4><a href=\"#");
                output.push_str(&method_anchor);
                output.push_str("\"><code>");
                output.push_str(&escape_html(&method.name));
                output.push_str("</code></a></h4>");
                output.push_str(&code_block(&method.signature));
                if let Some(docs) = &method.docs {
                    output.push_str("<div class=\"prose\">");
                    output.push_str(&markdown_to_html(docs));
                    output.push_str("</div>");
                }
                output.push_str("</article>");
            }
            output.push_str("</div>");
        }
        output.push_str("</section>");
    }
    output.push_str("</div>");
    output
}

fn render_breadcrumbs(title: &str, current: &Path) -> String {
    let prefix = root_prefix(current);
    let parts: Vec<&str> = title.split('/').collect();
    let mut output = format!(
        "<nav class=\"breadcrumbs\" aria-label=\"Breadcrumb\"><a href=\"{}index.html\">Docs</a>",
        prefix
    );
    let mut path = String::new();
    for (index, part) in parts.iter().enumerate() {
        output.push_str("<span aria-hidden=\"true\">/</span>");
        if index + 1 == parts.len() {
            output.push_str("<span>");
            output.push_str(&escape_html(part));
            output.push_str("</span>");
        } else {
            if !path.is_empty() {
                path.push('/');
            }
            path.push_str(part);
            output.push_str("<a href=\"");
            output.push_str(&prefix);
            output.push_str(&escape_html(&path));
            output.push_str("/index.html\">");
            output.push_str(&escape_html(part));
            output.push_str("</a>");
        }
    }
    output.push_str("</nav>");
    output
}

fn render_html_landing(docs: &Documentation, pages: &[(String, PathBuf)]) -> String {
    let (title, intro) = match docs.package.as_deref() {
        Some(name) => (
            format!("<code>{}</code>", escape_html(name)),
            format!("API documentation for the <code>{}</code> Haven package.", escape_html(name)),
        ),
        None => ("Haven API".to_string(), "Generated API documentation.".to_string()),
    };
    let mut output = format!(
        "<header class=\"page-heading landing-heading\"><p class=\"eyebrow\">Haven documentation</p><h1>{}</h1><p>{}</p></header><section><h2>Modules</h2><div class=\"module-grid\">",
        title, intro
    );
    for (name, path) in pages {
        output.push_str("<a class=\"module-card\" href=\"");
        output.push_str(&escape_html(&path.to_string_lossy().replace('\\', "/")));
        output.push_str("\"><code>");
        output.push_str(&escape_html(name));
        output.push_str("</code><span>Open module <span aria-hidden=\"true\">→</span></span></a>");
    }
    output.push_str("</div></section>");
    output
}

fn html_shell(docs: &Documentation, title: &str, current: &Path, body: &str) -> String {
    let prefix = root_prefix(current);
    let navigation = render_html_navigation(docs, current, &prefix);
    render_page_template(
        &escape_html(title),
        &prefix,
        &docs.package.as_deref().map(escape_html).unwrap_or_default(),
        &navigation,
        body,
    )
}

/// Expand the fixed placeholders in `page.html` in one pass. Replacement text
/// is appended directly to the output and is never interpreted as template
/// syntax, which keeps source documentation separate from the template itself.
fn render_page_template(
    title: &str,
    root: &str,
    package: &str,
    navigation: &str,
    content: &str,
) -> String {
    let mut output = String::with_capacity(HTML_PAGE_TEMPLATE.len() + content.len());
    let mut remaining = HTML_PAGE_TEMPLATE;
    while let Some(start) = remaining.find("{{") {
        output.push_str(&remaining[..start]);
        let after_open = &remaining[start + 2..];
        let Some(close) = after_open.find("}}") else {
            output.push_str(&remaining[start..]);
            return output;
        };
        let token = after_open[..close].trim();
        let replacement = match token {
            "title" => title,
            "root" => root,
            "package" => package,
            "navigation" => navigation,
            "content" => content,
            _ => {
                output.push_str(&remaining[start..start + 2 + close + 2]);
                remaining = &after_open[close + 2..];
                continue;
            }
        };
        output.push_str(replacement);
        remaining = &after_open[close + 2..];
    }
    output.push_str(remaining);
    output
}

const HTML_PAGE_TEMPLATE: &str = include_str!("../assets/page.html");

fn render_html_navigation(docs: &Documentation, current: &Path, prefix: &str) -> String {
    let mut output = String::new();
    for module in &docs.modules {
        let path = title_to_html_path(&module.title);
        let active = if path == current {
            " class=\"active\" aria-current=\"page\""
        } else {
            ""
        };
        output.push_str("<li><a");
        output.push_str(active);
        output.push_str(" href=\"");
        output.push_str(prefix);
        output.push_str(&escape_html(&path.to_string_lossy().replace('\\', "/")));
        output.push_str("\"><code>");
        output.push_str(&escape_html(&module.title));
        output.push_str("</code></a></li>");
    }
    output
}

fn write_html_module_indexes(
    docs: &Documentation,
    out_dir: &Path,
    tree: &TreeNode,
) -> Result<(), ()> {
    for (name, node) in &tree.children {
        write_html_index_rec(docs, out_dir, name, node)?;
    }
    Ok(())
}

fn write_html_index_rec(
    docs: &Documentation,
    out_dir: &Path,
    title: &str,
    node: &TreeNode,
) -> Result<(), ()> {
    if node.page.is_none() && !node.children.is_empty() {
        let relative = PathBuf::from(title).join("index.html");
        let mut body = render_breadcrumbs(title, &relative);
        body.push_str("<header class=\"page-heading\"><p class=\"eyebrow\">Namespace</p><h1><code>");
        body.push_str(&escape_html(title));
        body.push_str("</code></h1></header><section><h2>Modules</h2><div class=\"module-grid\">");
        for (child, child_node) in &node.children {
            let link = if child_node.page.is_some() {
                format!("{}.html", child)
            } else {
                format!("{}/index.html", child)
            };
            body.push_str("<a class=\"module-card\" href=\"");
            body.push_str(&escape_html(&link));
            body.push_str("\"><code>");
            body.push_str(&escape_html(child));
            body.push_str("</code><span>Open module <span aria-hidden=\"true\">→</span></span></a>");
        }
        body.push_str("</div></section>");
        let page = html_shell(docs, title, &relative, &body);
        let path = out_dir.join(&relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                return Err(());
            }
        write_file(&path, page.as_bytes())?;
    }
    for (child, child_node) in &node.children {
        write_html_index_rec(docs, out_dir, &format!("{}/{}", title, child), child_node)?;
    }
    Ok(())
}

fn markdown_to_html(markdown: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    let parser = MarkdownParser::new_ext(markdown, options).map(|event| match event {
        // Documentation may eventually come from untrusted registry packages.
        // Preserve raw HTML visibly instead of injecting it into the output.
        Event::Html(raw) | Event::InlineHtml(raw) => Event::Text(raw),
        Event::Start(Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        }) if !safe_doc_url(&dest_url) => Event::Start(Tag::Link {
            link_type,
            dest_url: "#".into(),
            title,
            id,
        }),
        Event::Start(Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        }) if !safe_doc_url(&dest_url) => Event::Start(Tag::Image {
            link_type,
            dest_url: "".into(),
            title,
            id,
        }),
        other => other,
    });
    let mut output = String::new();
    html::push_html(&mut output, parser);
    output
}

fn safe_doc_url(url: &str) -> bool {
    let url = url.trim();
    if url.is_empty()
        || url.starts_with('#')
        || url.starts_with('/')
        || url.starts_with("./")
        || url.starts_with("../")
    {
        return true;
    }
    let scheme_end = url.find(':');
    let path_start = url.find(['/', '?', '#']).unwrap_or(usize::MAX);
    match scheme_end.filter(|colon| *colon < path_start) {
        Some(colon) => matches!(
            url[..colon].to_ascii_lowercase().as_str(),
            "http" | "https" | "mailto"
        ),
        None => true,
    }
}

fn code_block(code: &str) -> String {
    format!(
        "<pre class=\"signature\"><code class=\"language-hv\">{}</code></pre>",
        escape_html(code)
    )
}

fn item_anchor(item: &ItemDoc) -> String {
    format!("{}-{}", item.kind.label(), slug(&item.name))
}

fn slug(value: &str) -> String {
    let mut output = String::new();
    let mut separated = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            output.push(character.to_ascii_lowercase());
            separated = false;
        } else if !separated && !output.is_empty() {
            output.push('-');
            separated = true;
        }
    }
    while output.ends_with('-') {
        output.pop();
    }
    if output.is_empty() { "item".to_string() } else { output }
}

fn root_prefix(path: &Path) -> String {
    "../".repeat(path.parent().map_or(0, |parent| parent.components().count()))
}

fn escape_html(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\'' => output.push_str("&#39;"),
            other => output.push(other),
        }
    }
    output
}

const HTML_STYLE: &str = include_str!("../assets/style.css");

// ---------------------------------------------------------------------------
// Markdown indexes
// ---------------------------------------------------------------------------

/// Write the documentation landing page (`index.md`) with links to every
/// documented module.
fn write_landing(
    out_dir: &Path,
    package: Option<&str>,
    pages: &[(String, PathBuf)],
) -> Result<(), ()> {
    let mut md = match package {
        Some(name) => format!("# `{}`\n\nAPI documentation for the `{}` Haven package.\n\n", name, name),
        None => "# Haven API\n\nGenerated API documentation.\n\n".to_string(),
    };
    md.push_str("## Modules\n\n");
    for (title, path) in pages {
        md.push_str(&format!("- [{}]({})\n", title, path.to_string_lossy().replace('\\', "/")));
    }
    let path = out_dir.join("index.md");
    std::fs::write(&path, md).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
}

/// A node in the module tree built from page titles: `std/dsp/osc` descends
/// `std` -> `dsp` -> `osc`. A node carries its own page when a `.hv` file sits
/// at exactly that path; pure directories (`dsp`) have `page == None` and get a
/// generated index page so they can still be a clickable parent chapter.
#[derive(Default)]
struct TreeNode {
    /// Path of this node's own documented page, relative to the output directory.
    page: Option<PathBuf>,
    /// Child modules/directories, keyed by their last path component. `BTreeMap`
    /// keeps the sidebar order stable and alphabetical.
    children: BTreeMap<String, TreeNode>,
}

/// Fold the flat `(title, rel_md)` page list into a directory tree keyed on the
/// `/`-separated title components.
fn build_tree(pages: &[(String, PathBuf)]) -> TreeNode {
    let mut root = TreeNode::default();
    for (title, rel) in pages {
        let mut node = &mut root;
        for comp in title.split('/') {
            node = node.children.entry(comp.to_string()).or_default();
        }
        node.page = Some(rel.clone());
    }
    root
}

/// Write an index page for every directory node that lacks its own `.hv` page,
/// so it can serve as a clickable parent chapter (à la a Rust module page that
/// lists its submodules). Real module pages are left untouched.
fn write_module_indexes(out_dir: &Path, tree: &TreeNode) -> Result<(), ()> {
    for (name, node) in &tree.children {
        write_index_rec(out_dir, name, node)?;
    }
    Ok(())
}

fn write_index_rec(out_dir: &Path, path: &str, node: &TreeNode) -> Result<(), ()> {
    if node.page.is_none() && !node.children.is_empty() {
        let mut md = format!("# `{}`\n\n## Modules\n\n", path);
        for (child, child_node) in &node.children {
            // Links are relative to this index page's own directory
            // (`<out>/<path>/index.md`), so a child is just its last component.
            let link = if child_node.page.is_some() {
                format!("{}.md", child)
            } else {
                format!("{}/index.md", child)
            };
            md.push_str(&format!("- [{}]({})\n", child, link));
        }
        let page_path = out_dir.join(path).join("index.md");
        if let Some(parent) = page_path.parent()
            && let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                return Err(());
            }
        write_file(&page_path, md.as_bytes())?;
    }
    for (child, child_node) in &node.children {
        write_index_rec(out_dir, &format!("{}/{}", path, child), child_node)?;
    }
    Ok(())
}

/// Write `bytes` to `path`, reporting any error under the `havendoc:` prefix.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), ()> {
    std::fs::write(path, bytes).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_builds_a_renderer_independent_module_model() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("math.hv");
        std::fs::write(
            &path,
            "/// Package arithmetic.\n\n/// Return the supplied value.\npub proc identity(value: i32) i32 { return value; }\n\nproc hidden() void {}\n",
        )
        .unwrap();
        let source = SourceFile {
            title: "sample/math".to_string(),
            file: path,
            package: Some("sample".to_string()),
        };

        let module = extract_module(&source).unwrap();

        assert_eq!(module.title, "sample/math");
        assert_eq!(module.docs.as_deref(), Some("Package arithmetic."));
        assert_eq!(module.items.len(), 1);
        assert_eq!(module.items[0].kind, ItemKind::Function);
        assert_eq!(module.items[0].name, "identity");
        assert_eq!(
            module.items[0].signature.as_deref(),
            Some("proc identity(value: i32) i32")
        );
        assert_eq!(
            module.items[0].docs.as_deref(),
            Some("Return the supplied value.")
        );
    }

    #[test]
    fn html_renderer_writes_a_portable_safe_static_site() {
        let temp = tempfile::tempdir().unwrap();
        let docs = Documentation {
            package: Some("audio".to_string()),
            modules: vec![ModuleDoc {
                title: "audio/math".to_string(),
                docs: Some(
                    "Use **carefully**. <script>alert('no')</script> [unsafe](javascript:alert(1))"
                        .to_string(),
                ),
                items: vec![ItemDoc {
                    kind: ItemKind::Function,
                    name: "identity".to_string(),
                    signature: Some("proc identity(value: i32) i32".to_string()),
                    docs: None,
                    methods: Vec::new(),
                }],
            }],
        };

        render_html(&docs, temp.path()).unwrap();

        assert!(temp.path().join("index.html").is_file());
        assert!(temp.path().join("audio/index.html").is_file());
        assert!(temp.path().join("assets/style.css").is_file());
        assert_eq!(
            std::fs::read_to_string(temp.path().join("assets/style.css")).unwrap(),
            HTML_STYLE
        );
        assert!(!HTML_STYLE.contains("border-radius"));
        assert!(HTML_STYLE.contains("scrollbar-color"));
        assert!(HTML_STYLE.contains("::-webkit-scrollbar-thumb"));
        let module = std::fs::read_to_string(temp.path().join("audio/math.html")).unwrap();
        assert!(!module.contains("{{"));
        assert!(module.contains("href=\"../assets/style.css\""));
        assert!(module.contains("id=\"function-identity\""));
        assert!(module.contains("<strong>carefully</strong>"));
        assert!(module.contains("&lt;script&gt;alert('no')&lt;/script&gt;"));
        assert!(!module.contains("<script>"));
        assert!(!module.contains("javascript:"));
    }

    #[test]
    fn package_input_uses_manifest_name_and_src_as_root() {
        let temp = tempfile::tempdir().unwrap();
        let package = temp.path().join("checkout-name-does-not-matter");
        std::fs::create_dir_all(package.join("src/dsp")).unwrap();
        std::fs::write(
            package.join("vestry.toml"),
            "[project]\nname = \"audio\"\nkind = [\"lib\"]\n",
        )
        .unwrap();
        std::fs::write(package.join("src/lib.hv"), "").unwrap();
        std::fs::write(package.join("src/dsp/osc.hv"), "").unwrap();
        std::fs::write(package.join("build.hv"), "").unwrap();

        let files = collect_hv_files(&[package]);
        let titles: Vec<&str> = files.iter().map(|file| file.title.as_str()).collect();

        assert_eq!(titles, ["audio/dsp/osc", "audio/lib"]);
    }

    #[test]
    fn ordinary_directory_keeps_its_directory_name() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("sources");
        std::fs::create_dir_all(input.join("nested")).unwrap();
        std::fs::write(input.join("root.hv"), "").unwrap();
        std::fs::write(input.join("nested/item.hv"), "").unwrap();

        let files = collect_hv_files(&[input]);
        let titles: Vec<&str> = files.iter().map(|file| file.title.as_str()).collect();

        assert_eq!(titles, ["sources/nested/item", "sources/root"]);
    }
}
