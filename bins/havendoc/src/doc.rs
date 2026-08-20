//! `havendoc`: turn `.hv` source into mdBook-ready Markdown.
//!
//! The lexer discards comments as padding (see `parse::lexer`), so doc text
//! never reaches the AST. Rather than thread doc tokens through the whole
//! grammar, it parses each file for *real* signatures (generics, types,
//! attributes all rendered via the AST's `Display` impls) and recovers the
//! `///` doc bodies straight from the source bytes, keyed by each item's `Span`.
//! Same result as attaching docs to the AST.
//!
//! Output layout, given `havendoc std -o docs`:
//! ```text
//! docs/
//!   book.toml
//!   src/
//!     SUMMARY.md
//!     std/alloc.md
//!     std/dsp/osc.md
//!     ...
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bumpalo::Bump;

use crate::DocArgs;
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

    let src_dir = args.out.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("havendoc: cannot create {}: {}", src_dir.display(), e);
        return Err(());
    }

    // (module title, page path relative to src/) for the SUMMARY, in the order
    // pages were emitted.
    let mut pages: Vec<(String, PathBuf)> = Vec::new();

    for (strip_base, file) in &files {
        let title = module_title(strip_base, file);
        let rel_md = title_to_rel_path(&title);
        let page_path = src_dir.join(&rel_md);

        let markdown = match render_file(&title, file) {
            Ok(md) => md,
            Err(()) => {
                eprintln!("havendoc: skipping {} (failed to parse)", file.display());
                continue;
            }
        };

        if let Some(parent) = page_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                continue;
            }
        }
        if let Err(e) = std::fs::write(&page_path, markdown) {
            eprintln!("havendoc: cannot write {}: {}", page_path.display(), e);
            continue;
        }
        pages.push((title, rel_md));
    }

    if pages.is_empty() {
        eprintln!("havendoc: no pages generated");
        return Err(());
    }

    // Stable, human-friendly ordering in the sidebar.
    pages.sort_by(|a, b| a.0.cmp(&b.0));

    // Build a directory tree from the `a/b/c` page titles so nested modules
    // (`dsp/osc`) render as collapsible subsections instead of a flat list.
    let tree = build_tree(&pages);
    write_landing(&src_dir, "haven std", &pages)?;
    write_module_indexes(&src_dir, &tree)?;
    write_summary(&src_dir, &tree)?;
    write_book_toml(&args.out, "haven std")?;
    write_theme(&args.out)?;

    println!(
        "havendoc: wrote {} page(s) to {}",
        pages.len(),
        args.out.display()
    );
    Ok(())
}

/// Recursively gather `.hv` files from each input. Returns `(strip_base, file)`
/// pairs, where `strip_base` is the path prefix stripped off `file` to form its
/// module title. For a directory input it's the directory's *parent*, so the
/// directory name is kept (`std/` -> `std/dsp/osc`); for a file input it's the
/// file's parent, so a lone `foo/bar.hv` titles as just `bar`.
fn collect_hv_files(inputs: &[PathBuf]) -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let base = input.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
            walk_dir(&base, input, &mut out);
        } else if is_hv(input) {
            let base = input.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
            out.push((base, input.clone()));
        } else {
            eprintln!("havendoc: skipping {} (not a .hv file)", input.display());
        }
    }
    out
}

fn walk_dir(base: &Path, dir: &Path, out: &mut Vec<(PathBuf, PathBuf)>) {
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
            walk_dir(base, &path, out);
        } else if is_hv(&path) {
            out.push((base.to_path_buf(), path));
        }
    }
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
// Per-file rendering
// ---------------------------------------------------------------------------

fn render_file(title: &str, file: &Path) -> Result<String, ()> {
    let src = std::fs::read_to_string(file).map_err(|e| {
        eprintln!("havendoc: cannot read {}: {}", file.display(), e);
    })?;

    // Parse into real signatures. A fresh arena per file is fine: everything we
    // keep (the rendered strings) is owned by the time it drops.
    let arena = Bump::new();
    let src_ref: &str = arena.alloc_str(&src);

    // havendoc renders one file at a time and never reports a span-carrying
    // diagnostic (a lex/parse failure just fails the page), so a one-entry file
    // table is all the spans need to be well-formed.
    let mut files = Files::new();
    let key = files.add(file.to_string_lossy().into_owned(), src_ref);

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

    let mut md = String::new();
    md.push_str(&format!("# `{}`\n\n", title));

    // Module-level doc: a leading `///` block separated from the first item by a
    // blank line (mirrors the intent of Rust's `//!`).
    if let Some(doc) = lines.module_doc() {
        md.push_str(&doc);
        md.push_str("\n\n");
    }

    if items.is_empty() {
        md.push_str("_This module exposes no documented items._\n");
        return Ok(md);
    }

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

    for item in &items {
        match &item.value {
            // extend blocks are rendered as method sections under their type, not
            // as items in their own right; handled above/below.
            TopLevelNode::Extend { .. } => continue,
            // docs describe a module's public surface; private (non-`pub`) items
            // are implementation details and are omitted entirely.
            node if !item_is_pub(node) => continue,
            node => {
                render_item(&mut md, item, &src, &lines);
                // a type carries its methods directly beneath its declaration.
                if let TopLevelNode::Struct { name, .. } | TopLevelNode::Enum { name, .. } = node {
                    if let Some(methods) = method_groups.remove(*name) {
                        render_methods(&mut md, &methods, &lines);
                    }
                }
            }
        }
    }

    // methods extending a type not declared here (`extend i32`, or an imported
    // type). Only emitted when a group actually has visible methods.
    for (target, methods) in &method_groups {
        if methods.is_empty() {
            continue;
        }
        md.push_str(&format!("## `{}`\n\n", target));
        md.push_str("_Methods on this type, declared in this module._\n\n");
        render_methods(&mut md, methods, &lines);
    }

    Ok(md)
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

/// Render a type's methods under a `### Methods` heading nested in the type's
/// `##` section, each method its own `#### <name>` subsection: a heading, its
/// `hv` signature block, then its `///` doc block if it has one.
fn render_methods(md: &mut String, methods: &[&Method], lines: &LineIndex) {
    if methods.is_empty() {
        return;
    }
    md.push_str("### Methods\n\n");
    for m in methods {
        md.push_str(&format!("#### `{}`\n\n", m.value.name));
        md.push_str("```hv\n");
        md.push_str(&method_signature(&m.value));
        md.push_str("\n```\n\n");
        if let Some(doc) = lines.doc_above(m.span.start) {
            md.push_str(&doc);
            md.push_str("\n\n");
        }
    }
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

fn render_item(md: &mut String, item: &TopLevel, src: &str, lines: &LineIndex) {
    let name = item_name(&item.value);
    let signature = signature(&item.value, src, item.span.start, item.span.end);
    let doc = lines.doc_above(item.span.start);

    md.push_str(&format!("## `{}`\n\n", name));
    md.push_str("```hv\n");
    md.push_str(&signature);
    md.push_str("\n```\n\n");
    if let Some(doc) = doc {
        md.push_str(&doc);
        md.push_str("\n\n");
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
// Book scaffolding
// ---------------------------------------------------------------------------

/// Write the book's landing page (`src/index.md`). mdBook renders the *first*
/// chapter to the book root's `index.html`, and `write_summary` lists this as a
/// prefix chapter ahead of everything else, so this becomes the home page a
/// bare visit to the book lands on (à la `doc.rust-lang.org/std/`). It's a title,
/// a blurb, and an auto-generated index of every documented module.
fn write_landing(src_dir: &Path, title: &str, pages: &[(String, PathBuf)]) -> Result<(), ()> {
    let mut md = format!(
        "# {}\n\nThe `haven` standard library.\n\n## Modules\n\n",
        title
    );
    for (title, path) in pages {
        md.push_str(&format!("- [{}]({})\n", title, path.to_string_lossy().replace('\\', "/")));
    }
    let path = src_dir.join("index.md");
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
    /// Rel-to-`src/` `.md` path of this node's own documented page, if any.
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
fn write_module_indexes(src_dir: &Path, tree: &TreeNode) -> Result<(), ()> {
    for (name, node) in &tree.children {
        write_index_rec(src_dir, name, node)?;
    }
    Ok(())
}

fn write_index_rec(src_dir: &Path, path: &str, node: &TreeNode) -> Result<(), ()> {
    if node.page.is_none() && !node.children.is_empty() {
        let mut md = format!("# `{}`\n\n## Modules\n\n", path);
        for (child, child_node) in &node.children {
            // Links are relative to this index page's own directory
            // (`src/<path>/index.md`), so a child is just its last component.
            let link = if child_node.page.is_some() {
                format!("{}.md", child)
            } else {
                format!("{}/index.md", child)
            };
            md.push_str(&format!("- [{}]({})\n", child, link));
        }
        let page_path = src_dir.join(path).join("index.md");
        if let Some(parent) = page_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                return Err(());
            }
        }
        write_file(&page_path, md.as_bytes())?;
    }
    for (child, child_node) in &node.children {
        write_index_rec(src_dir, &format!("{}/{}", path, child), child_node)?;
    }
    Ok(())
}

fn write_summary(src_dir: &Path, tree: &TreeNode) -> Result<(), ()> {
    // The landing page (`index.md`) goes first as a prefix chapter (no list
    // marker), so mdBook renders it to the book root's `index.html`. The rest is
    // an indented tree: nesting depth = directory depth, which mdBook turns into
    // collapsible subsections.
    let mut summary = String::from("# Summary\n\n[Overview](index.md)\n\n");
    for (name, node) in &tree.children {
        emit_summary_node(&mut summary, name, name, node, 0);
    }
    let path = src_dir.join("SUMMARY.md");
    std::fs::write(&path, summary).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
}

/// Append one chapter line (4-space indent per level, as mdBook requires for
/// nesting) plus its descendants. `path` is the full title path used for link
/// resolution; `name` is just the displayed last component.
fn emit_summary_node(out: &mut String, path: &str, name: &str, node: &TreeNode, depth: usize) {
    let indent = "    ".repeat(depth);
    // Summary links resolve from `src/`, so use the full path. A directory
    // without its own page links to the index we generated for it.
    let link = match &node.page {
        Some(rel) => rel.to_string_lossy().replace('\\', "/"),
        None => format!("{}/index.md", path),
    };
    out.push_str(&format!("{}- [{}]({})\n", indent, name, link));
    for (child, child_node) in &node.children {
        emit_summary_node(out, &format!("{}/{}", path, child), child, child_node, depth + 1);
    }
}

fn write_book_toml(out: &Path, title: &str) -> Result<(), ()> {
    // The Oxocarbon theme reuses mdBook's built-in `navy` slot for its dark
    // variant (navy already selects the dark syntax-highlight stylesheet) and
    // `light` for its light variant; see `write_theme`. `navy` is therefore the
    // default and the preferred dark theme.
    let toml = format!(
        "[book]\ntitle = \"{}\"\nlanguage = \"en\"\nsrc = \"src\"\n\n\
         [output.html]\ndefault-theme = \"navy\"\npreferred-dark-theme = \"navy\"\nhash-files = false\n",
        title.replace('"', "\\\"")
    );
    let path = out.join("book.toml");
    std::fs::write(&path, toml).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
}

/// Oxocarbon (nyoom-engineering, an IBM Carbon Design System palette) as an
/// mdBook theme, emitted so a plain `havendoc` run needs no manual theming.
///
/// Only `theme/css/variables.css` and `theme/highlight.js` are overridden:
/// mdBook falls back to its built-in `index.hbs`/`book.js` for everything else.
/// (`highlight.js` is the one vendored mdBook internal; see `THEME_HIGHLIGHT_JS`
/// for why.) The dark variant is mapped onto the built-in
/// `navy` theme slot (which already loads the dark highlight CSS) and the light
/// variant onto `light`; the trailing rules restrict the theme picker to just
/// those two and relabel them "Oxocarbon" / "Oxocarbon Light".
const THEME_VARIABLES_CSS: &str = include_str!("../assets/variables.css");

/// `theme/fonts/fonts.css`, overriding mdBook's default Open Sans / Source Code
/// Pro `@font-face` set with Geist / Geist Mono. Uses mdBook's `{{ resource }}`
/// helper, so it is rendered as a template at book-build time.
const THEME_FONTS_CSS: &str = include_str!("../assets/fonts.css");

/// `theme/highlight.js`: mdBook's bundled highlight.js (10.1.1, BSD-3-Clause)
/// with a haven (`hv`) language grammar appended. mdBook highlights client-side
/// and has no append hook, only a wholesale `theme/highlight.js` override, so the
/// core is necessarily vendored and version-pinned here. Without it, ```` ```hv ````
/// fences render unhighlighted since stock highlight.js has no `hv` language. The
/// grammar emits standard hljs scopes (`keyword`/`string`/`number`/`meta`/`type`
/// /...), which the theme's highlight CSS already colors. To refresh the base:
/// grab `highlight.js` from any `mdbook build` output and re-append the grammar.
const THEME_HIGHLIGHT_JS: &str = include_str!("../assets/highlight.js");

/// The Geist family (variable, weight axis 100-900) plus its SIL OFL license,
/// embedded so a plain `havendoc` run ships self-contained fonts. `(filename,
/// bytes)`; mdBook copies everything in `theme/fonts/` into the book and
/// content-hashes the woff2s. The `OFL.txt` accompanies the fonts as the license
/// requires.
const THEME_FONTS: &[(&str, &[u8])] = &[
    ("Geist.woff2", include_bytes!("../assets/fonts/Geist.woff2")),
    ("Geist-Italic.woff2", include_bytes!("../assets/fonts/Geist-Italic.woff2")),
    ("GeistMono.woff2", include_bytes!("../assets/fonts/GeistMono.woff2")),
    ("GeistMono-Italic.woff2", include_bytes!("../assets/fonts/GeistMono-Italic.woff2")),
    ("OFL.txt", include_bytes!("../assets/fonts/OFL.txt")),
];

/// Write the theme override under `<out>/theme`: the Oxocarbon color variables
/// (`css/variables.css`), the haven syntax-highlighting grammar (`highlight.js`),
/// plus the Geist fonts and their `@font-face` declarations (`fonts/`). mdBook
/// falls back to its built-in `index.hbs`/`book.js`/`general.css` for everything
/// else.
fn write_theme(out: &Path) -> Result<(), ()> {
    let theme = out.join("theme");

    let css_dir = theme.join("css");
    if let Err(e) = std::fs::create_dir_all(&css_dir) {
        eprintln!("havendoc: cannot create {}: {}", css_dir.display(), e);
        return Err(());
    }
    write_file(&css_dir.join("variables.css"), THEME_VARIABLES_CSS.as_bytes())?;

    write_file(&theme.join("highlight.js"), THEME_HIGHLIGHT_JS.as_bytes())?;

    let fonts_dir = theme.join("fonts");
    if let Err(e) = std::fs::create_dir_all(&fonts_dir) {
        eprintln!("havendoc: cannot create {}: {}", fonts_dir.display(), e);
        return Err(());
    }
    write_file(&fonts_dir.join("fonts.css"), THEME_FONTS_CSS.as_bytes())?;
    for (name, bytes) in THEME_FONTS {
        write_file(&fonts_dir.join(name), bytes)?;
    }
    Ok(())
}

/// Write `bytes` to `path`, reporting any error under the `havendoc:` prefix.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), ()> {
    std::fs::write(path, bytes).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
}
