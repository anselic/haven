//! `havendoc`: turn `.hv` source into Markdown.
//!
//! Signatures come from the parsed AST. Since the lexer discards comments, doc
//! bodies are read from the source and matched to items by span.
//!
//! Output layout, given `havendoc std -o docs`:
//! ```text
//! docs/
//!   index.md
//!   std/alloc.md
//!   std/dsp/osc.md
//!   ...
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

    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("havendoc: cannot create {}: {}", args.out.display(), e);
        return Err(());
    }

    // Module title and page path relative to the output directory.
    let mut pages: Vec<(String, PathBuf)> = Vec::new();

    for source in &files {
        let rel_md = title_to_rel_path(&source.title);
        let page_path = args.out.join(&rel_md);

        let markdown = match render_file(&source.title, &source.file) {
            Ok(md) => md,
            Err(()) => {
                eprintln!(
                    "havendoc: skipping {} (failed to parse)",
                    source.file.display()
                );
                continue;
            }
        };

        if let Some(parent) = page_path.parent()
            && let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("havendoc: cannot create {}: {}", parent.display(), e);
                continue;
            }
        if let Err(e) = std::fs::write(&page_path, markdown) {
            eprintln!("havendoc: cannot write {}: {}", page_path.display(), e);
            continue;
        }
        pages.push((source.title.clone(), rel_md));
    }

    if pages.is_empty() {
        eprintln!("havendoc: no pages generated");
        return Err(());
    }

    // Stable, human-friendly ordering in generated indexes.
    pages.sort_by(|a, b| a.0.cmp(&b.0));

    // Build a directory tree from `a/b/c` page titles so directories without a
    // corresponding module can get a useful Markdown index.
    let tree = build_tree(&pages);
    let package_name = files
        .first()
        .and_then(|source| source.package.as_deref())
        .filter(|name| {
            files
                .iter()
                .all(|source| source.package.as_deref() == Some(*name))
        });
    write_landing(&args.out, package_name, &pages)?;
    write_module_indexes(&args.out, &tree)?;

    println!(
        "havendoc: wrote {} page(s) to {}",
        pages.len(),
        args.out.display()
    );
    Ok(())
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
                if let TopLevelNode::Struct { name, .. } | TopLevelNode::Enum { name, .. } = node
                    && let Some(methods) = method_groups.remove(*name) {
                        render_methods(&mut md, &methods, &lines);
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
