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
//!   std/index.md
//!   std/alloc.md
//!   std/dsp/osc.md
//!   ...
//! ```
//! `--format html` emits the same module structure as `.html` pages plus a
//! shared `assets/style.css`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::{DocArgs, OutputFormat};

mod extract;
mod highlight;
mod html;
mod markdown;
#[cfg(test)]
mod tests;

#[cfg(test)]
use extract::SourceFile;
use extract::{collect_hv_files, extract_module};
use html::render_html;
#[cfg(test)]
use html::{HTML_FONT_ASSETS, HTML_STYLE};
use markdown::render_markdown;

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
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ItemDoc {
    kind: ItemKind,
    name: String,
    /// Haven declaration without its implementation body.
    signature: Option<String>,
    docs: Option<String>,
    methods: Vec<MethodDoc>,
    extended_methods: Vec<MethodDoc>,
}

#[derive(Debug, PartialEq, Eq)]
struct MethodDoc {
    name: String,
    signature: String,
    docs: Option<String>,
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

/// Write `bytes` to `path`, reporting any error under the `havendoc:` prefix.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), ()> {
    std::fs::write(path, bytes).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
}