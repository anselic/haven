use super::extract::title_to_rel_path;
use super::*;

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

pub(super) fn render_markdown(docs: &Documentation, out_dir: &Path) -> Result<(), ()> {
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("havendoc: cannot create {}: {}", out_dir.display(), e);
        return Err(());
    }

    let mut pages = Vec::with_capacity(docs.modules.len());
    for module in &docs.modules {
        let relative = title_to_rel_path(&module.title);
        let path = out_dir.join(&relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
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
        if let Some(signature) = &item.signature {
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
        Some(name) => format!(
            "# `{}`\n\nAPI documentation for the `{}` Haven package.\n\n",
            name, name
        ),
        None => "# Haven API\n\nGenerated API documentation.\n\n".to_string(),
    };
    md.push_str("## Modules\n\n");
    for (title, path) in pages {
        md.push_str(&format!(
            "- [{}]({})\n",
            title,
            path.to_string_lossy().replace('\\', "/")
        ));
    }
    let path = out_dir.join("index.md");
    std::fs::write(&path, md).map_err(|e| {
        eprintln!("havendoc: cannot write {}: {}", path.display(), e);
    })
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
            && let Err(e) = std::fs::create_dir_all(parent)
        {
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
