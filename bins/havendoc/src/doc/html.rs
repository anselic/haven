use super::*;
use pulldown_cmark::{CodeBlockKind, Event, Options, Parser as MarkdownParser, Tag, TagEnd, html};

// ---------------------------------------------------------------------------
// Static HTML renderer
// ---------------------------------------------------------------------------

pub(super) fn render_html(docs: &Documentation, out_dir: &Path) -> Result<(), ()> {
    let assets_dir = out_dir.join("assets");
    let fonts_dir = assets_dir.join("fonts");
    if let Err(e) = std::fs::create_dir_all(&fonts_dir) {
        eprintln!("havendoc: cannot create {}: {}", out_dir.display(), e);
        return Err(());
    }
    write_file(&assets_dir.join("style.css"), HTML_STYLE.as_bytes())?;
    for (name, contents) in HTML_FONT_ASSETS {
        write_file(&fonts_dir.join(name), contents)?;
    }

    let mut pages = Vec::with_capacity(docs.modules.len());
    for module in &docs.modules {
        let relative = title_to_html_path(&module.title);
        let path = out_dir.join(&relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
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
        html_shell(
            docs,
            docs.package.as_deref().unwrap_or("Haven API"),
            landing_path,
            &landing,
        )
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
        if let Some(signature) = &item.signature {
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
            format!(
                "API documentation for the <code>{}</code> Haven package.",
                escape_html(name)
            ),
        ),
        None => (
            "Haven API".to_string(),
            "Generated API documentation.".to_string(),
        ),
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

const HTML_PAGE_TEMPLATE: &str = include_str!("../../assets/page.html");

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
        body.push_str(
            "<header class=\"page-heading\"><p class=\"eyebrow\">Namespace</p><h1><code>",
        );
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
            body.push_str(
                "</code><span>Open module <span aria-hidden=\"true\">→</span></span></a>",
            );
        }
        body.push_str("</div></section>");
        let page = html_shell(docs, title, &relative, &body);
        let path = out_dir.join(&relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
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
    let mut parser = MarkdownParser::new_ext(markdown, options).map(|event| match event {
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
    let mut events = Vec::new();
    while let Some(event) = parser.next() {
        match event {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(ref info)))
                if matches!(info.split_whitespace().next(), Some("hv" | "haven")) =>
            {
                let mut code = String::new();
                for content in parser.by_ref() {
                    match content {
                        Event::End(TagEnd::CodeBlock) => break,
                        Event::Text(text) => code.push_str(&text),
                        Event::SoftBreak | Event::HardBreak => code.push('\n'),
                        _ => {}
                    }
                }
                events.push(Event::Html(
                    format!(
                        "<pre><code class=\"language-hv\">{}</code></pre>",
                        super::highlight::haven(&code)
                    )
                    .into(),
                ));
            }
            other => events.push(other),
        }
    }
    let mut output = String::new();
    html::push_html(&mut output, events.into_iter());
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
        super::highlight::haven(code)
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
    if output.is_empty() {
        "item".to_string()
    } else {
        output
    }
}

fn root_prefix(path: &Path) -> String {
    "../".repeat(
        path.parent()
            .map_or(0, |parent| parent.components().count()),
    )
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

pub(super) const HTML_STYLE: &str = include_str!("../../assets/style.css");
pub(super) const HTML_FONT_ASSETS: &[(&str, &[u8])] = &[
    (
        "Geist-VariableFont_wght.ttf",
        include_bytes!("../../assets/fonts/Geist-VariableFont_wght.ttf"),
    ),
    (
        "Geist-Italic-VariableFont_wght.ttf",
        include_bytes!("../../assets/fonts/Geist-Italic-VariableFont_wght.ttf"),
    ),
    (
        "GeistMono-VariableFont_wght.ttf",
        include_bytes!("../../assets/fonts/GeistMono-VariableFont_wght.ttf"),
    ),
    (
        "GeistMono-Italic-VariableFont_wght.ttf",
        include_bytes!("../../assets/fonts/GeistMono-Italic-VariableFont_wght.ttf"),
    ),
    ("OFL.txt", include_bytes!("../../assets/fonts/OFL.txt")),
];

#[cfg(test)]
mod tests {
    use super::{code_block, markdown_to_html};

    #[test]
    fn highlights_haven_signatures_and_fences() {
        let signature = code_block("pub proc identity(value: i32) i32");
        assert!(signature.contains("<span class=\"syntax-keyword\">proc</span>"));
        assert!(signature.contains("<span class=\"syntax-type\">i32</span>"));

        let prose =
            markdown_to_html("```hv\nlet value = \"<unsafe>\";\n```\n\n```text\nlet plain\n```");
        assert!(prose.contains("<span class=\"syntax-keyword\">let</span>"));
        assert!(prose.contains("&lt;unsafe&gt;"));
        assert!(prose.contains("<code class=\"language-text\">let plain"));
    }
}
