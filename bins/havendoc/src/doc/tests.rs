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
fn explicit_extensions_have_their_own_method_section() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("point.hv");
    std::fs::write(
        &path,
        r#"
pub struct Point {
    x: i32,
    pub proc new() Point { return Point { x: 0i32 }; }
}

extend Point {
    pub proc value(*self) i32 { return self.x; }
}

extend Point: Display {
    proc display(*self) str { return "point"; }
}
"#,
    )
    .unwrap();
    let source = SourceFile {
        title: "sample/point".to_string(),
        file: path,
        package: Some("sample".to_string()),
    };
    let module = extract_module(&source).unwrap();
    let point = &module.items[0];
    assert_eq!(
        point
            .methods
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        ["new"]
    );
    assert_eq!(
        point
            .extended_methods
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        ["value", "display"]
    );

    let docs = Documentation {
        package: Some("sample".to_string()),
        modules: vec![module],
    };
    let markdown_dir = temp.path().join("markdown");
    render_markdown(&docs, &markdown_dir).unwrap();
    let markdown = std::fs::read_to_string(markdown_dir.join("sample/point.md")).unwrap();
    assert!(markdown.contains("### Methods\n\n#### `new`"));
    assert!(markdown.contains("### Extended Methods\n\n#### `value`"));
    let html_dir = temp.path().join("html");
    render_html(&docs, &html_dir).unwrap();
    let html = std::fs::read_to_string(html_dir.join("sample/point.html")).unwrap();
    assert!(html.contains("<h3>Extended Methods</h3>"));
    assert!(html.contains("id=\"struct-point-method-display\""));
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
                extended_methods: Vec::new(),
            }],
        }],
    };

    render_html(&docs, temp.path()).unwrap();

    assert!(temp.path().join("index.html").is_file());
    assert!(temp.path().join("audio/index.html").is_file());
    assert!(temp.path().join("assets/style.css").is_file());
    assert!(temp.path().join("assets/search.js").is_file());
    let search_index = std::fs::read_to_string(temp.path().join("assets/search-index.js")).unwrap();
    assert!(search_index.contains("[\"audio/math\",\"module\",\"audio/math.html\"]"));
    assert!(
        search_index.contains(
            "[\"audio/math::identity\",\"function\",\"audio/math.html#function-identity\"]"
        )
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("assets/style.css")).unwrap(),
        HTML_STYLE
    );
    for (name, contents) in HTML_FONT_ASSETS {
        assert_eq!(
            std::fs::read(temp.path().join("assets/fonts").join(name)).unwrap(),
            *contents
        );
    }
    assert!(HTML_STYLE.contains("font-family: \"Geist\""));
    assert!(HTML_STYLE.contains("font-family: \"Geist Mono\""));
    assert!(HTML_STYLE.contains("font-weight: 100 900"));
    assert!(!HTML_STYLE.contains("border-radius"));
    assert!(!HTML_STYLE.contains("--accent"));
    assert!(HTML_STYLE.contains("scrollbar-color"));
    assert!(HTML_STYLE.contains("::-webkit-scrollbar-thumb"));
    let module = std::fs::read_to_string(temp.path().join("audio/math.html")).unwrap();
    assert!(!module.contains("{{"));
    assert!(module.contains("href=\"../assets/style.css\""));
    assert!(module.contains("src=\"../assets/search-index.js\""));
    assert!(module.contains("data-root=\"../\""));
    assert!(module.contains("aria-controls=\"search-dialog\""));
    assert!(module.contains("id=\"function-identity\""));
    assert!(module.contains("<strong>carefully</strong>"));
    assert!(module.contains("&lt;script&gt;alert('no')&lt;/script&gt;"));
    assert!(!module.contains("<script>"));
    assert!(!module.contains("javascript:"));
    let landing = std::fs::read_to_string(temp.path().join("index.html")).unwrap();
    assert!(landing.contains("API documentation for the <code>audio</code> Haven package."));
}

#[test]
fn package_lib_is_the_root_page_in_both_formats() {
    let temp = tempfile::tempdir().unwrap();
    let package_dir = temp.path().join("package");
    std::fs::create_dir_all(package_dir.join("src")).unwrap();
    std::fs::write(
        package_dir.join("vestry.toml"),
        "[project]\nname = \"audio\"\n",
    )
    .unwrap();
    let source_path = package_dir.join("src/lib.hv");
    std::fs::write(
        &source_path,
        "//! # Audio library\n//!\n//! Use **carefully**. <script>bad()</script>\n//!\n//! ```hv\n//! proc play() void\n//! ```\npub proc play() void {}\n",
    )
    .unwrap();
    std::fs::write(package_dir.join("src/voice.hv"), "pub struct Voice {}\n").unwrap();
    let sources = collect_hv_files(&[package_dir]);
    let source = sources
        .iter()
        .find(|source| source.file == source_path)
        .unwrap();
    let module = extract_module(&source).unwrap();
    assert_eq!(module.title, "audio");
    assert!(
        module
            .docs
            .as_deref()
            .unwrap()
            .starts_with("# Audio library")
    );
    assert!(module.items[0].docs.is_none());
    let docs = Documentation {
        package: Some("audio".to_string()),
        modules: sources
            .iter()
            .map(|source| extract_module(source).unwrap())
            .collect(),
    };
    let out = temp.path().join("site");
    std::fs::create_dir_all(out.join("audio")).unwrap();
    std::fs::write(out.join("audio/lib.html"), "stale").unwrap();
    render_html(&docs, &out).unwrap();
    let root = std::fs::read_to_string(out.join("audio/index.html")).unwrap();
    assert!(root.contains("<h1>Audio library</h1>"));
    assert!(root.contains("Use <strong>carefully</strong>."));
    assert!(root.contains("id=\"function-play\""));
    assert!(root.contains("&lt;script&gt;bad()&lt;/script&gt;"));
    assert!(!out.join("audio/lib.html").exists());
    let voice = std::fs::read_to_string(out.join("audio/voice.html")).unwrap();
    assert!(voice.contains("href=\"../audio/index.html\""));
    let search = std::fs::read_to_string(out.join("assets/search-index.js")).unwrap();
    assert!(search.contains("audio/index.html#function-play"));

    let md_out = temp.path().join("markdown");
    std::fs::create_dir_all(md_out.join("audio")).unwrap();
    std::fs::write(md_out.join("audio/lib.md"), "stale").unwrap();
    render_markdown(&docs, &md_out).unwrap();
    let root_md = std::fs::read_to_string(md_out.join("audio/index.md")).unwrap();
    assert!(root_md.starts_with("# Audio library"));
    assert!(root_md.contains("## `play`"));
    assert!(!md_out.join("audio/lib.md").exists());
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

    assert_eq!(titles, ["audio/dsp/osc", "audio"]);
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

#[test]
fn standalone_lib_remains_a_regular_module() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("lib.hv");
    std::fs::write(&path, "pub proc greet() void {}\n").unwrap();
    let files = collect_hv_files(&[path]);
    assert_eq!(files[0].title, "lib");
    assert_eq!(files[0].package, None);
    let docs = Documentation {
        package: None,
        modules: vec![extract_module(&files[0]).unwrap()],
    };
    let out = temp.path().join("site");
    render_markdown(&docs, &out).unwrap();
    assert!(out.join("lib.md").exists());
}
