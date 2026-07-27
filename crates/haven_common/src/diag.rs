use std::ops::Range;

use ariadne::{Color, Label, Report, ReportKind};

use crate::ast::{Error, FileId, Span};

/// Every source file a diagnostic might point into, indexed by [`FileId`].
///
/// Built up by the module loader as it loads, and threaded down so any stage can
/// quote the span's *owning* module rather than the entry file. Spans hold a
/// `FileId` into this table, so looking a source up is an index, not a scan, and
/// the path string is stored once instead of once per token.
#[derive(Default)]
pub struct Files<'a> {
    /// display path per `FileId`, e.g. an absolute path or `std/math`.
    paths: Vec<String>,
    srcs: Vec<&'a str>,
}

impl<'a> Files<'a> {
    pub fn new() -> Self { Self::default() }

    /// Register a file and get the id spans should carry for it.
    pub fn add(&mut self, path: String, src: &'a str) -> FileId {
        let id = FileId(self.paths.len() as u32);
        self.paths.push(path);
        self.srcs.push(src);
        id
    }

    /// Display path for `id`, or `<unknown>` for [`FileId::UNKNOWN`] (and any
    /// other out-of-range id, so a bad span degrades the diagnostic rather than
    /// panicking).
    pub fn path(&self, id: FileId) -> &str {
        self.paths.get(id.0 as usize).map(|s| s.as_str()).unwrap_or("<unknown>")
    }

    pub fn src(&self, id: FileId) -> Option<&'a str> {
        self.srcs.get(id.0 as usize).copied()
    }

    pub fn len(&self) -> usize { self.paths.len() }
    pub fn is_empty(&self) -> bool { self.paths.is_empty() }

    /// The `(path, source)` pairs ariadne needs to build its cache.
    fn ariadne_cache(&self) -> impl ariadne::Cache<String> + '_ {
        ariadne::sources(
            self.paths.iter().cloned().zip(self.srcs.iter().copied())
        )
    }
}

/// AST/chumsky spans are byte offsets, but ariadne's renderer indexes labels by
/// char offset. Convert here so multibyte source still underlines the right span
fn byte_to_char(src: &str, byte: usize) -> usize {
    src[..byte.min(src.len())].chars().count()
}

fn to_span(src: &str, path: &str, span: &Span) -> (String, Range<usize>) {
    let start = byte_to_char(src, span.start);
    let end = byte_to_char(src, span.end.max(span.start));
    (path.to_string(), start..end)
}

/// Render one diagnostic to stderr. `stage` is the header label (e.g.
/// `"Typecheck error"`); `span.file` selects which source in `files` to quote.
pub fn report(stage: &str, msg: &str, span: &Span, files: &Files) {
    let Some(src) = files.src(span.file) else {
        // no source on hand (shouldn't happen) but we don't want to swallow the
        // message
        eprintln!("{} in {}: {}", stage, files.path(span.file), msg);
        return;
    };

    let span = to_span(src, files.path(span.file), span);
    Report::build(ReportKind::Custom(stage, Color::Red), span.clone())
        .with_message(msg)
        .with_label(Label::new(span).with_color(Color::Red).with_message(msg))
        .finish()
        // ariadne indexes each Source by char offset - matches `to_span` above
        .eprint(files.ariadne_cache())
        .ok();
}

/// Convenience for the common `ast::Error` case.
pub fn report_error(stage: &str, err: &Error, files: &Files) {
    report(stage, &err.msg, &err.span, files);
}
