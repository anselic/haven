use std::io::IsTerminal;
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use ariadne::{Color, Config, Label, Report, ReportKind};

use crate::ast::{Error, FileId, Span};

/// Whether diagnostics may use ANSI colour. Decided once, from stderr - which is
/// where every diagnostic goes.
///
/// ariadne colours unconditionally by default, so a captured diagnostic arrived
/// with escape codes baked into it: `install.py` reading our stderr got literal
/// `\x1b[31m` in the text it printed, and CI logs get the same. A subprocess
/// inherits its parent's stderr, so a `havenc` driven by `haven` sees whatever
/// `haven` was given and this stays right either way.
///
/// `NO_COLOR` wins over the terminal check, per <https://no-color.org>: set and
/// non-empty disables, whatever the value.
///
/// Turning the answer *off* takes two switches, because ariadne's own config only
/// covers part of its output. [`Config::with_color`] governs the frame, labels and
/// margins, but the header's colour comes from the `ReportKind`, and the `Custom`
/// kind we use to print a stage name returns its colour unconditionally where the
/// built-in kinds filter theirs through the config - so `with_color(false)` alone
/// still emitted a red `Lang item error:`. Disabling yansi (ariadne's backend, and
/// the same instance thanks to the unified dependency) covers everything ariadne
/// paints, whichever route it took.
fn color_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = decide_color(std::env::var_os("NO_COLOR"), std::io::stderr().is_terminal());
        if !on {
            yansi::disable();
        }
        on
    })
}

/// The rule itself, split out from the environment it reads so both answers can
/// be tested - a test process has no terminal, so the enabled case is otherwise
/// unreachable.
fn decide_color(no_color: Option<std::ffi::OsString>, stderr_is_tty: bool) -> bool {
    match no_color {
        Some(v) if !v.is_empty() => false,
        _ => stderr_is_tty,
    }
}

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

/// How diagnostics are rendered. `Human` is the ariadne pretty-printer for a
/// terminal; `Json` emits one machine-readable object per line (NDJSON) so an
/// LSP server or the `haven` build orchestrator can stream-parse them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Human,
    Json,
}

// Chosen once at startup (see `set_format`) and read on every `report`. An atomic
// keeps `report`/`report_error`/`report_plain` free of any extra threading, so
// every existing call site emits in the selected format for free.
static FORMAT: AtomicU8 = AtomicU8::new(Format::Human as u8);

/// Select the diagnostic output format. Call once, before compilation starts.
pub fn set_format(fmt: Format) {
    FORMAT.store(fmt as u8, Ordering::Relaxed);
}

/// The format selected by [`set_format`], defaulting to [`Format::Human`].
pub fn format() -> Format {
    match FORMAT.load(Ordering::Relaxed) {
        x if x == Format::Json as u8 => Format::Json,
        _ => Format::Human,
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

/// 0-based `(line, character)` for a byte offset, in the units LSP expects:
/// lines split on `\n`, columns counted in UTF-16 code units so an editor can
/// map the position without re-scanning the source itself.
fn line_col(src: &str, byte: usize) -> (usize, usize) {
    let byte = byte.min(src.len());
    let mut line = 0usize;
    let mut col = 0usize; // utf-16 units since the last newline
    for (i, ch) in src.char_indices() {
        if i >= byte { break; }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += ch.len_utf16();
        }
    }
    (line, col)
}

/// Append `s` to `out` as a JSON string literal, quotes included.
fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Emit one diagnostic as a single NDJSON line on stderr. `file` and the
/// resolved `span` are optional so spanless errors (e.g. "no main function")
/// still land in the same stream with `null` fields.
fn emit_json(stage: &str, msg: &str, file: Option<&str>, span: Option<(&str, &Span)>) {
    let mut out = String::new();
    out.push_str("{\"severity\":\"error\",\"stage\":");
    push_json_str(&mut out, stage);
    out.push_str(",\"message\":");
    push_json_str(&mut out, msg);

    out.push_str(",\"file\":");
    match file {
        Some(f) => push_json_str(&mut out, f),
        None => out.push_str("null"),
    }

    out.push_str(",\"span\":");
    match span {
        Some((src, sp)) => {
            let end = sp.end.max(sp.start);
            let (sl, sc) = line_col(src, sp.start);
            let (el, ec) = line_col(src, end);
            out.push_str(&format!(
                "{{\"byte_start\":{},\"byte_end\":{},\
                 \"start\":{{\"line\":{},\"character\":{}}},\
                 \"end\":{{\"line\":{},\"character\":{}}}}}",
                sp.start, end, sl, sc, el, ec,
            ));
        }
        None => out.push_str("null"),
    }
    out.push('}');

    eprintln!("{}", out);
}

/// Render one diagnostic. `stage` is the header label (e.g. `"Typecheck error"`);
/// `span.file` selects which source in `files` to quote. Honors the format set by
/// [`set_format`]: pretty ariadne output, or one NDJSON line, both on stderr.
pub fn report(stage: &str, msg: &str, span: &Span, files: &Files) {
    let path = files.path(span.file);
    let Some(src) = files.src(span.file) else {
        // no source on hand (shouldn't happen) but we don't want to swallow the
        // message
        match format() {
            Format::Json => emit_json(stage, msg, Some(path), None),
            Format::Human => eprintln!("{} in {}: {}", stage, path, msg),
        }
        return;
    };

    if let Format::Json = format() {
        emit_json(stage, msg, Some(path), Some((src, span)));
        return;
    }

    let span = to_span(src, path, span);
    Report::build(ReportKind::Custom(stage, Color::Red), span.clone())
        .with_config(Config::default().with_color(color_enabled()))
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

/// Report an error with no source location (a driver-level failure like a
/// missing entry file or absent `main`). In human mode this is a plain stderr
/// line; in JSON mode it joins the NDJSON stream with `null` file/span so a
/// consumer never has to parse free-form text to notice the build failed.
pub fn report_plain(stage: &str, msg: &str) {
    match format() {
        Format::Json => emit_json(stage, msg, None, None),
        Format::Human => eprintln!("{}: {}", stage, msg),
    }
}

#[cfg(test)]
mod tests {
    use super::decide_color;
    use std::ffi::OsString;

    #[test]
    fn no_color_beats_a_terminal() {
        // https://no-color.org: present and non-empty disables, whatever the value
        assert!(!decide_color(Some(OsString::from("1")), true));
        assert!(!decide_color(Some(OsString::from("0")), true));
        assert!(!decide_color(Some(OsString::from("anything")), true));
    }

    #[test]
    fn empty_no_color_does_not_count() {
        // an empty value is "unset" per the spec, so the terminal decides
        assert!(decide_color(Some(OsString::new()), true));
        assert!(!decide_color(Some(OsString::new()), false));
    }

    #[test]
    fn unset_defers_to_the_terminal() {
        assert!(decide_color(None, true));
        assert!(!decide_color(None, false));
    }
}
