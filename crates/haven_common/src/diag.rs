use std::io::IsTerminal;
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use ariadne::{Color, Config, Label, Report, ReportKind};

use crate::ast::{Error, FileId, Span};

/// Whether diagnostics may use ANSI colour. Decided once, from stderr, where
/// every diagnostic goes.
///
/// ariadne colours by default, so a captured diagnostic carried escape codes:
/// `install.py` reading our stderr got literal `\x1b[31m`, and CI logs too. A
/// subprocess inherits its parent's stderr, so a `havenc` under `haven` stays
/// right either way.
///
/// `NO_COLOR` beats the terminal check, per <https://no-color.org>: set and
/// non-empty disables it, whatever the value.
///
/// Turning colour off takes two switches: ariadne's config covers only part of
/// its output. [`Config::with_color`] governs the frame, labels, and margins,
/// but the header's colour comes from the `ReportKind`. Our `Custom` kind
/// returns its colour unconditionally, so `with_color(false)` alone still drew
/// a red `Lang item error:`. Disabling yansi (ariadne's backend) covers
/// everything it paints.
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

/// The rule itself, split from the environment it reads so both answers are
/// testable: a test process has no terminal, so the on-case is otherwise
/// unreachable.
fn decide_color(no_color: Option<std::ffi::OsString>, stderr_is_tty: bool) -> bool {
    match no_color {
        Some(v) if !v.is_empty() => false,
        _ => stderr_is_tty,
    }
}

/// Every source file a diagnostic might point into, indexed by [`FileId`].
///
/// The module loader fills it as it loads, so any stage can quote the span's
/// owning module, not the entry file. A span holds a `FileId`, so a lookup is
/// an index, not a scan, and each path is stored once, not once per token.
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

// chosen once at startup (`set_format`), read on every `report`. an atomic keeps
// the report fns free of extra threading, so every call site gets the format free.
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

/// Longest `msg` still worth repeating on the underline when a diagnostic has
/// no label of its own.
///
/// ariadne draws label text on one line beside the source, indented to the
/// span's column, and never wraps it, so a long one runs off the terminal and
/// the box's frame breaks. The header above says the same text with no such
/// limit, so a bare underline loses nothing.
const LABEL_LINE_BUDGET: usize = 100;

/// Columns the `NNN │ ` gutter takes before any label text. Approximate: it
/// grows with the line-number width, which is all [`fits_inline`] needs to
/// choose between repeating the header here and not.
const LABEL_GUTTER: usize = 6;

/// Whether `msg` fits on the underline for the span `start..end` (char offsets
/// into `src`), without running past [`LABEL_LINE_BUDGET`].
///
/// The span's position decides this, not just the message length. ariadne runs
/// the elbow rightwards past the span's far edge before the text starts, so the
/// text begins near the span's end column. The same message fits under a short
/// name at the margin but overflows under a long expression nested deep.
fn fits_inline(src: &str, start: usize, end: usize, msg: &str) -> bool {
    let byte = src.char_indices().nth(end.max(start))
        .map_or(src.len(), |(i, _)| i);
    let col = src[..byte].rsplit('\n').next().map_or(0, |l| l.chars().count());
    col + msg.chars().count() + LABEL_GUTTER <= LABEL_LINE_BUDGET
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

/// Append a span as a JSON object: byte offsets plus LSP-style line/character.
fn push_json_span(out: &mut String, src: &str, sp: &Span) {
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

/// Emit one diagnostic as a single NDJSON line on stderr. `file` and the
/// resolved `span` are optional so spanless errors (e.g. "no main function")
/// still land in the same stream with `null` fields.
///
/// `labels` and `note` are always present as keys (`[]` / `null` when absent),
/// so a consumer never branches on whether a producer filled them in. A label
/// whose file has no source is dropped, not emitted spanless: it would tell an
/// editor nothing.
fn emit_json(
    stage: &str,
    msg: &str,
    file: Option<&str>,
    span: Option<(&str, &Span)>,
    labels: &[(Span, String)],
    note: Option<&str>,
    files: &Files,
) {
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
        Some((src, sp)) => push_json_span(&mut out, src, sp),
        None => out.push_str("null"),
    }

    out.push_str(",\"labels\":[");
    let mut first = true;
    for (sp, text) in labels {
        let Some(src) = files.src(sp.file) else { continue };
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str("{\"file\":");
        push_json_str(&mut out, files.path(sp.file));
        out.push_str(",\"message\":");
        push_json_str(&mut out, text);
        out.push_str(",\"span\":");
        push_json_span(&mut out, src, sp);
        out.push('}');
    }
    out.push(']');

    out.push_str(",\"note\":");
    match note {
        Some(n) => push_json_str(&mut out, n),
        None => out.push_str("null"),
    }
    out.push('}');

    eprintln!("{}", out);
}

/// Render one diagnostic. `stage` is the header label (e.g. `"Typecheck error"`);
/// `span.file` selects which source in `files` to quote. Honors the format set by
/// [`set_format`]: pretty ariadne output, or one NDJSON line, both on stderr.
pub fn report(stage: &str, msg: &str, span: &Span, files: &Files) {
    report_full(stage, msg, span, &[], None, files);
}

/// The full diagnostic: headline, primary span, extra labelled spans, note.
///
/// The labels keep a diagnostic inside its box. ariadne draws underline text on
/// one line beside the snippet, so a long `msg` used as the label wraps and the
/// frame breaks. With `labels` filled in, `msg` stays the short header, each
/// span carries only its own phrase, and `note` takes anything longer, printed
/// below where it can wrap.
///
/// With no labels this falls back to one underline repeating `msg`, so any call
/// site not yet split up renders as before.
fn report_full(
    stage: &str,
    msg: &str,
    span: &Span,
    labels: &[(Span, String)],
    note: Option<&str>,
    files: &Files,
) {
    let path = files.path(span.file);
    let Some(src) = files.src(span.file) else {
        // no source on hand (shouldn't happen); don't swallow the message
        match format() {
            Format::Json => emit_json(stage, msg, Some(path), None, labels, note, files),
            Format::Human => {
                eprintln!("{} in {}: {}", stage, path, msg);
                for (_, text) in labels {
                    eprintln!("  {}", text);
                }
                if let Some(n) = note {
                    eprintln!("  note: {}", n);
                }
            }
        }
        return;
    };

    if let Format::Json = format() {
        emit_json(stage, msg, Some(path), Some((src, span)), labels, note, files);
        return;
    }

    let primary = to_span(src, path, span);
    let mut report = Report::build(ReportKind::Custom(stage, Color::Red), primary.clone())
        .with_config(Config::default().with_color(color_enabled()))
        .with_message(msg);

    if labels.is_empty() {
        // no label of its own, so `msg` doubles as the underline text - fine
        // while short. past that it wraps and breaks the frame, and it is
        // redundant: the header above already says it, on a line where length
        // costs nothing. so a long one underlines bare.
        let inline = fits_inline(src, primary.1.start, primary.1.end, msg);
        let label = Label::new(primary).with_color(Color::Red);
        report = report.with_label(if inline { label.with_message(msg) } else { label });
    } else {
        // the label on the error's own span is primary: red, drawn with
        // priority. anything elsewhere is context, so it reads as secondary. a
        // label into a file we hold no source for is dropped: ariadne fails the
        // whole render on a cache miss, losing the diagnostic.
        //
        // sorted by position, and `order` follows, because ariadne opens a new
        // source group whenever a label's line goes backwards. declaring the use
        // before the move - the natural "used here / moved here" - otherwise
        // split one snippet into two boxes for the same file. source order keeps them one.
        let mut sorted: Vec<(String, Range<usize>, bool, &str)> = labels.iter()
            .filter_map(|(sp, text)| {
                let lsrc = files.src(sp.file)?;
                let same = sp.file == span.file && sp.start == span.start && sp.end == span.end;
                let (path, range) = to_span(lsrc, files.path(sp.file), sp);
                Some((path, range, same, text.as_str()))
            })
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.start.cmp(&b.1.start)));

        for (i, (path, range, same, text)) in sorted.into_iter().enumerate() {
            report = report.with_label(
                Label::new((path, range))
                    .with_color(if same { Color::Red } else { Color::Yellow })
                    .with_order(i as i32)
                    .with_priority(if same { 1 } else { 0 })
                    .with_message(text));
        }
        // an explicit label is kept whatever its width: unlike the fallback it
        // is not a copy of the header, so dropping it would lose the only place
        // that detail is written.
    }

    if let Some(n) = note {
        report = report.with_note(n);
    }

    report.finish()
        // ariadne indexes each Source by char offset, matching `to_span` above
        .eprint(files.ariadne_cache())
        .ok();
}

/// Convenience for the common `ast::Error` case.
pub fn report_error(stage: &str, err: &Error, files: &Files) {
    report_full(stage, &err.msg, &err.span, &err.labels, err.note.as_deref(), files);
}

/// Report an error with no source location (a driver failure like a missing
/// entry file or absent `main`). Human mode prints a plain stderr line; JSON
/// mode joins the NDJSON stream with `null` file/span, so a consumer need not
/// parse free-form text to see the build failed.
pub fn report_plain(stage: &str, msg: &str) {
    match format() {
        Format::Json => emit_json(stage, msg, None, None, &[], None, &Files::new()),
        Format::Human => eprintln!("{}: {}", stage, msg),
    }
}

#[cfg(test)]
mod tests {
    use super::{decide_color, fits_inline, LABEL_LINE_BUDGET};
    use std::ffi::OsString;

    #[test]
    fn a_short_message_stays_on_the_underline() {
        let src = "let x = 1;\n";
        assert!(fits_inline(src, 4, 5, "not a number"));
    }

    #[test]
    fn a_long_message_leaves_the_underline_bare() {
        let src = "let x = 1;\n";
        assert!(!fits_inline(src, 4, 5, &"x".repeat(LABEL_LINE_BUDGET)));
    }

    #[test]
    fn the_span_column_counts_against_the_budget() {
        // the same message under the same-width span, once at the left margin and
        // once indented far in: only the indented one runs out of room.
        let msg = &"x".repeat(LABEL_LINE_BUDGET / 2);
        let flush = "ab\n";
        let deep = format!("{}ab\n", " ".repeat(LABEL_LINE_BUDGET / 2));
        assert!(fits_inline(flush, 0, 2, msg));
        assert!(!fits_inline(&deep, LABEL_LINE_BUDGET / 2, LABEL_LINE_BUDGET / 2 + 2, msg));
    }

    #[test]
    fn the_column_is_measured_from_the_spans_own_line() {
        // ariadne indents the label to the span, so what matters is the column on
        // the line the span ends on - not the offset into the whole file.
        let src = "a very long first line that says nothing at all about the span\nlet x = 1;\n";
        assert!(fits_inline(src, 67, 68, "not a number"));
    }

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
