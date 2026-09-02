//! The internal-compiler-error report: what a user sees when the compiler
//! panics. `--internal-panic` (hidden) trips one on purpose right after
//! startup, so these tests need no real bug and no real input.
//!
//! The contract: exit 101 (Rust's panic exit, so a parent can tell a crash from
//! a diagnostic), a report that says it is a compiler bug and where to file it,
//! and - under `--message-format json` - that report as one NDJSON line rather
//! than free text in the stream.

use std::process::Command;

const HAVENC: &str = env!("CARGO_BIN_EXE_havenc");

fn crash(extra: &[&str]) -> std::process::Output {
    Command::new(HAVENC)
        .arg("does-not-need-to-exist.hv")
        .arg("--no-prelude")
        .arg("--internal-panic")
        .args(extra)
        .env_remove("RUST_BACKTRACE")
        .output()
        .expect("failed to spawn havenc")
}

fn stderr(o: &std::process::Output) -> String {
    String::from_utf8_lossy(&o.stderr).replace("\r\n", "\n")
}

#[test]
fn a_panic_exits_101_with_a_report_not_a_rust_backtrace() {
    let o = crash(&[]);
    assert_eq!(o.status.code(), Some(101), "a crash keeps Rust's panic exit code");

    let err = stderr(&o);
    assert!(err.starts_with("internal compiler error: deliberate internal panic"),
        "report should lead with the panic message; got:\n{err}");
    assert!(err.contains("havenc "), "the tool and version, for the bug report:\n{err}");
    assert!(err.contains("invoked as: ") && err.contains("--internal-panic"),
        "the invocation, for the bug report:\n{err}");
    assert!(err.contains("bug in the compiler"), "say whose fault it is:\n{err}");
    assert!(err.contains("https://github.com/anselic/haven/issues"), "and where to report it:\n{err}");
    assert!(err.contains("RUST_BACKTRACE=1"), "and how to get a backtrace:\n{err}");
    assert!(!err.contains("thread 'main' panicked"),
        "Rust's own panic line must not appear:\n{err}");
}

#[test]
fn rust_backtrace_puts_a_backtrace_under_the_report() {
    let o = Command::new(HAVENC)
        .args(["x.hv", "--no-prelude", "--internal-panic"])
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("failed to spawn havenc");
    let err = stderr(&o);
    assert!(err.contains("backtrace:"), "got:\n{err}");
    assert!(!err.contains("rerun with RUST_BACKTRACE=1"),
        "the hint is pointless once a backtrace is printed:\n{err}");
}

#[test]
fn under_json_the_report_is_one_ndjson_line() {
    let o = crash(&["--message-format", "json"]);
    assert_eq!(o.status.code(), Some(101));

    let err = stderr(&o);
    let lines: Vec<&str> = err.lines().collect();
    assert_eq!(lines.len(), 1, "one line, so a stream consumer keeps parsing; got:\n{err}");
    let line = lines[0];
    assert!(line.starts_with("{\"severity\":\"error\",\"stage\":\"internal compiler error\""),
        "same shape as every other diagnostic; got:\n{line}");
    assert!(line.contains("\"message\":\"deliberate internal panic"));
    assert!(line.contains("\"file\":null,\"span\":null"));
    assert!(line.contains("https://github.com/anselic/haven/issues"));
    assert!(line.ends_with('}'));
}
