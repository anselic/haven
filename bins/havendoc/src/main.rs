//! `havendoc`: generate mdBook-ready Markdown documentation from `.hv` source.
//!
//! Parses each file for real signatures (via `haven_front`) and pulls `///` doc
//! comments straight from the source, writing one page per module plus a
//! `book.toml` and `SUMMARY.md` so the output is a buildable mdBook. See the
//! `doc` module for the details of that hybrid parse-plus-source-scan approach.

use std::path::PathBuf;

use clap::Parser;

mod doc;

#[derive(Parser, Debug)]
#[command(
    name = "havendoc",
    about = "Generate mdBook-ready Markdown docs from .hv source"
)]
pub struct DocArgs {
    /// Source files or directories to document. Directories are searched
    /// recursively for `.hv` files.
    #[arg(required = true, num_args = 1..)]
    pub inputs: Vec<PathBuf>,

    /// Output directory for the generated book (pages land under `<out>/src`).
    #[arg(short, long, default_value = "docs")]
    pub out: PathBuf,
}

fn main() {
    haven_common::diag::install_ice_hook("havendoc");
    let args = DocArgs::parse();
    match doc::generate(&args) {
        Ok(()) => std::process::exit(0),
        Err(()) => std::process::exit(1),
    }
}
