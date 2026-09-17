//! `havendoc`: generate documentation from `.hv` source.
//!
//! Parses each file for real signatures (via `haven_front`) and pulls `///` doc
//! comments straight from the source, writing one page per module. See the `doc`
//! module for the details of that hybrid parse-plus-source-scan approach.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

mod doc;

#[derive(Parser, Debug)]
#[command(
    name = "havendoc",
    about = "Generate documentation from .hv source",
    version,
)]
pub struct DocArgs {
    /// Source files or directories to document. Vestry package directories use
    /// their manifest name and `src/` root; other directories are searched
    /// recursively for `.hv` files.
    #[arg(required = true, num_args = 1..)]
    pub inputs: Vec<PathBuf>,

    /// Output directory for the generated documentation files.
    #[arg(short, long, default_value = "docs")]
    pub out: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    pub format: OutputFormat,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputFormat {
    Markdown,
    Html,
}

fn main() {
    haven_common::diag::install_ice_hook("havendoc");
    let args = DocArgs::parse();
    match doc::generate(&args) {
        Ok(()) => std::process::exit(0),
        Err(()) => std::process::exit(1),
    }
}
