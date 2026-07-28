//! Rebuild whenever the embedded standard library changes.
//!
//! `module.rs` pulls the whole `std/` tree into the binary with
//! `include_dir!`, which expands at compile time but tells cargo nothing about
//! the files it read. Without this, editing a `.hv` file under `std/` leaves the
//! *previous* copy embedded in `havenc` and the change appears to have no
//! effect - which is a particularly quiet failure now that the prelude carries a
//! lang item (`trait Delete`): a stale embed silently turns ownership off.

fn main() {
    println!("cargo:rerun-if-changed=../../std");
}
