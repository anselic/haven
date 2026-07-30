use std::fs;

fn main() {
    // crt/ lives at the workspace root, two levels up from this bin crate.
    println!("cargo:rerun-if-changed=../../crt");

    let mut build = cc::Build::new();

    if let Ok(entries) = fs::read_dir("../../crt") {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "c") {
                    build.file(path);
                }
            }
        }
    } else {
        panic!("Failed to read crt/ directory");
    }

    build.compiler("clang")
        .opt_level(3)
        .compile("runtime");
}