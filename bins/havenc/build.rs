fn main() {
    // crt/ lives at the workspace root, two levels up from this bin crate.
    println!("cargo:rerun-if-changed=../../crt/rt.c");

    cc::Build::new()
        .file("../../crt/rt.c")
        .compiler("clang")
        .opt_level(3)
        .compile("runtime");
}