# haven

haven is a statically typed programming language and compiler, built specifically
for DSP and audio plugin development.

This repository includes:
- `haven`: build system and project manager for haven projects
- `havenc`: the compiler for the haven programming language
- `havendoc`: a documentation generator for haven projects, built for mdBook

> [!NOTE]
> This is very alpha and work in progress, codebase can be messy and bugs may
> arise, please report if you find one.

## Dependencies
- (Developer dependencies)
  - rust/cargo
  - python (for installation script)
- LLVM IR compiler
  - clang
  - opt + llc (untested)

## Installation

Install the binaries using the provided `install.py` script:
```shell
$ cargo build --release
$ python install.py [--debug] [--path <install_path>]
# To uninstall the binaries with the script, you can also run:
$ python install.py --uninstall [--path <install_path>]
```
The script will also install and compile the standard library, which is required
for compiling any haven program.

## Directory Structure
```
bins/
├── haven/          # build system and project manager
├── havenc/         # compiler
└── havendoc/       # documentation generator
crates/
├── haven_back/     # backend-related code (LLVM IR, ABI)
├── haven_common/   # common code & types (AST, Diagnostics, memory layout, etc.)
├── haven_front/    # frontend-related code (lexer, parser, modules)
├── haven_meta/     # .hvmeta metadata artifacts (for dependencies compilation)
└── haven_mid/      # middle-end code (type checking, semantic analysis, etc.)
extensions/
└── vscode/         # VSCode extension for syntax highlighting
std/                # haven standard library package
├── dsp/            # DSP-related modules
├── plug/           # CLAP plugin framework modules
└── std/            # its modules (lib.hv is the entry point / prelude)
```

## License
This project is dual-licensed under the MIT and Apache 2.0 licenses. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.