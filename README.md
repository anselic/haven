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
- LLVM IR compiler
  - clang
  - opt + llc (untested)

## Installation

Install the binaries using the provided `install.py` script:
```shell
$ cargo build --release
$ python install.py [--debug] [--path <install_path>]
# To remove the installed binaries with the script, you can also run:
$ python install.py --remove [--path <install_path>]
```
The script will also install and compile the standard library, which is required
for compiling any haven program.

## Usage
```shell
# compile to an executable
$ havenc program.hv
$ ./output

# or, compile to a library
$ havenc lib.hv --shared
$ clang host.c output.lib -o output

# use the help flag for more info
$ havenc -h
```

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
├── src/            # its modules (lib.hv is the entry point / prelude)
└── c/              # C runtime
```

The compiler embeds no standard library. `std` is built into a `std.hvmeta`
artifact (see `install.py`) that `havenc` discovers on disk - beside its own
binary, or via `$HAVEN_STD`.

## License
This project is dual-licensed under the MIT and Apache 2.0 licenses. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.