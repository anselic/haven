# haven

haven is a statically typed programming language and compiler, built specifically
for DSP and audio plugin development.

This repository includes:

- `vestry`: build system and project manager for Haven projects
- `havenc`: the compiler for the haven programming language
- `havendoc`: a documentation generator that emits Markdown or static HTML

> [!NOTE]
> This is pre-alpha software. The compiler is not yet stable, and the language is still under active development. Expect breaking changes and bugs.

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
# Use `--debug` if you want to build the binaries in debug mode (default is release mode).
$ python install.py [--debug] [--path <install_path>]

$ havenc --version
# To uninstall the binaries with the script, you can also run:
$ python install.py --uninstall [--path <install_path>]
```
The script will also install and compile the standard library, which is required
for compiling any haven program.

## Directory Structure
```
bins/
├── vestry/         # build system and project manager
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
stdlib/
└── std/            # haven standard library package
```

## License
This project is dual-licensed under the MIT and Apache 2.0 licenses. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.
