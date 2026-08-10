# Handoff — prelude & lang items as package-supplied things

Written 2026-08-10. Read this, then `git log -1` and `cargo test --workspace` to confirm
you're where this says you are.

## 0. Get the work onto the laptop first

`HEAD` is `a8f8b2b "attribute validation, prelude and lang support"`, which contains
steps 1–3 below. **Everything after that is uncommitted** — 22 modified files, 3 new,
1 deleted. Commit and push before switching machines, or you'll be reading about work
that isn't there.

```
bins/havenc/src/{args,main}.rs          --prelude flag + PreludeSource wiring
crates/haven_front/src/module.rs        the bulk: PreludeSource, enqueue_dep, self/
bins/havenc/tests/prelude_dep.rs        NEW — 8 tests for --prelude
bins/havenc/tests/dep_consume.rs        +2 tests for `import self/`
bins/havenc/tests/cases/fail/test_import_self_bare.hv            NEW
bins/havenc/tests/cases/fail/test_lang_item_not_prelude_pkg.hv   NEW (replaces
                                        test_lang_item_not_std.hv, now deleted)
stdpkg/src/**                           17 files: the self-contained rewrite
```

The untracked `test/` directory at the repo root is not mine — leave it alone.

Verify: `cargo test --workspace` → **253 passing, 186 fixtures**, nothing ignored.

## 1. What this arc was about

The compiler used to identify the prelude and the `Delete` lang item by *where they
were filed* — a module keyed `std/prelude`, a trait named `Delete` inside it. That is a
rule about the embedded stdlib's directory layout pretending to be a rule about the
language, and it has nothing to say once the stdlib is an ordinary package. Worse, the
lang-item lookup failed **silently**: `own.rs` treats a missing `Delete` as "nothing in
this program owns anything", so a mismatched stdlib compiled to a program with no
destructors — a miscompile, not a build failure.

Four steps, all landed:

1. **`trait` takes attributes + an attribute table.** `ast::KNOWN_ATTRIBUTES` +
   `check_attribute(attr, AttrTarget)`. Unknown / misplaced / misshapen attributes are
   errors; before, `@totally_made_up(xyz)` compiled and did nothing, which is also what
   a typo'd `@export` did (a symbol that silently fails to link).
2. **`@lang(delete)`** marks the lang item. `defs::LANG_ITEMS` + `ast::LANG_ATTR` are
   the single source of truth; the table validates the written name, the loader fills
   `Defs::lang()`. A prelude that declares no `Delete` is now a hard error.
3. **`@!prelude`** marks the prelude module. `@!name` is a *module* attribute — the `!`
   distinguishes "about this file" from "about the item below", which has to be
   decidable before the file's items are known. `PRELUDE_KEY` survives but means only
   "which embedded module gets loaded to serve as the prelude".
4. **`--prelude <package>`.** `PreludeSource::{None, Std, Package(&str)}` replaced an
   `inject_prelude: bool`. The nominated package is enqueued eagerly (a program is not
   obliged to import the package its prelude comes from). It may also name **the
   package being compiled** — that case is load-bearing, not a convenience: without it
   a prelude package can't be built at all, since during its own build its claims are
   made by a package that is nobody's prelude.

Then: **`stdpkg` rewritten** to be self-contained (zero `import std/...` left), which
required a fifth thing — **`import self/<module>`**, a reserved segment anchoring at
the package root, because an ordinary import only reaches *downward* from the importing
module's own directory.

## 2. Invariants — do not quietly undo these

- **Marking beats locating.** Nothing may identify a prelude or a lang item by module
  key, file name, or item name. If you find yourself writing `key == "..."`, stop.
- **`@lang` outside the prelude package is an error; `@!prelude` outside it is inert.**
  This asymmetry is deliberate and was argued for twice. Providing a prelude is an
  ordinary thing for a package to do, and that same package is just a library to
  whoever depends on it — its mark must not follow it in and displace the consumer's
  prelude. A lang item has no such reading: two `Delete` traits means the ownership
  pass can insert calls for only one, silently treating the other package's owners as
  `Copy`. Ignoring *that* mark is the miscompile, so it errors.
- **`Origin` vs `Module.prelude_pkg` are different questions.** `Origin` (Own/Std/Dep)
  records where source came from and is fixed for the build. `prelude_pkg` records
  whether a module may make the two claims above, and depends on what the build was
  *asked* for. Don't collapse them.
- **A library's root must reach every module it ships.** A `.hvmeta` carries the
  modules that were *loaded*, and loading starts at the root — an unreachable module is
  absent for consumers, not merely unused. `stdpkg/src/lib.hv` has a private-import
  block for exactly this (7 of 17 modules shipped before it existed).
- **A duplicated module is silent.** `Defs::add_module` disambiguates a slug collision
  with an fnv1a suffix, so double-loading a module produces `std.math` *and*
  `std.math_ee88998f` and links fine. When checking "is this loaded once", grep the
  emitted `.ll` for the mangled slug; don't assume the linker would have told you.

## 3. Remaining gaps, roughly in dependency order

### (a) `import std/...` always means the embedded stdlib — THE blocker

In `load_and_merge`'s import loop the dep branch is filtered
`*f != "std" && *f != SELF_SEG && deps.contains_key(*f)`, and `resolve_target` sends any
`std/...` straight to the embedded tree. Consequence, measured: a package bound as
`--dep std=stdpkg.hvmeta` **can only ever be the prelude** — `import std/vec` in a
consumer silently reaches the embedded std instead. Bind the same artifact as `hstd`
and everything works (`import hstd/dsp/osc { Phasor }` builds and runs), so the gap is
purely this guard.

The guard exists to stop std being loaded twice (`dep_consume.rs::std_is_shared_not_doubled`
pins that). Replacing it means deciding what `std` *is* when a package supplies it —
which is the "treat std as a normal dependency" work that was explicitly deferred. Do
not just delete the guard; the double-load trap is real and the fnv1a suffix will hide
it from you.

### (b) `rt_abort` has no declaration owner

`abort()` lowers to `Callee::Direct("rt_abort")` (`haven_mid/src/mil/expr.rs:116`) and
the backend emits **no** declare for it. It has always worked because `std/prelude.hv`
happens to declare `extern rt_abort`. Any replacement prelude that omits it fails at
LLVM with `use of undefined value '@rt_abort'` — I hit this building `stdpkg` and fixed
it in `stdpkg/src/lib.hv`, but that's papering over it. The call is compiler-generated,
so codegen should emit the declare itself.

### (c) `[[c]]` and native artifacts

`stdpkg/haven.toml` declares C files (`env.c`, `fs.c`, `process.c`, `rt.c`).
`bins/haven/src/config.rs`'s `Manifest` has only `project` + `dependencies`, and serde
**silently drops unknown fields** — the same trap `[dependencies]` had before. Separately,
`.hvmeta` is a source-blob format that cannot carry C or objects at all. (Both of these
are from earlier in this work — re-verify before acting.)

A principled split was sketched: `rt.c` is compiler-owned (the runtime the compiler
lowers to), while `env/fs/process.c` are std's and need real native-artifact support —
a v2 hybrid `.hvmeta`.

### (d) Smaller

- `bins/haven`'s `dependencies()` is v1: direct deps only, no versions, no transitive
  resolution (rejected deliberately by manifest policy).
- `.clap` bundling in `build.hv` — last item of the CLAP roadmap.

## 4. Test map

| target | what it covers |
|---|---|
| `cargo test -p havenc --test harness` | 186 `.hv` fixtures under `tests/cases/{run,fail}` |
| `--test prelude_dep` | `--prelude` end to end over a `mini` stand-in stdlib |
| `--test dep_consume` | `--dep`, package-anchored symbols, `import self/` |
| `--test lib_meta` | `.hvmeta` producer, incl. deps-excluded-from-own-artifact |

Fixture conventions: a `run` case needs a `.out` golden in **CRLF, no BOM** (PowerShell's
`Set-Content -Encoding utf8` writes a BOM and will fail the test — use
`[System.IO.File]::WriteAllText` with `UTF8Encoding $false`). A `fail` case carries a
`//@ error: <substring>` marker.

Manual check for the paths no fixture can reach (the embedded std always satisfies them):
temporarily edit `std/prelude.hv`, rebuild, run any program, then revert and confirm
with `git diff std/prelude.hv`.

## 5. Reproducing the stdpkg loop

```
cargo run -q -p havenc -- stdpkg/src/lib.hv --package-name hstd --lib --prelude hstd -o hstd
cargo run -q -p havenc -- main.hv --package-name app --dep hstd=hstd.hvmeta --prelude hstd -o app --emit-ir
```

where `main.hv` can be `import hstd/dsp/osc { Phasor }` + a `main`. Grep the `.ll` for
`math\$` — you should see `hstd.math$*` and nothing else. Substituting `std` for `hstd`
also works but only through the prelude, per gap (a).
