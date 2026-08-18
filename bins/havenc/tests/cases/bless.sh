#!/usr/bin/env bash
# Regenerate the `.out` golden files for every run/ fixture by compiling and
# running it with the current compiler. Use this after an intentional change to
# program output; review the diff before committing. Fixtures with no output
# get no `.out` file (exit code is still checked by the harness).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# here = bins/havenc/tests/cases; the workspace root is four levels up, and its
# shared target/ dir holds the built binary.
root="$(cd "$here/../../../.." && pwd)"

echo "building havenc..."
cargo build --manifest-path "$root/Cargo.toml" -p havenc
bin="$root/target/debug/havenc"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# The compiler embeds no std; build the standalone std package into a discoverable
# artifact and point $HAVEN_STD at it, exactly as the test harness does.
echo "building std.hvmeta..."
export HAVEN_STD="$tmp/std.hvmeta"
"$bin" "$root/stdlib/std/src/lib.hv" --lib --package-name std --prelude std \
    --c-file "$root/stdlib/std/c/rt.c" --c-file "$root/stdlib/std/c/env.c" \
    --c-file "$root/stdlib/std/c/fs.c" --c-file "$root/stdlib/std/c/process.c" \
    --link-lib m -o "$HAVEN_STD"

for f in "$here/run"/*.hv; do
    n="$(basename "$f" .hv)"
    if ! "$bin" "$f" -o "$tmp/$n" >"$tmp/$n.log" 2>&1; then
        echo "SKIP $n (does not compile):"
        sed 's/\x1b\[[0-9;]*m//g' "$tmp/$n.log" | head -3
        continue
    fi
    "$tmp/$n" > "$tmp/$n.out" 2>/dev/null || true
    if [ -s "$tmp/$n.out" ]; then
        cp "$tmp/$n.out" "$here/run/$n.out"
        echo "blessed $n.out"
    else
        rm -f "$here/run/$n.out"
        echo "no output: $n (removed stale .out if any)"
    fi
done
