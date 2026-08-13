#!/usr/bin/env python3

import os
import sys
import shutil
import platform
import argparse
import subprocess
from pathlib import Path

# The compiler embeds no standard library: it discovers `std.hvmeta` on disk,
# first looking right next to the `havenc` binary. So a working install is the
# binaries *plus* this artifact, built from the standalone `std` package here and
# installed alongside them.
STD_ARTIFACT = "std.hvmeta"

# Build the standard library artifact from the `std` package and copy it to the
# destination directory.
def build_std(source_dir, dest_dir):
    exe_ext = ".exe" if platform.system() == "Windows" else ""
    # Resolve to absolute: we run with `cwd` set to the std package, and `haven`
    # locates `havenc` relative to its own (absolute) path, so a relative
    # `target/release/haven` would otherwise be looked up under `cwd`.
    haven = (source_dir / f"haven{exe_ext}").resolve()
    havenc = (source_dir / f"havenc{exe_ext}").resolve()
    if not haven.exists() or not havenc.exists():
        print(f"Warning: {haven} or its sibling {havenc.name} not found; cannot build "
              f"{STD_ARTIFACT}. Programs will need $HAVEN_STD or --dep std.")
        return
    std_dir = Path("std").resolve()
    result = subprocess.run([str(haven), "build"], cwd=str(std_dir),
                            encoding="utf-8", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to build {STD_ARTIFACT}:\n{result.stderr}")
        return
    else:
        print(result.stdout, end="")
    artifact = std_dir / ".haven" / "target" / STD_ARTIFACT
    if not artifact.exists():
        print(f"Failed to build {STD_ARTIFACT}: `haven build` produced no artifact "
              f"at {artifact}.")
        return
    out = dest_dir / STD_ARTIFACT
    try:
        shutil.copy2(artifact, out)
        print(f"Built and installed {STD_ARTIFACT} to {out}")
    except Exception as e:
        print(f"Failed to install {STD_ARTIFACT}: {e}")

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Manage haven binaries.")
    parser.add_argument("--debug", action="store_true", help="Copy debug builds instead of release.")
    parser.add_argument("--path", type=str, help="Specific destination directory (e.g., ~/.local/bin).")
    parser.add_argument("--uninstall", action="store_true", help="Delete the binaries instead of installing them.")
    args = parser.parse_args()

    is_windows = platform.system() == "Windows"
    exe_ext = ".exe" if is_windows else ""
    binaries = [
        f"haven{exe_ext}",
        f"havenc{exe_ext}",
        f"havendoc{exe_ext}",
    ]

    # Handle uninstall
    if args.uninstall:
        dirs_to_check = []
        if args.path:
            dest_dir = Path(args.path).expanduser().resolve()
            if not dest_dir.exists():
                print(f"Error: The path '{dest_dir}' does not exist.")
                sys.exit(1)
            dirs_to_check = [dest_dir]
            print(f"Checking for binaries in: {dest_dir}")
        else:
            dirs_to_check = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
            print("No path provided. Scanning entire PATH for binaries to remove...")

        removed_any = False
        for d in dirs_to_check:
            if not d.is_dir(): continue
            for bin_name in binaries + [STD_ARTIFACT]:
                target_file = d / bin_name
                if target_file.exists():
                    try:
                        target_file.unlink()
                        print(f"Deleted {target_file}")
                        removed_any = True
                    except Exception as e:
                        print(f"Failed to delete {target_file}: {e}")

        if not removed_any:
            print("No binaries found to remove.")

    # Handle installation
    else:
        build_type = "debug" if args.debug else "release"
        source_dir = Path("target") / build_type

        # Build the binaries
        print(f"Building {build_type} binaries...")
        result = subprocess.run(
            ["cargo", "build",
            "--bin", "haven",
            "--bin", "havenc",
            "--bin", "havendoc"]
            + (["--release"] if not args.debug else []),
            encoding="utf-8", text=True)
        if result.returncode != 0:
            print(f"Failed to build binaries:\n{result.stderr}")
            sys.exit(1)

        dest_dir = None

        if args.path:
            dest_dir = Path(args.path).expanduser().resolve()
            if not dest_dir.exists():
                print(f"Error: The path '{dest_dir}' does not exist. Create it first.")
                sys.exit(1)
            if not os.access(dest_dir, os.W_OK):
                print(f"Error: You do not have write permissions for '{dest_dir}'.")
                sys.exit(1)
        else:
            path_dirs = os.environ.get("PATH", "").split(os.pathsep)
            for p in path_dirs:
                if not p: continue
                p_path = Path(p)
                if p_path.is_dir() and os.access(p_path, os.W_OK):
                    if "Windows" not in p_path.parts and "usr" not in p_path.parts:
                        dest_dir = p_path
                        break

            if not dest_dir:
                for p in path_dirs:
                    if not p: continue
                    p_path = Path(p)
                    if p_path.is_dir() and os.access(p_path, os.W_OK):
                        dest_dir = p_path
                        break

        if not dest_dir:
            print("Error: Could not automatically find a writable directory in your PATH.")
            print("Specify one manually using the --path argument.")
            sys.exit(1)

        print(f"Targeting directory: {dest_dir}")

        for bin_name in binaries:
            src_file = source_dir / bin_name
            if src_file.exists():
                dest_file = dest_dir / bin_name
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"Copied {bin_name} to {dest_file}")
                except Exception as e:
                    print(f"Failed to copy {bin_name}: {e}")
            else:
                print(f"Warning: {src_file} not found in {source_dir}. Skipping.")

        # the compiler carries no std of its own; build and install it alongside.
        build_std(source_dir, dest_dir)

if __name__ == "__main__":
    main()