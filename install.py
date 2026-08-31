#!/usr/bin/env python3

import os
import sys
import shutil
import platform
import argparse
import subprocess
from pathlib import Path

STD_ARTIFACT = "std.hvmeta"

DEFAULT_DEST = Path.home() / ".haven"

def build_lib(source_dir, dest_dir, pkg, artifact, env=None):
    exe_ext = ".exe" if platform.system() == "Windows" else ""
    haven = (source_dir / f"haven{exe_ext}").resolve()
    havenc = (source_dir / f"havenc{exe_ext}").resolve()
    if not haven.exists() or not havenc.exists():
        print(f"Warning: {haven} or its sibling {havenc.name} not found; cannot build "
              f"{artifact}. Programs will need $HAVEN_STD or --dep.")
        return
    pkg_dir = (Path("stdlib") / pkg).resolve()
    result = subprocess.run([str(haven), "build"], cwd=str(pkg_dir), env=env,
                            encoding="utf-8", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to build {artifact}:\n{result.stderr}")
        return
    else:
        print(result.stdout, end="")
    built = pkg_dir / ".haven" / "target" / artifact
    if not built.exists():
        print(f"Failed to build {artifact}: `haven build` produced no artifact "
              f"at {built}.")
        return
    out = dest_dir / artifact
    try:
        shutil.copy2(built, out)
        print(f"Built and installed {artifact} to {out}")
    except Exception as e:
        print(f"Failed to install {artifact}: {e}")

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Manage haven binaries.")
    parser.add_argument("--debug", action="store_true", help="Copy debug builds instead of release.")
    parser.add_argument("--path", type=str, help="Destination directory (default: ~/.haven, created if missing).")
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
        dest_dir = Path(args.path).expanduser().resolve() if args.path else DEFAULT_DEST
        if not dest_dir.exists():
            print(f"Error: The path '{dest_dir}' does not exist.")
            sys.exit(1)
        dirs_to_check = [dest_dir]
        print(f"Checking for binaries in: {dest_dir}")

        removed_any = False
        for d in dirs_to_check:
            if not d.is_dir(): continue
            for bin_name in binaries + STD_ARTIFACT:
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

        # Default to `~/.haven` (created if missing); `--path` overrides it.
        # `havenc` discovers `std.hvmeta` beside its own binary, so the binaries
        # and the stdlib artifacts live together in one directory.
        dest_dir = Path(args.path).expanduser().resolve() if args.path else DEFAULT_DEST
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: could not create the destination '{dest_dir}': {e}")
            sys.exit(1)
        if not os.access(dest_dir, os.W_OK):
            print(f"Error: You do not have write permissions for '{dest_dir}'.")
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

        build_lib(source_dir, dest_dir, "std", STD_ARTIFACT)

        # `~/.haven` is not on `PATH` by default; nudge the user to add it.
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        if str(dest_dir) not in path_dirs:
            print(f"\nAdd {dest_dir} to your PATH to run haven/havenc directly, e.g.:")
            print(f'  export PATH="{dest_dir}:$PATH"')

if __name__ == "__main__":
    main()