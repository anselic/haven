import os
import sys
import shutil
import platform
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Manage haven binaries.")
    parser.add_argument("--debug", action="store_true", help="Copy debug builds instead of release.")
    parser.add_argument("--path", type=str, help="Specific destination directory (e.g., ~/.local/bin).")
    parser.add_argument("--remove", action="store_true", help="Delete the binaries instead of installing them.")
    args = parser.parse_args()

    is_windows = platform.system() == "Windows"
    exe_ext = ".exe" if is_windows else ""
    binaries = [
        f"haven{exe_ext}",
        f"havenc{exe_ext}",
        f"havendoc{exe_ext}",
    ]

    # Handle removal
    if args.remove:
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
            for bin_name in binaries:
                target_file = d / bin_name
                if target_file.exists():
                    try:
                        target_file.unlink()
                        print(f"Success: Deleted {target_file}")
                        removed_any = True
                    except Exception as e:
                        print(f"Failed to delete {target_file}: {e}")

        if not removed_any:
            print("No binaries found to remove.")

    # Handle installation
    else:
        build_type = "debug" if args.debug else "release"
        source_dir = Path("target") / build_type

        if not source_dir.exists():
            print(f"Error: Directory '{source_dir}' does not exist.")
            print("Did you forget to build the damn project?")
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
                    print(f"Success: Copied {bin_name} to {dest_file}")
                except Exception as e:
                    print(f"Failed to copy {bin_name}: {e}")
            else:
                print(f"Warning: {src_file} not found in {source_dir}. Skipping.")

if __name__ == "__main__":
    main()