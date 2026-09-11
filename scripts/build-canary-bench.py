#!/usr/bin/env python3
"""Build the Canary benchmark against an installed native build's link manifest."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_prefix", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sdk", default=os.environ.get("SDKROOT"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = args.native_prefix.resolve()
    manifest = json.loads((prefix / "lib/transcribe-link.json").read_text())
    if manifest["shared"]:
        parser.error("use a static native build for reproducible benchmark binaries")
    command = ["clang++", "-std=c++17", "-O3", "-DNDEBUG"]
    if args.sdk:
        command += ["-isysroot", args.sdk]
    command += [str(root / "examples/canary_bench.cpp"), "-I" + str(prefix / manifest["include_dir"])]
    command += ["-L" + str(prefix / manifest["lib_dir"])]
    command += ["-L" + str(prefix / path) for path in manifest["library_paths"]]
    command += ["-l" + name for name in manifest["libraries"] + manifest["system_libs"]]
    for framework in manifest["frameworks"]:
        command += ["-framework", framework]
    command += manifest["link_flags"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command + ["-o", str(args.output)], check=True)


if __name__ == "__main__":
    main()
