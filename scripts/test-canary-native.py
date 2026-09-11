#!/usr/bin/env python3
"""Build/run the real-model Canary lifecycle fixture against a native prefix.

Inputs are native float32 mono PCM at 16 kHz. Use the same inputs and options
for baseline and candidate, then compare their JSONL output. --build-only
requires --output and does not load a model or run inference.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_prefix", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("short_pcm", type=Path)
    parser.add_argument("long_pcm", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--sdk", default=os.environ.get("SDKROOT"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--kv-f32", action="store_true")
    args = parser.parse_args()
    if args.build_only and not args.output:
        parser.error("--build-only requires --output")
    root = Path(__file__).resolve().parents[1]
    prefix = args.native_prefix.resolve()
    manifest = json.loads((prefix / "lib/transcribe-link.json").read_text())
    if manifest["shared"]:
        parser.error("use a static native build for this fixture")
    command = ["clang++", "-std=c++17", "-O3", "-DNDEBUG"]
    if args.sdk:
        command += ["-isysroot", args.sdk]
    command += [str(root / "tests/canary-native/lifecycle.cpp"), "-I" + str(prefix / manifest["include_dir"])]
    command += ["-L" + str(prefix / manifest["lib_dir"])]
    command += ["-L" + str(prefix / path) for path in manifest["library_paths"]]
    command += ["-l" + name for name in manifest["libraries"] + manifest["system_libs"]]
    for framework in manifest["frameworks"]:
        command += ["-framework", framework]
    command += manifest["link_flags"]
    with tempfile.TemporaryDirectory(prefix="canary-lifecycle-") as temporary:
        output = args.output.resolve() if args.output else Path(temporary) / "lifecycle"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command + ["-o", str(output)], check=True)
        if not args.build_only:
            invocation = [str(output), str(args.model), str(args.short_pcm), str(args.long_pcm),
                          "--language", args.language, "--threads", str(args.threads)]
            if args.cpu:
                invocation.append("--cpu")
            if args.kv_f32:
                invocation.append("--kv-f32")
            subprocess.run(invocation, check=True)


if __name__ == "__main__":
    main()
