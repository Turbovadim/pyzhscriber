#!/usr/bin/env python3
"""Run native loading/mapping checks and optional bitwise mel comparison on macOS.

Pass the transcribe-cpp-sys Cargo OUT_DIR containing lib/transcribe-link.json.
The optional baseline source is an unchanged transcribe-cpp-sys directory.
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
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--sdk", default=os.environ.get("SDKROOT"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "vendor/transcribe-cpp-sys"
    tests = root / "tests/whisper-native"
    prefix = args.native_prefix.resolve()
    manifest = json.loads((prefix / "lib/transcribe-link.json").read_text())
    if manifest["shared"] or "metal" not in manifest["backends"]:
        parser.error("loader checks require a static Metal build")

    compiler = ["clang++", "-std=c++17", "-O3", "-DNDEBUG"]
    if args.sdk:
        compiler += ["-isysroot", args.sdk]
    includes = ["-I" + str(source / "src"), "-I" + str(prefix / "include")]
    libraries = ["-L" + str(prefix / manifest["lib_dir"])]
    libraries += ["-L" + str(prefix / path) for path in manifest["library_paths"]]
    libraries += ["-l" + name for name in manifest["libraries"] + manifest["system_libs"]]
    for framework in manifest["frameworks"]:
        libraries += ["-framework", framework]
    libraries += manifest["link_flags"]

    with tempfile.TemporaryDirectory(prefix="pyzhscriber-native-tests-") as directory:
        work = Path(directory)
        mapping_includes = includes + [
            "-I" + str(source / "src/arch/whisper"),
            "-I" + str(source / "ggml/src"),
        ]
        for name, test_includes in [("loader", includes), ("mmap", mapping_includes)]:
            executable = work / name
            subprocess.run(
                compiler + test_includes + [str(tests / f"{name}.cpp"), "-o", str(executable)] + libraries,
                check=True,
            )
            for mode in ["cpu", "metal-shared", "metal-private"]:
                subprocess.run([str(executable), mode], check=True)

        if args.baseline_source:
            baseline = args.baseline_source.resolve()
            baseline_object = work / "baseline.o"
            baseline_wrapper = work / "baseline-wrapper.o"
            candidate_object = work / "candidate.o"
            baseline_includes = ["-I" + str(baseline / "src"), "-I" + str(baseline / "include")]
            subprocess.run(
                compiler + baseline_includes + [
                    "-DMelFrontend=BaselineMelFrontend", "-c", str(baseline / "src/transcribe-mel.cpp"),
                    "-o", str(baseline_object),
                ], check=True,
            )
            # The wrapper must see the baseline method declaration, including
            # its original default arguments, independently of the new header.
            subprocess.run(
                compiler + baseline_includes + [
                    "-c", str(tests / "mel-baseline.cpp"), "-o", str(baseline_wrapper),
                ], check=True,
            )
            subprocess.run(
                compiler + includes + [
                    "-c", str(source / "src/transcribe-mel.cpp"), "-o", str(candidate_object),
                ], check=True,
            )
            mel = work / "mel"
            subprocess.run(
                compiler + includes + [
                    str(tests / "mel.cpp"), str(baseline_wrapper), str(baseline_object),
                    str(candidate_object), "-framework", "Accelerate", "-o", str(mel),
                ], check=True,
            )
            subprocess.run([str(mel), "--full"], check=True)


if __name__ == "__main__":
    main()
