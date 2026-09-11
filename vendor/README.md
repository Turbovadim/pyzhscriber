# Vendored dependencies

Keep these directories in the repository. The root `Cargo.toml` uses them through Cargo's `[patch]` tables.

## transcribe.cpp

[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) is Handy's speech recognition layer. Pyzhscriber uses its Rust bindings and native implementation for local inference. Credit belongs to the Handy and transcribe.cpp authors; their MIT notices are preserved in the vendored license files.

`transcribe-cpp/` and `transcribe-cpp-sys/` contain the published 0.2.2 crates with local extensions for constrained Whisper language detection, Core ML encoder support, and inference optimizations. Native sources, GGML, and their license files are included. See [Whisper inference performance](../docs/whisper-performance.md) for the optimizations and comparison tools.

## GPUI macOS platform

`zed-gpui-occlusion/` contains patched `gpui_apple` and `gpui_macos` crates from Zed commit `fecc3273ed32643c2ea1b04a74c8780e2c9ffaf8`. The patch releases per-window Metal resources after a window has been fully occluded for two seconds.

See its [README](zed-gpui-occlusion/README.md) for implementation details and standalone checks. The upstream Apache-2.0 license is included there.
