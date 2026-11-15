# Vendored dependencies

Keep these directories in the repository. The root `Cargo.toml` uses them through Cargo's `[patch]` tables.

## transcribe.cpp

[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) is Handy's speech recognition layer. Pyzhscriber uses its Rust bindings and native implementation for local inference. Credit belongs to the Handy and transcribe.cpp authors; their MIT notices are preserved in the vendored license files.

`transcribe-cpp/` and `transcribe-cpp-sys/` contain the published 0.4.1 crates with local extensions for constrained Whisper language detection, Core ML encoder support, Whisper and Canary inference optimizations, and loading pre-compiled ggml Metal libraries in the embedded build. Weight memory mapping is upstream's. Native sources, GGML, and their license files are included. See [Whisper inference performance](../docs/whisper-performance.md) for the optimizations and comparison tools.

## GPUI

GPUI is not vendored. It comes from [gpui-bisquit](https://github.com/Bisquit-Host/gpui-bisquit), a fork of gpui-fast whose patches include releasing per-window Metal resources after a window has been fully occluded for two seconds.
