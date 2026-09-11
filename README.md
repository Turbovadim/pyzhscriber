# Pyzhscriber

Pyzhscriber is a fork of [Handy](https://github.com/cjpais/Handy) by CJ Pais and contributors, adapted into a native macOS app built with Rust and GPUI. Hold a shortcut, speak, and release to insert the transcript into the focused app.

Speech recognition runs locally through Handy's [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) layer with Metal acceleration. This project builds on Handy's work, including its model catalog and distribution infrastructure. The vendored transcribe.cpp Rust bindings and native code include local patches described in [vendor/README.md](vendor/README.md).

This is a developer preview for macOS with Apple Silicon. Download the DMG from [GitHub Releases](https://github.com/Turbovadim/pyzhscriber/releases), open it, and drag Pyzhscriber to Applications. The app is development-signed and not notarized, so macOS may require approval in System Settings → Privacy & Security on first launch. Other platforms are not supported by this setup.

## What it does

- Single-model transcription with optional language limits, or language-to-model routing. The default routes use GigaAM for Russian and Parakeet for English.
- Configurable push-to-talk or toggle shortcuts, microphone selection, and a recording overlay.
- Menu bar controls with optional Dock hiding, and a pill voice sensitivity slider from 0.25× to 6×.
- Direct typing, clipboard paste, or clipboard-only output.
- Local transcript history and recording playback.
- Optional transcript cleanup through an OpenAI-compatible API.

## Build and run

Install Rust through rustup, Xcode with its command-line tools, and CMake. The repository pins Rust 1.97.1 in `rust-toolchain.toml`; rustup installs it when you run Cargo. Native dependencies compile C/C++ and Metal code, so the first build can take a while.

With Homebrew installed, CMake can be installed with:

```sh
brew install cmake
```

From the project directory:

```sh
cargo run --locked --bin pyzhscriber
```

Keep `Cargo.lock` and `vendor/` in your checkout. Cargo uses local patches for transcribe.cpp and GPUI; these are required to build this version.

### First launch

1. Grant Microphone, Accessibility, and Input Monitoring access when prompted. These allow recording, text insertion, and shortcuts while another app is focused.
2. Open the model settings and download the models for your transcription mode. Single-model mode defaults to Whisper Large V3 Turbo Q8_0. Default routed mode needs GigaAM V3 E2E CTC Q8_0, Parakeet Unified English 0.6B Q8_0, and Whisper Tiny Q8_0 for language detection.
3. Hold Right Command, speak, and release. You can change the shortcut and recording mode in settings.

On first launch, Pyzhscriber imports compatible settings from Handy if available. Imported settings can change those defaults. It also reuses matching models in the Hugging Face caches at `~/.cache/huggingface/hub` and `~/Library/Caches/huggingface/hub`.

Model downloads come from `blob.handy.computer` and are checked against SHA-256 hashes in the bundled [model catalog](resources/model-catalog.json). Once the required models are installed, transcription works offline.

## Build and install a signed app

The packaging script requires an installed Apple Development signing identity and an Xcode Metal toolchain that supports `-std=metal4.0`.

```sh
./scripts/package-macos.sh
open /Applications/Pyzhscriber.app
```

This builds a release bundle at `dist/Pyzhscriber.app`, signs it, and installs it at `/Applications/Pyzhscriber.app`, replacing an existing installation. If Pyzhscriber is running, the script quits it before replacement and relaunches it afterwards. This is a local development build, not a notarized distribution.

Use `./scripts/package-macos.sh --dmg` to create a DMG and SHA-256 checksum in `dist/` without replacing the installed app.

By default, the script selects the first installed Apple Development identity. To choose one explicitly:

```sh
PYZHSCRIBER_SIGN_IDENTITY='Apple Development: Your Name (TEAMID)' ./scripts/package-macos.sh
```

Keep the signing identity and installation path stable so macOS permissions remain associated with the app. For daily use, prefer the signed bundle: launching through a terminal can attribute permissions to the terminal instead.

## Command line

Transcribe a WAV file after downloading the model in the app:

```sh
cargo run --locked --bin pyzhscriber -- transcribe recording.wav --mode single
cargo run --locked --bin pyzhscriber -- transcribe recording.wav --mode routed
cargo run --locked --bin pyzhscriber -- transcribe recording.wav \
  --model handy-computer/whisper-large-v3-turbo-gguf/whisper-large-v3-turbo-Q8_0.gguf
```

The CLI uses saved settings. `--model` selects single-model mode and clears language limits for that invocation. WAV input supports integer PCM up to 16 bits and 32-bit float, with channels mixed to mono and resampled to 16 kHz. The CLI prints the transcript without inserting it into another app or applying AI cleanup.

To diagnose permissions and microphone capture for the installed bundle, quit the app, then run:

```sh
open -n -g --stdout /tmp/pyzhscriber-doctor.txt \
  -a /Applications/Pyzhscriber.app --args doctor
cat /tmp/pyzhscriber-doctor.txt
```

Allow a few seconds for the capture test before reading the output. `doctor` prints permission status, keyboard event-tap availability, input devices, and the peak level of a short recording. A zero peak means no audio reached the process.

## Local data and optional network access

Settings, downloaded models, transcript history, and recordings live under:

```text
~/Library/Application Support/com.yolki.pyzhscriber/
```

The directory contains `settings.json`, `models/`, `history.db`, and `recordings/`. Logs are written to `~/Library/Logs/Pyzhscriber.log`.

AI cleanup is disabled by default. Enabling it sends transcript text to the configured API endpoint. Its API key is stored in plain text in `settings.json`. Keep settings, history, recordings, and logs out of commits and issue attachments.

## Development checks

```sh
cargo test --locked --lib
cargo check --locked --all-targets
```

The optional `visual-qa` feature enables the `pyzhscriber-visual-qa` binary. Generated screenshots and local design notes are excluded from Git.

See [vendor/README.md](vendor/README.md) for the native patches and their provenance.

## License

Pyzhscriber is licensed under [MIT](LICENSE), with Handy's original copyright notice preserved. Vendored dependencies retain their own licenses, including the Apache-2.0 GPUI platform code. Model weights have their own licenses and are downloaded separately.
