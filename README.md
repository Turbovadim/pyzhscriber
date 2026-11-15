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
- Optional cloud dictation through OpenAI, Groq, Mistral, Google Gemini, OpenRouter or any OpenAI-compatible server, with your own API key. Streaming models (OpenAI `gpt-live-transcribe`, Mistral `voxtral-mini-transcribe-realtime-3`) receive audio while you speak, so the text is ready almost as soon as you stop. Local models take over when a cloud request fails.

## Build and run

Install Rust through rustup, Xcode with its command-line tools, and CMake. The repository pins Rust 1.98.1 in `rust-toolchain.toml`; rustup installs it when you run Cargo. Native dependencies compile C/C++ and Metal code, so the first build can take a while.

With Homebrew installed, CMake can be installed with:

```sh
brew install cmake
```

From the project directory:

```sh
cargo run --locked --bin pyzhscriber
```

Keep `Cargo.lock` and `vendor/` in your checkout. Cargo uses a local patch for transcribe.cpp, which is required to build this version. GPUI comes from the [gpui-bisquit](https://github.com/Bisquit-Host/gpui-bisquit) fork.

### First launch

1. Grant Microphone, Accessibility, and Input Monitoring access when prompted. These allow recording, text insertion, and shortcuts while another app is focused.
2. Open the model settings and download the models for your transcription mode. Single-model mode defaults to Whisper Large V3 Turbo Q8_0. Default routed mode needs GigaAM V3 E2E CTC Q8_0, Parakeet Unified English 0.6B Q8_0, and Whisper Tiny Q8_0 for language detection.
3. Hold Right Command, speak, and release. You can change the shortcut and recording mode in settings.

On first launch, Pyzhscriber imports compatible settings from Handy if available. Imported settings can change those defaults. It also reuses matching models in the Hugging Face caches at `~/.cache/huggingface/hub` and `~/Library/Caches/huggingface/hub`.

Model downloads come from `blob.handy.computer` and are checked against SHA-256 hashes in the bundled [model catalog](resources/model-catalog.json). Once the required models are installed, transcription works offline.

The catalog is a verbatim copy of `catalog.json` from the [`@handy-computer/models`](https://www.npmjs.com/package/@handy-computer/models) npm package, currently 0.2.3. To update it, replace the file with a newer package's `catalog.json` and run `cargo test`, which fails if a default or ranked model is gone.

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

After moving the app to Applications, open **Advanced → Install CLI**. Then open a new terminal:

```sh
pyzhscriber status --json
pyzhscriber transcribe recording.m4a --json
```

Setup creates `~/.local/bin/pyzhscriber` pointing to the app and adds that directory to PATH in your zsh or bash startup files. It preserves existing content and refuses to overwrite another command. No administrator access is required. Other shells need `~/.local/bin` added to PATH manually. The link continues to work when the app is updated at the same location.

You can also run setup from a terminal:

```sh
/Applications/Pyzhscriber.app/Contents/MacOS/Pyzhscriber install-cli
```

Open Pyzhscriber and download the required models, then call its executable directly:

```sh
APP=/Applications/Pyzhscriber.app/Contents/MacOS/Pyzhscriber
"$APP" status --json
"$APP" transcribe recording.wav
"$APP" transcribe recording.wav --json > transcript.json
```

`transcribe` uses the running app's current settings and loaded model. It prints the transcript to stdout without focusing a window, inserting text, changing the clipboard, saving history, or applying AI cleanup. Model inference is serialized with other CLI requests and microphone dictation. Overrides apply only to that request.

For development or model overrides:

```sh
cargo run --locked --bin pyzhscriber -- transcribe recording.wav --mode single
cargo run --locked --bin pyzhscriber -- transcribe recording.wav --mode routed
cargo run --locked --bin pyzhscriber -- transcribe recording.wav \
  --model handy-computer/whisper-large-v3-turbo-gguf/whisper-large-v3-turbo-Q8_0.gguf
```

`--model` selects single-model mode and clears language limits for that invocation. Input supports WAV, MP3, FLAC, Ogg/Vorbis, Ogg/Opus, AAC, and M4A/MP4 with AAC audio. The decoder detects the format from the file contents, mixes channels to mono, and resamples to 16 kHz. Both app and standalone modes support these formats without FFmpeg. Opus voice messages can use `.ogg` or `.opus` extensions; mono and stereo mapping family 0 are supported, including encoder padding removal and header gain. Chained/multiplexed Opus streams and Apple Lossless are not supported.

For agents, `--json` returns `{"type":"transcript","text":"..."}` on success or `{"type":"error","error":"..."}` on failure. Exit status is 0 on success, 1 for runtime errors, and 2 for invalid arguments. Argument errors and help use Clap's usual text format. Diagnostics go to stderr. `status --json` reports the app version, runtime status, and loaded model ID. Requests wait for completion; disconnecting the CLI does not cancel inference.

The app must be running a build with CLI support. If it is unavailable, the command fails with instructions. Add `--standalone` to load a separate model using saved settings without the app:

```sh
"$APP" transcribe recording.wav --standalone --json
```

Opus files that end on complete packets without a final end-of-stream marker are accepted. In that case, all decoded frames after the initial encoder padding are retained because no final trim is available.

Communication uses a Unix socket in the app's data directory, with access restricted to your macOS user. No network port is opened.

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

Pyzhscriber is licensed under [MIT](LICENSE), with Handy's original copyright notice preserved. Vendored dependencies retain their own licenses. Model weights have their own licenses and are downloaded separately.
