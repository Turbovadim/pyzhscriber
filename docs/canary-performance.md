# Canary GPU performance

> **Since the 0.4.1 merge (November 2026):** these measurements were taken against the 0.2.2 engine. Upstream 0.4.1 added its own read-only weight mapping (`transcribe-mmap.cpp`, `load_common::WeightStorage`), which replaced the local mapping helper described here, including for the Core ML decoder-only path and its encoder fallback. Over 100 local recordings, the merged engine's Whisper and Canary output matched unmodified 0.4.1 exactly. The remaining local changes kept Whisper about 5% and Canary about 9% faster than unmodified 0.4.1.

Implemented September 11, 2026. Baseline: `3ce6e1e12e9a1a8bf7a1e720dbee076be756f9ac`.

This report covers the changes committed as `3b34cb4`. See the [encoder follow-up](canary-encoder-performance.md) for subsequent speed measurements and validation.

The changes target standard Canary on Metal, tested with the installed Canary 1B v2 Q5_K_M model. The main measured improvement is process memory. Warm transcription latency remains close to the baseline. See the [earlier investigation](canary-optimization-investigation.md) for the other paths considered.

## Changes

1. **Map original weights read-only.** Whisper's existing mapping helper now lives in [shared loader code](../vendor/transcribe-cpp-sys/src/transcribe-weights-mmap.cpp), and [Canary's loader](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp) uses it before attempting copied allocation. The installed model maps 797.47 MiB. Backend compatibility, tensor layout, file bounds and alignment checks remain intact. Incompatible files or backends use the original copied loader. Derived weights retain separate writable allocations. Backend buffers and references are freed before unmapping.
2. **Remove two frontend allocations.** [The shared mel frontend](../vendor/transcribe-cpp-sys/src/transcribe-mel.cpp) writes FP64 pre-emphasis directly into its padded buffer and normalizes channel-major per-feature mel output in place. Arithmetic order, precision, reflection, variance correction and retained final frame are preserved. At 16 kHz this removes an `8 × samples` byte temporary plus a `4 × mel bins × frames` byte temporary, about 5.13 MiB combined for 30 seconds of Canary audio.
3. **Keep decoder intermediates only when debugging.** [Decoder graph output marking](../vendor/transcribe-cpp-sys/src/arch/canary/decoder.cpp) is conditional, so ordinary inference lets the graph allocator reuse those intermediate buffers. Debug runs retain their named outputs.
4. **Project the last prompt position.** Ordinary greedy prompt inference passes only the final hidden row to the vocabulary head. All prompt layers and KV writes still run. Debug/full-softmax paths retain the full projection, and single-token decoding keeps its existing head shape.
5. **Grow Metal's attention read window.** Single-utterance decoding starts with 64 positions and doubles the graph's read window as needed. It preserves existing KV contents and masks through graph rebuilds. The physical KV allocation, context ceiling, token budget, EOS handling and flash-attention policy are unchanged. Other GPU backends keep their original static window.

These are engine changes. They do not alter the model file, quantization, language routing or application settings. The shared frontend changes also cover compatible frontends in other model families; their model-level accuracy is not established by the Canary results below. Canary-Qwen's loader and decoder have not been optimized in this change.

## Measurement setup

Apple M5 Pro, 18 CPU cores, 24 GB unified memory, macOS 26.5. Both native builds use Release mode, arm64, the same SDK and the same precompiled Metal 4 shader library. Tests select Metal explicitly, use default F16 KV, and retain the default encoder/decoder flash policy. The model stays loaded; each recording gets a fresh session, matching the app's normal inference lifetime.

The benchmark's `elapsed_ms` includes session initialization and inference. It excludes WAV reading, model loading, result serialization and session destruction. The engine's mel/encoder/decoder timers cover their respective stages. This is native engine timing; UI work and a separate Whisper language-detection pass are outside it.

The local corpus contains 1,595 recordings. One is only 170 samples, 10.625 ms, which the baseline rejects as too short for reflect padding. Timing comparisons use the other 1,594, with identical fixed English/Russian hints supplied to both builds. These hints are input constraints, not ground-truth language annotations. Public validation uses 400 evenly spread files from each existing LibriSpeech test-clean and test-other manifest, not the full official splits.

All runs are sequential, without concurrent native builds or another benchmark. The full corpus compares baseline, the first candidate without growing windows, and the candidate with growing windows. Additional reversed-order runs check timing drift. Raw JSONL, stderr, manifests, snapshots and memory samples are retained locally under the ignored `artifacts/canary-performance/` directory. Private recordings and their transcripts are not added to the repository.

## Results

Ranges below span two complete passes over the same 1,594 local recordings per build. The first recording is excluded from each latency/stage median; total inference time includes all recordings. Baseline/candidate order is reversed for the second pair. Public validation and an intermediate candidate run occur between these pairs, so these are repeated sequential measurements, not a controlled thermal experiment.

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Peak process physical footprint | 1,013.6 to 1,016.9 MiB | 207.6 to 212.3 MiB |
| Peak RSS | 954.9 to 955.8 MiB | 150.7 to 151.3 MiB |
| Warm median session/inference time | 61.69 to 61.88 ms | 61.39 to 61.58 ms |
| Median mel time | 0.264 to 0.283 ms | 0.251 ms |
| Median decoder time | 21.08 to 21.18 ms | 20.74 to 20.83 ms |
| Total session/inference time per pass | 106.53 to 106.81 s | 105.63 to 106.28 s |
| Model load, warm file cache | 219.8 to 314.1 ms | 57.9 to 155.1 ms |

The process-footprint reduction is about 79%. Ordinary transcription speed changes are small, under 1% in these full-corpus totals. Loading is generally cheaper, but its timing varies substantially with Metal initialization and cache state; another public-sample candidate load took 252 ms. Do not interpret the fastest load as a guaranteed startup time.

The initial candidate already contains mapping, frontend storage changes and prompt/output pruning. Its full-corpus median was 61.63 ms. Growing windows do not produce a substantial short-dictation improvement by themselves. On the separate 30/65/96-second sequence, including three short recordings afterward in the same session, measured totals were 2.43 s for baseline, 2.11 s for the initial candidate and 2.05 s with growing windows. That sequence ran once per build and is supporting evidence, not a general long-audio speed claim.

## What the memory reduction means

The large decrease comes from replacing a private copied weight allocation with read-only file-backed storage shared with Metal. The model's actively used pages still occupy unified RAM. macOS can reclaim clean file pages and reload them from the GGUF, and charges them differently from private allocations. A lower process physical footprint therefore does **not** mean the model no longer resides in RAM or that total system RAM fell by the same amount.

Reported peaks use macOS `proc_pid_rusage` physical footprint, sampled every 50 ms. The runner also records memory at model/session lifecycle points. RSS is recorded separately. Neither value is a measurement of all system memory occupied by the mapped file. File-page residency, memory pressure and swapping were not controlled. No cold-disk load claim is made: repeated tests have a warm filesystem cache.

## Validation and reproduction

All 2,394 supported recordings produced identical text, raw text, language, segment results, status and truncation flags. This is output parity on the tested corpus, not a proof that every floating-point intermediate or token ID is identical for every possible input. The frontend comparison separately establishes bitwise equality on its test inputs.

| Public subset | Files | Reference words | Word errors, both builds | WER, both builds |
| --- | ---: | ---: | ---: | ---: |
| LibriSpeech test-clean sample | 400 | 8,177 | 141 | 1.724% |
| LibriSpeech test-other sample | 400 | 7,347 | 242 | 3.294% |

WER uses the existing English normalizer pinned to Whisper revision `86098128c0b4f24f0e2aa2994de830614b474227`, with its local source checksums verified. It normalizes references and hypotheses identically before word-level edit distance. These sample scores are not full-split benchmark results. The private corpus has no manually verified reference transcription, so it supports parity testing, not an accuracy score.

The native regression suite compares mel output bit for bit with a separately compiled baseline frontend. Coverage includes padding and normalization modes, FFT shapes, thread counts, input/output aliasing, reused output vectors and long input. The mapper fixtures cover CPU, shared Metal, private-Metal fallback, malformed layouts, failure cleanup, mapped-file unlink and separate writable F32 promotion from mapped F16 weights, including Q5_K/Q6_K tensor spans.

The [real-model fixture](../tests/canary-native/lifecycle.cpp) passed against baseline and final libraries with identical results. It uses 3.06-second and 96-second inputs to check fresh and reused sessions, both mixed-length batch orders followed by single inference, cancellation recovery, context limits 63/64/65/127/128/129 and five model destruction/reload cycles. All six constrained-context runs truncated identically. Both batches also matched their fresh single results for these inputs. For future inputs, compare corresponding baseline batch rows: batch kernels can already differ from single inference.

Further baseline/final comparisons passed:

- A 300-second concatenation of public speech at context limits 511/512/513/1024. All runs reached the expected output limit and returned identical text and truncation status. The default-context run generated the full 512-token budget, exercising growth past 64, 128, 256 and 512 cache positions.
- F32 KV and disabled flash attention on the 30/65/96-second sequence plus three short recordings, six per mode.
- Three recordings each for punctuation off, Russian-to-English translation, private Metal buffers and CPU inference. Logs confirm private Metal uses copied loading and CPU uses mapped originals with its separate promoted weights. These are compatibility checks, not a broader CPU optimization claim.
- The original 170-sample recording still returns the same invalid-argument status. Its exclusion from successful inference timing does not hide a candidate-only failure.

Growing the window and changing the prompt projection can select different GPU kernels. Output parity here does not establish identical results for every model quantization, GPU generation or debug configuration. Other GPU backends were not run on this Mac.

The Rust library suite passed 21 tests, with its existing optional Metal worker-shutdown test left ignored. The release app and Whisper benchmark built successfully. Three real Whisper recordings matched the previously deployed engine, checking the shared mapping extraction alongside the native frontend/mapping fixtures.

Build the benchmark against an installed static native prefix:

```bash
cmake -S vendor/transcribe-cpp-sys -B BUILD \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_SYSROOT="$SDKROOT" -DCMAKE_INSTALL_PREFIX="$(pwd)/NATIVE_PREFIX" \
  -DTRANSCRIBE_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DTRANSCRIBE_INSTALL=ON -DTRANSCRIBE_BUILD_TESTS=OFF \
  -DTRANSCRIBE_BUILD_EXAMPLES=OFF -DTRANSCRIBE_BUILD_TOOLS=OFF
cmake --build BUILD --target install -j10
python3 scripts/build-canary-bench.py NATIVE_PREFIX canary_bench --sdk "$SDKROOT"
./canary_bench MODEL.gguf MANIFEST.tsv 1
```

Each manifest line is a 16 kHz mono PCM16 WAV path, optionally followed by a tab and language code. Place the same `default.metallib` and `ggml-tensor.metallib` (built by `scripts/package-macos.sh`) beside both benchmark binaries to avoid comparing precompiled shader loading with source compilation. The runner also supports session reuse, model reloads, CPU, F32 KV, context limits, translation and punctuation-off checks; see [its source](../examples/canary_bench.cpp) for arguments.

```bash
python3 scripts/test-whisper-native.py NATIVE_PREFIX \
  --baseline-source BASELINE_SOURCE/vendor/transcribe-cpp-sys
python3 scripts/test-canary-native.py NATIVE_PREFIX MODEL.gguf SHORT.f32 LONG.f32 \
  --sdk "$SDKROOT"
```

Lifecycle inputs are mono 16 kHz native float32 PCM; convert PCM16 using division by 32767 to match the app and benchmark. For performance comparisons, use `--build-only --output BIN` on the lifecycle wrapper and compile both versions before running either.

Physical KV shrinking, encoder/decoder device-buffer handoff, different encoder flash policy, Core ML and Canary-Qwen loading/packing remain separate changes. They need their own ownership or numerical validation and are not included in these measurements.
