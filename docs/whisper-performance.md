# Whisper inference performance

> **Since the 0.4.1 merge (November 2026):** these measurements were taken against the 0.2.2 engine. Upstream 0.4.1 added its own read-only weight mapping (`transcribe-mmap.cpp`, `load_common::WeightStorage`), which replaced the local mapping helper described here, including for the Core ML decoder-only path and its encoder fallback. Over 100 local recordings, the merged engine's Whisper and Canary output matched unmodified 0.4.1 exactly. The remaining local changes kept Whisper about 5% and Canary about 9% faster than unmodified 0.4.1.

The Whisper optimizations preserve the installed model weights, encoder precision, decoding thresholds, temperature fallback, language constraints, and timestamp rules. They remove redundant computation and temporary storage in the vendored `transcribe-cpp-sys` engine.

- The mel frontend skips FFT/filterbank work only when every windowed input is exactly zero and the filterbank is finite. It retains the original normalization over the complete recording.
- Whisper requests frame-major mel output directly and lets the frontend apply logical zero-padding. This removes the padded PCM copy, intermediate mel matrix, and transpose. Complete encoder windows borrow that output; only a final partial window needs a padding buffer.
- When Whisper needs no minimum-length padding and pre-emphasis is disabled, FFT windows also borrow the original PCM. Only boundary samples need reflection or zero handling. Other configurations and inputs that overlap the output retain the copying path.
- Core ML can write into an existing CPU/shared Metal encoder-output buffer. Models that reject or ignore the offered output backing retain a copying fallback.
- Automatic language detection projects only the aligned span containing allowed language tokens. Its existing F32 cross-attention projections also populate the persistent cache, avoiding a second projection pass.
- Prompt decoding projects the SOT row needed for no-speech detection and the final row needed for token selection. Every prompt position still participates in the transformer and KV cache. Validation dumps retain the full projection.
- Greedy token selection uses independent maximum lanes followed by the first matching token. The scalar exponentials, accumulation order, rounding, tie behavior, and nonfinite fallbacks remain intact. Timestamp processing skips masks that have already been applied.
- The native encoder avoids redundant convolution transposes and lets supported CPU/Metal flash-attention kernels read strided Q views. K/V remain contiguous for Metal's padding requirements.
- Weight loading reads directly into ordinary CPU and shared Metal buffers. Other buffer types retain their `tensor_set` path, including backends that transform weight storage.
- Compatible contiguous Whisper weights use a read-only file mapping on macOS CPU/shared Metal. The existing Core ML path already loaded only decoder weights; those weights now use a file mapping. Native inference maps the full model, and lazy encoder fallback gets its own mapping. Private Metal, other backends, incompatible layouts, and mapping failures retain copied loading. The mappings outlive model buffers and backends. Model files must remain immutable while loaded; the app's temporary-file/hash/rename installer satisfies this, and unlink/rename remains safe.

Measured on September 10, 2026, on an Apple M5 Pro with 24 GB RAM and macOS 26.5, using the installed large-v3-turbo Q8_0 model and its Core ML encoder. The comparison used 1,595 local dictation recordings, English/Russian language constraints, seed 42, no custom prompt, and a fresh session per recording. Four complete runs used baseline/candidate/candidate/baseline order and the same packaged Metal shader library. The baseline was commit `c8806839c7629ff515c886d332f27338c881850f`.

The four hybrid trials and 64-cycle lifecycle comparison used benchmark build v5, which includes decoder mapping. Benchmark build v6 also maps native encoder weights and retains the same hybrid inference path. The public reference, full CPU, and initial long-recording comparisons below use v6. Final build v7 adds borrowing of unextended PCM; its additional validation is reported after those comparisons. Reported timing medians exclude the first recording of each run; output comparisons include every recording, and memory sampling includes model load and the complete run.

| Hybrid Core ML / Metal path | Original | Optimized |
| --- | ---: | ---: |
| Median latency, each complete run | 127.2–127.6 ms | 116.4–116.5 ms |
| Peak physical footprint, each complete run | 784–786 MiB | 593–622 MiB |
| Complete output agreement | 6,380 / 6,380 runs | Identical text, raw text, language, and segments |

The median latency reduction was 8.6%. Physical footprint was sampled every 50 ms in the inference process, including model load. RSS was more variable: 1,233–1,236 MiB for the baseline and 1,041–1,305 MiB for the candidate. RSS includes reclaimable file pages; the physical-footprint result is the memory-pressure comparison. These are local measurements, not a guarantee for other hardware or models.

The additional native mapping was measured separately against the optimized engine with copied native weights. In two trials per version, peak physical footprint fell from 1,004 to 160 MiB on Metal and from 982 to 138 MiB on CPU. Metal used 61 recordings per trial; CPU used three, so its timing evidence is limited. Every output matched. A forced Core ML fallback also produced the same native output, including token probabilities, while mapping the encoder's 667 MiB span. File-backed weights can remain resident in the system file cache.

A full native CPU comparison used the original baseline and final integrated build, eight threads, and all 1,595 local recordings. Every output matched. Median latency fell from 2,922.8 to 2,866.8 ms (1.9%); median decoding time fell from 114.0 to 69.4 ms. Peak physical footprint fell from 1,098 to 148 MiB, and peak RSS fell from 1,107 to 1,003 MiB. Model load took 218.6 versus 44.0 ms. This was one complete run per version; encoder computation dominates this path's inference time.

Build v6 matched every output in each reference comparison below. [LibriSpeech test-clean and test-other](https://www.openslr.org/12) cover 5.4 and 5.34 hours of read English speech. Scoring used [Whisper's English normalizer](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/normalizers/english.py). Each row used one complete run per version and the same decoding settings, including English/Russian language constraints.

| Backend | Reference set | Recordings | Word errors / reference words, both | WER, both | Median latency, original → optimized |
| --- | --- | ---: | ---: | ---: | ---: |
| Core ML / Metal | test-clean | 2,620 | 1,069 / 53,027 | 2.016% | 144.2 → 132.0 ms |
| Core ML / Metal | test-other | 2,939 | 2,068 / 52,882 | 3.911% | 140.9 → 127.6 ms |
| Native CPU, eight threads | test-clean subset | 800 | 321 / 17,337 | 1.852% | 2,958.0 → 2,898.8 ms |
| Native Metal | test-clean | 2,620 | 1,042 / 53,027 | 1.965% | 289.6 → 266.1 ms |

The CPU subset selected 20 clips spread across each of the 40 speakers. The full native Metal comparison reduced median latency by 8.1% and peak physical footprint from 1,124 to 162 MiB; peak RSS fell from 1,069 to 110 MiB. On hybrid test-other, median latency fell 9.5% and peak physical footprint fell from 778 to 585 MiB. These results establish unchanged accuracy on the tested material.

Validation also includes the app tests and live Metal shutdown check, generated CPU/shared/private Metal fixtures, cancellation recovery, batch/single session reuse, and 64 model load/run/unload cycles per version. Outputs remained equal and each mapping was released. Memory settled into a narrow range over the last 16 reloads; Core ML runtime caches remained after model release.

A 30-minute fixture concatenated public reference clips with 250 ms gaps. Both builds produced the same text and all 284 segments, ending at 30:00. In this single pair, inference took 15.83 versus 14.86 seconds and peak physical footprint was 983 versus 784 MiB. Three short recordings then ran in the same session, also with identical outputs. This checks long-recording window transitions and reuse after a much larger mel allocation.

A two-hour fixture used 1,063 complete test-other clips with the same gaps. Both builds produced identical text and all 1,228 segments, followed by matching outputs on three short recordings in the same session. Inference took 64.03 versus 61.14 seconds, and peak physical footprint was 2,019 versus 1,820 MiB.

Final build v7 additionally matched all 1,595 local recordings and the two-hour fixture, including subsequent session reuse. Its two-hour inference took 60.91 seconds and peak physical footprint was 1,381 MiB, down from the original baseline's 2,019 MiB. The extra PCM optimization also passed a bitwise comparison of the real two-hour mel output, the native fixture suite, and input/output alias checks. The final release build, app tests, and all-target check passed.

After compilation finished, a fresh baseline/candidate/candidate/baseline comparison used 61 representative recordings per run. Every output matched. Median-of-run medians fell from 133.74 to 124.19 ms (7.1%), and peak physical footprint was 813–814 versus 610 MiB. The full 1,595-recording v7 run overlapped compilation and is used for output agreement only, not timing claims.

Changing the prompt projection shape can select a different Metal matrix kernel. Its floating-point rounding can differ slightly even though the model weights and decoding settings are unchanged. Comparisons therefore check complete outputs and decoding decisions; they do not claim bitwise identity for every intermediate tensor. The mel frontend and CPU token-processing changes also have separate bitwise comparisons.

`examples/whisper_bench.rs` exercises the same Rust API and normal session lifetime as the app. Each manifest line names a local mono PCM16 WAV at 16 kHz. A fixed seed makes sampled fallback comparable between builds.

```sh
cargo run --locked --release --example whisper_bench -- model.gguf manifest.txt 3 > results.jsonl
```

The JSON Lines output contains text, raw text, language, segments, stage timings, and current macOS RSS/physical footprint after model load, each recording, and model release. It contains recording paths and transcript content; keep it local. Those memory readings are snapshots, not a continuous peak measurement.

Useful comparison options:

| Option | Purpose |
| --- | --- |
| `--language en` | Exercise an explicit language hint |
| `--allowed-languages ''` | Allow all model languages |
| `--prompt 'domain vocabulary'` | Exercise prompt projection and truncation |
| `--timestamps none` | Exercise decoding without timestamp output |
| `--temperature 0.4` | Exercise sampled decoding with the fixed seed |
| `--kv-f32` | Preserve the explicit F32 cache path |
| `--reuse-session` | Check buffers across repeated runs in one session |
| `--cpu --threads 8` | Select the CPU decoder/backend |

Core ML remains active when a valid encoder companion is present, including with `--cpu`. To compare native encoders, place a model symlink in a separate directory without its Core ML companion. Use the same backend, model, manifest, seed, thread settings, and Metal shader library for both builds. `TRANSCRIBE_PERF_DEBUG=all` adds detailed CPU/GPU stage counters to stderr; leave it unset for ordinary latency measurements.

The focused native checks use small generated fixtures and need no model download. Pass the Cargo native build output containing `lib/transcribe-link.json`. An unchanged baseline source directory enables bitwise frontend comparisons, including the new output layout and implicit padding.

```sh
python3 scripts/test-whisper-native.py \
  target/release/build/transcribe-cpp-sys-<id>/out \
  --baseline-source /path/to/unchanged/transcribe-cpp-sys
```

On the development Mac, the command-line tools selected a macOS 27 SDK that the installed linker could not read. Builds and native checks used the matching Xcode SDK through `SDKROOT`; no global toolchain setting was changed.
