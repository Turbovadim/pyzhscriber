# Canary encoder performance

September 11, 2026. Baseline: `3b34cb436fc6271f0b636f04e0cd843a73b6b5c1`, which already includes the [previous Canary memory and decoder changes](canary-performance.md).

This change removes encoder copies and changes the input layout of Metal's existing depthwise-convolution kernel. Two full local-corpus passes reduced total session/inference time by 3.8% and 5.4%. All 2,394 local/public recordings match the baseline. Peak process memory remains within the observed baseline range.

## Changes

The implementation is in the [shared Conformer graph builder](../vendor/transcribe-cpp-sys/src/conformer/conformer.cpp) and [Canary encoder policy](../vendor/transcribe-cpp-sys/src/arch/canary/encoder.cpp).

1. **Read relative-position scores through a view.** The old shift padded a zero column, reshaped, copied and cropped the score tensor. `rel_shift_view` directly addresses the required values: `out[k,q] = input[k-q+Tq-1,q]`. It preserves head and batch strides, including rectangular query/key lengths. Local-attention padding and masks remain in place. Manual attention reads the view directly; flash attention still materializes the contiguous bias it needs. This removes shift intermediates, not the underlying quadratic position-score matrix.
2. **Keep position-attention head splits as views.** The `Q_v` and projected-position `P` tensors no longer receive a contiguous copy after splitting heads. Their matrix multiplication reads the existing strides. Projection weights, precision and arithmetic expressions remain unchanged, although different layouts can affect backend dispatch.
3. **Feed adjacent channels directly to Metal depthwise convolution.** After the first pointwise convolution and GLU, Canary supplies a `[T,1,D,B]` view of its `[D,T,B]` activation. This skips the input transpose copy and selects Metal's existing channel-tiled kernel. The singleton-height stride keeps output in the ordinary time-contiguous layout; a defensive copy handles an unexpected non-contiguous output before reshaping. The optimization requires direct pointwise/depthwise paths, symmetric padding, no convolution mask and no streaming time cache. Other cases retain the existing path.

The depthwise policy is enabled only by standard Canary on Metal. The shared relative-position changes also affect Conformer callers such as Parakeet, Cohere and Canary-Qwen. Their full-model accuracy and performance are not established by the standard-Canary corpus results. No model weights, quantization, attention precision, context limits, frontend values or app session lifetime change.

## Measurement method

The hardware and runner match the [previous report](canary-performance.md#measurement-setup): Apple M5 Pro, 24 GB unified memory, macOS 26.5, Release arm64 builds, explicit Metal, default F16 KV and the existing flash policy. Normal candidate comparisons use the same precompiled Metal shader library. Each recording gets a fresh session while the model stays loaded. Builds and inference jobs run sequentially.

`elapsed_ms` includes session initialization and inference. It excludes WAV reading, model loading, result serialization and session destruction. `encode_ms` measures synchronized encoder graph computation only. Encoder graph construction, scheduler allocation, positional-table generation/upload and output readback sit outside that timer. Cross-KV setup and computation are included in `decode_ms`.

Screening uses 61 representative recordings. The full local set has 1,594 supported recordings with identical fixed English/Russian hints; the previously rejected 170-sample recording remains excluded. Public validation uses the same 400-file samples from each LibriSpeech test-clean and test-other split. Local recordings have no manually verified reference text, so they measure output parity rather than WER.

Artifacts remain under ignored `artifacts/canary-speed/`. Successful warm medians omit the first recording; totals include all recordings. Physical footprint is sampled through `proc_pid_rusage` every 50 ms. These are sequential measurements with warm filesystem caches, not a controlled thermal or system-memory experiment.

## Results

Ranges span two complete passes per build. The first pair is `validation/summary.json`, rows `all-baseline` and `all-tiled`. The reversed-order pair is in `final-validation/summary.json`. Public validation and rejected-candidate runs occur between the pairs. The first candidate pass predates a defensive contiguity guard that adds no graph operation for the tested output layout; the second uses the final implementation.

| Metric, 1,594 local recordings | Baseline | Candidate |
| --- | ---: | ---: |
| Warm median session/inference | 61.04 to 62.58 ms | 59.19 to 59.69 ms |
| Median encoder compute | 34.93 to 35.30 ms | 33.21 to 33.29 ms |
| Median decoder time | 20.62 to 21.17 ms | 20.52 to 20.57 ms |
| Total session/inference time | 105.20 to 107.84 s | 101.24 to 101.99 s |
| Peak process physical footprint | 204.19 to 211.83 MiB | 205.66 to 211.36 MiB |

Median encoder compute fell 4.9% and 5.7%; median session/inference time fell 3.0% and 4.6%. Process footprint varied between runs, with overlapping ranges. This update does not establish another process-memory reduction. Mapped model pages still occupy unified RAM; process footprint does not include all system memory occupied by those file pages.

A separate diagnostic build measured encoder scheduler buffers immediately after graph allocation. Metal's reserved buffer size fell by 385,024 bytes for 30 seconds, 832,512 bytes for 65 seconds and 1,229,824 bytes for 96 seconds. CPU scheduler buffers were unchanged. Short-input reservations also fell. These are allocator buffer sizes, not resident-RAM measurements. Encoder graph nodes fell from 3,096 to 2,808; some removed nodes are views, so this is not a GPU-launch count. The diagnostic logging exists only in local benchmark artifacts.

The longer public clips benefited more. An initial 30/65/96-second sequence followed by three short recordings took 2.22 s before and 1.77 s after. Repeating the sequence three times in reversed build order gave 5.81 s before and 5.20 s after, a 10.4% reduction. Both tests reuse one session within each process. The difference between the first sequence and repeated results shows why the initial 20% result should not be generalized.

| Public clip | Baseline median, 3 repeats | Candidate median, 3 repeats | Reduction |
| --- | ---: | ---: | ---: |
| 30 seconds | 276.36 ms | 257.86 ms | 6.7% |
| 65 seconds | 605.12 ms | 534.45 ms | 11.7% |
| 96 seconds | 829.62 ms | 683.60 ms | 17.6% |

These are three concatenated public-speech inputs, not a representative long-audio corpus. All 18 repeated results match. Repeated-sequence peak process footprint was 252.95 MiB before and 251.10 MiB after.

The final build matched text, raw text, language, segments, status and truncation flags on all 1,594 local recordings and both 400-file public samples. Every result returned success. Output parity does not establish bitwise equality of every encoder intermediate or token-ID equality for arbitrary inputs.

| Public sample | Files | Reference words | Word errors, both builds | WER, both builds |
| --- | ---: | ---: | ---: | ---: |
| LibriSpeech test-clean | 400 | 8,177 | 141 | 1.724% |
| LibriSpeech test-other | 400 | 7,347 | 242 | 3.294% |

Identical hypotheses retain the previous scores and pinned normalizer described in the earlier report. These are samples, not full-split results.

The final real-model lifecycle fixture matched all 26 baseline output rows. It covers fresh/reused sessions, both mixed-length batch orders, cancellation recovery, context limits 63/64/65/127/128/129 and five model reloads. Further paired comparisons matched on 30/65/96-second audio followed by three short recordings, F32 KV and forced flash on those six inputs, and three inputs each for private Metal, CPU and Russian-to-English translation. A 300-second input at the default context reached the same 512-token limit and truncation status; the original 170-sample recording returned the same invalid-argument status.

The Rust library suite passed 21 tests, with the existing optional Metal shutdown test ignored. The release app built and was installed locally. Its binary and shader hashes match the signed distribution bundle, and strict signature verification passed. The generated Metal shader source matches the benchmark build. A benchmark linked against the exact release native library and using the installed shader matched all 61 screening recordings.

## Focused native checks

The [Conformer fixture](../tests/canary-native/conformer.cpp) compares old and new graph operations bit for bit:

- 75 relative-shift shapes per backend, including square and rectangular attention, one query or key, boundaries around 8/16/32/64/128, multiple heads and batches. It checks an independent indexing oracle, direct contiguous copies and both addition operand orders, including negative infinity.
- 40 position-matmul shapes per backend, with head dimensions 32/128, sequence boundaries, multiple heads and batch broadcasting.
- 40 depthwise layouts per Metal mode, with kernels 9/31, lengths 1/7/32/65/129, channels 32/1024 and batch sizes 1/2.

Each shape uses two input passes. CPU, shared Metal and private Metal cover shift/matmul; depthwise comparisons run in both Metal modes. The fixture explicitly assigns graph nodes to the selected backend and verifies support, so CPU fallback cannot conceal a broken Metal stride path. These operation-level checks and existing loader/mapping checks passed against the final native library. The wrapper also supports the optional frontend comparison:

```bash
python3 scripts/test-whisper-native.py NATIVE_PREFIX --sdk "$SDKROOT" \
  --baseline-source BASELINE_SOURCE/vendor/transcribe-cpp-sys
```

## Experiments left out

A candidate that additionally enabled direct depthwise convolution in the pre-encoder changed one local transcript and one LibriSpeech test-other transcript. It had passed the 61-file screening set, showing why that screen was insufficient. The additional path was removed; `direct_dw_in_pre_encode` remains false. A changed transcript is not automatically less accurate, but it fails the intended parity gate and was not accepted on speed alone.

Forcing encoder flash attention, disabling Metal concurrency and disabling Metal tensor operations did not justify changing defaults in the small screening run. Forced flash and disabled tensor operations were slower; disabling concurrency did not show a convincing gain. The tensor-disabled startup also compiled a different shader path, so its load time is not comparable to precompiled-library loading.

## Next larger pipeline change

Canary still copies encoder output to host and uploads it into a separate cross-KV graph. Appending cross-KV projections to the encoder graph could remove those copies, staging storage, one graph allocation and one synchronization. It must initialize the persistent KV cache before allocation and preserve peak-memory behavior. Passing the old encoder tensor across a scheduler reset is unsafe because its allocation can be reused.

The embedding is only about 1.47 MiB for 30 seconds of audio, so copy bandwidth alone is unlikely to explain a large speedup. Profile graph setup and synchronization separately before committing to this change. Retaining sessions across recordings would change idle memory and is outside this update.
