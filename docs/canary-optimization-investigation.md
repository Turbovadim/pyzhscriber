# Canary GPU optimization investigation

Investigated September 11, 2026, against commit `3ce6e1e12e9a1a8bf7a1e720dbee076be756f9ac`.

This document records the investigation before implementation. The subsequent changes and measured results are in [Canary performance](canary-performance.md). Descriptions of the current implementation below refer to the baseline commit above; source line links may have shifted after implementation.

The investigation consisted of a source review and a read-only inspection of the installed Canary GGUF directory. No engine changes, model conversions, application settings changes, or candidate performance runs were made during that phase. The byte counts below are file measurements or calculations from tensor dimensions. They are not measured reductions in process memory or latency.

The intended deployment is Metal GPU inference on this Mac. Performance and memory priorities below target that path. CPU-only model transformations and decoder optimizations are retained as compatibility notes, not target workload requirements. Audio preprocessing still runs on the CPU before GPU inference, so its buffer and timing improvements remain relevant.

The best first memory/load-time transfer is Whisper's mapped weight loader for shared Metal buffers. The installed Canary model passes its static file-layout requirements. For GPU decoding speed, prioritize measuring smaller attention read windows and final-position prompt projection. Several smaller opportunities also exist in temporary audio storage and graph construction. Canary-Qwen needs separate treatment because its decoder, transformed weights, and cache allocation differ from standard Canary.

The original constraints remain: retain the model and decoding behavior, preserve accuracy, and keep memory usage at or below the current implementation. A smaller macOS process footprint from file-backed weights does not imply the same reduction in total occupied RAM.

## Priorities

| Path | Current state | Recommendation |
| --- | --- | --- |
| Direct weight loading without temporary tensor copies | Already shared by Canary and Canary-Qwen | Keep; do not count it as a new gain |
| Read-only mmap for shared Metal weights | Whisper only; installed Canary layout is compatible | First GPU memory/load-time candidate; preserve CPU compatibility |
| Direct weight conversion and packing | Standard Canary conversion is CPU-only; Qwen transforms also affect Metal | Secondary Qwen GPU candidate; no conversion saving for standard Canary on Metal |
| Remove duplicate FP64 pre-emphasis buffer | Both Canary frontends still allocate it | Early candidate with bitwise frontend validation |
| Normalize mel output in place | Per-feature path still allocates separate input/output matrices | Early memory candidate |
| Retain decoder intermediates only when debugging | Standard Canary retains several unconditionally | Early graph-memory candidate |
| Project only the final prompt position | Standard Canary projects all positions; Qwen already slices | Small standard-Canary candidate |
| Smaller growing decoder attention window | Standard Canary single GPU path starts at 1024; batch starts at 64 | Profile for short dictation |
| Size physical KV to reachable work | Standard Canary allocates full context; Qwen eagerly allocates 2048 | GPU cache-memory candidates; CPU graph reservation is outside the target |
| Remove encoder-output host round-trip | Present in both architectures | Investigate after simpler changes; ownership matters |
| Deliver mel in the encoder's required layout | Shared pre-encoder still transposes/materializes input | Medium effort; coordinate frontend and encoder contracts |
| Skip exactly zero FFT windows | Whisper-only fast path | Conditional opportunity; preserve per-feature normalization |
| Remove redundant encoder work | Shared attention already uses strided views; Qwen has a debug-only transpose branch | Remove only verified redundant work |
| Change encoder flash-attention policy | Disabled on Metal for standard Canary, disabled by default for Qwen | Accuracy-sensitive experiment, not a mechanical transfer |
| Core ML output backing/decoder-only loading | No Canary Core ML integration in this engine | A separate backend project |
| Batch packing and cross-run graph reuse | Batch execution exists; app uses fresh single sessions | Secondary to the actual dictation path |

## What is installed and what that means for memory

The installed model is `canary-1b-v2-Q5_K_M.gguf`, the default quantization for Canary 1B v2 in the [catalog](../resources/model-catalog.json#L372). Its GGUF header and tensor directory were read without loading the weight payload or recomputing the model checksum.

| Property | Observed value |
| --- | ---: |
| File size | 836,664,032 bytes, 797.905 MiB |
| GGUF version / tensor count | 3 / 1,476 |
| Contiguous tensor-data span | 836,194,304 bytes, 797.457 MiB |
| Tensor alignment | 32 bytes, no gaps beyond alignment padding and no overlaps |
| Page-aligned mapping on this Mac's 16 KiB pages | 836,206,592 bytes, 797.469 MiB |
| Largest tensor | Token embedding, Q6_K, 13.125 MiB |
| Frontend | 16 kHz; 512-point FFT; 400-sample window; 160-sample hop; 128 mel bins |
| Frontend arithmetic | Pre-emphasis approximately 0.97; per-feature normalization; reflect padding |
| Encoder | 32 layers, width 1024, 8 heads, subsampling factor 8 |
| Decoder | 8 layers, width 1024, 8 heads, vocabulary 16,384, context ceiling 1024 |

The loader defaults to reflect padding when the metadata field is absent. The installed model has that omission. The runtime's existing frontend behavior, including its window and normalization rules, is the baseline to preserve.

On the target Metal path, standard Canary does not create the 384 MiB of F32 pointwise-convolution replacements discussed during the initial investigation. That [promotion is restricted to a CPU primary backend](../vendor/transcribe-cpp-sys/src/transcribe-load-common.cpp#L537). Its roughly 797.5 MiB original weight span is the mmap candidate. Batch-normalization fusion adds 256 KiB; graph scratch, decoder caches, audio buffers, and resident file pages remain. This is storage accounting, not a prediction of total GPU inference memory.

For CPU compatibility only, the file contains 192 MiB of F16 pointwise-convolution weights whose F32 replacements add 384 MiB, while retaining originals. The nominal copied weight payload plus replacements is about 1,181.5 MiB. These extra allocations are not part of the intended standard-Canary GPU workload. Canary-Qwen differs: some of its conversions run on Metal too, as detailed below.

## Weight loading and mmap

Both families call [`load_common::stream_tensor_data`](../vendor/transcribe-cpp-sys/src/transcribe-load-common.cpp#L460). That helper already reads directly into ordinary CPU and shared Metal buffers. Other backends retain their staging and `tensor_set` route, including storage that needs backend-specific transformation. Canary already receives this part of the Whisper work.

The remaining transfer is [`try_map_weights`](../vendor/transcribe-cpp-sys/src/transcribe-weights-mmap.cpp#L36), subsequently moved from Whisper into shared loader code. Its implementation operates on a GGUF directory, an unallocated tensor context, and a backend. Apart from its namespace and location, it is largely model-independent. Standard Canary at the baseline commit [allocates and fills the entire context](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L491), then performs its transformations.

Proposed design:

1. Move the mapping helper into shared loader code and retain the existing Whisper behavior.
2. Before copied allocation, attempt mapping of Canary's original GGUF tensor context. Record mapping ownership in the model.
3. Keep fused batch-normalization and promoted convolution tensors in their existing separate allocations. They must never write into the read-only mapping.
4. On unsupported storage, incompatible layout, or mapping failure, use the existing copied loader. Keep partial-failure cleanup and destruction explicit: backend references must be released before unmapping their storage.

Preserve the existing checks for macOS CPU/shared Metal, device host-pointer support, default buffer compatibility, tensor type/shape/size, contiguity, alignment, file bounds, regular files, allocation size, and maximum Metal buffer size. A generic `is_host` check is insufficient because some host-accessible backends repack tensors.

The installed file passes the directory-level alignment and contiguous-span checks. Actual backend buffer creation and all per-tensor backend allocation checks have not been executed for Canary. Other model variants and quantizations still require validation. The helper deliberately falls back for incompatible layouts; broadening it to arbitrary sparse tensor spans is unnecessary for this installed file.

The app [downloads to a temporary file, verifies it, then renames it](../src/models.rs#L203). That fits the immutable-file requirement. Replacement by rename or unlink can preserve existing mappings, whereas modifying or truncating a loaded file in place is incompatible with the design. Preserve that lifetime contract for models resolved from external caches too.

Expected benefits are a smaller private weight allocation, potentially lower loading/copying cost, and less pressure from duplicate storage. First inference may still fault pages in. Report model-load time, first-result time, warm inference, process physical footprint, RSS, and system memory separately. Do not apply Whisper's measured percentage to Canary.

Canary-Qwen's [loader](../vendor/transcribe-cpp-sys/src/arch/canary_qwen/model.cpp#L657) can use the same mechanism, but mapping covers only original weights. Its separate convolution promotions, CPU BF16 linear promotions, and packed gate/up tensors remain. Packing already reduces decoder operations while adding another copy of those weight rows. Avoiding those copies would require a separate storage/packing design, with attention to mixed tensor types and the current fallback behavior. No installed Qwen file was inspected, so its exact mapped span and residual allocations are unverified. A large Qwen asset can also pass the file-layout checks yet exceed Metal's single-buffer limit and require copied loading.

There are two follow-up loading paths beyond mmap. First, the [F16-to-F32 conversion](../vendor/transcribe-cpp-sys/src/transcribe-load-common.cpp#L593) copies through both a source and a destination staging vector. For the installed model on CPU their largest combined payload is 12 MiB; this is not an allocation to eliminate on its Metal path. Compatible source/destination buffers could allow the same conversion directly into its final allocation, preserving staged access elsewhere. Qwen's GPU conversions and [gate/up packing](../vendor/transcribe-cpp-sys/src/causal_lm/causal_lm.cpp#L811) have related copying opportunities. These affect loading, not repeated decoder arithmetic.

Second, original tensors that have been replaced cannot simply be freed individually from a full-model allocation. Avoiding their allocated copies requires partitioned loading or mapped sources plus separate replacement buffers. For Qwen, any zero-copy packed view additionally requires compatible tensor types and verified physical adjacency without interleaved data or padding. CPU BF16 assets can retain original, converted, and packed representations; quantify each before redesigning ownership. Preserve Qwen's F16 convolution promotions on all backends: the source records correctness problems that motivated them.

## Audio preprocessing

Standard Canary [requires a power-of-two FFT and per-feature normalization](../vendor/transcribe-cpp-sys/src/arch/canary/weights.cpp#L345). Both families request channel-major output. Whisper's PCM borrowing, direct frame-major output, and zero-window skipping are gated to a different frontend path in [`transcribe-mel.cpp`](../vendor/transcribe-cpp-sys/src/transcribe-mel.cpp#L385). Sharing that file does not activate those optimizations for Canary.

### Remove the extra pre-emphasis allocation

The power-of-two path builds `emph[N]` as doubles, then copies it into a second padded double buffer. See [the current allocation and padding loops](../vendor/transcribe-cpp-sys/src/transcribe-mel.cpp#L441).

The simplest adaptation is to write pre-emphasized samples directly into the interior of the padded buffer. Retain the exact double expression, the first sample rule, and reflection of the already pre-emphasized sequence. This removes one `8 × N` byte allocation and one full copy. At 16 kHz that allocation is about 0.122 MiB per second, or 3.66 MiB for 30 seconds.

Borrowing raw PCM and calculating pre-emphasis as each FFT window is read could also remove the padded double buffer. It is more involved: overlapping windows repeat work, boundary handling must preserve the current sequence, and changing floating-point contraction or rounding can change results. Start with eliminating `emph`, then measure whether full borrowing is worthwhile.

### Remove the duplicate mel matrix

The frontend writes raw log-mel into a float matrix and then [normalizes into another equally sized matrix](../vendor/transcribe-cpp-sys/src/transcribe-mel.cpp#L917). For channel-major output, a bin can be normalized in place after both of its statistics passes finish. Transfer ownership of that storage to the caller, or use caller-owned storage with safe input/output alias handling.

Keep FP64 sums, variance accumulation order, Bessel correction, epsilon, float storage conversion, and the distinction between constant and reflect padding. Reflect-padded Canary retains the final frame. The source records a transcription regression from masking it incorrectly. Also preserve reused-vector capacity behavior; swapping buffers without considering the old output can retain or repeatedly allocate unnecessary capacity.

One duplicate matrix is `4 × 128 × frame_count` bytes. With the installed hop length, this is about 0.049 MiB per second, or 1.47 MiB for a 30-second recording. This and the pre-emphasis allocation disappear at different stages, so their sizes must not simply be added and presented as a measured peak reduction.

### Feed the encoder its native input layout

The shared pre-encoder [permutes channel-major mel and materializes it](../vendor/transcribe-cpp-sys/src/conformer/conformer.cpp#L966) before convolution. The per-feature normalization loop already knows how to write frame-major output. An explicit frontend/encoder layout contract could put frequency-contiguous data into the input tensor directly and remove that GPU copy.

This requires coordinated changes to tensor dimensions, pre-encoder handling, batch input packing, and debug dump shapes. Merely changing a shape label would swap time and frequency in a spatially asymmetric convolution. Preserve the default layout for other Conformer families. This is a separate choice from channel-major in-place normalization; investigate their combined storage plan before assuming both savings can be obtained independently.

### Zero windows, power storage, and threads

The power-of-two path uses double-precision FFT arithmetic, writes a full F32 power matrix, and calls one [SGEMM for the mel filterbank](../vendor/transcribe-cpp-sys/src/transcribe-mel.cpp#L629). A safe zero-window experiment would detect exact zeros after pre-emphasis, reflection, and windowing, then skip the FFT while preserving the same power/log/normalization results. Skipping filterbank work too requires more restructuring because SGEMM currently spans every frame. Check finite filters and nonfinite inputs explicitly. Silence thresholds or dropping frames would change recognition behavior and are outside this transfer.

The power matrix occupies `4 × 257 × frame_count` bytes for this model, about 2.94 MiB at 30 seconds. Tiling FFT/filterbank work could bound that allocation, but smaller SGEMM shapes can change performance and floating-point reduction behavior. Treat it as a later experiment, not an exact copy removal.

Standard Canary's [single-run mel call](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L707) omits the session thread count; Qwen passes it. Forwarding the setting is a small candidate, followed by a thread-count sweep. Measure short and longer clips, and avoid oversubscribing frontend workers and BLAS. Persisting FFT setup or worker scratch is another tradeoff to measure only if setup is material; retained caches count against the memory constraint.

## Encoder computation

The shared Conformer already uses [strided Q/K/V views](../vendor/transcribe-cpp-sys/src/conformer/conformer.cpp#L602). Its remaining contiguous Q-position and position-projection tensors serve matrix multiplication, so Whisper's Q-view change is not an absent optimization to copy. Canary also already selects direct in-block depthwise convolution and pointwise matrix-multiply paths. Removing their layout operations blindly would change operation requirements.

One concrete Qwen-only redundancy remains. [`canary_qwen/encoder.cpp`](../vendor/transcribe-cpp-sys/src/arch/canary_qwen/encoder.cpp#L195) creates a permute-plus-contiguous copy solely for the `enc.final` debug dump and explicitly expands that branch into the production graph. The perception projection consumes the original tensor. Constructing and expanding this branch only when debugging is enabled can remove one copy and allocation. Preserve the dump's existing shape when enabled. Standard Canary does not have this same side branch.

Flash attention deserves a fresh experiment, with a clear accuracy boundary. [Standard Canary disables encoder flash on Metal](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L536); [Qwen disables it by default on all backends](../vendor/transcribe-cpp-sys/src/arch/canary_qwen/model.cpp#L716). The current [Metal support check accepts head dimensions 96 and 128](../vendor/transcribe-cpp-sys/ggml/src/ggml-metal/ggml-metal-device.m#L1297), despite the [older header rationale](../vendor/transcribe-cpp-sys/src/arch/canary/canary.h#L137). This removes one presumed blocker, not every backend constraint.

The Conformer flash path [casts relative-position bias to F16](../vendor/transcribe-cpp-sys/src/conformer/conformer.cpp#L705) and can also narrow K/V activations. Even an explicit F32 KV setting does not eliminate the bias cast. Relative-position score construction still materializes quadratic storage in sequence length, so enabling flash does not remove every large attention tensor.

Use existing `TRANSCRIBE_FORCE_FLASH` and `TRANSCRIBE_NO_FLASH` only as controlled process-local experiments. They affect both stages; record the actual encoder and decoder configuration and backend placement. Compare intermediate differences, token decisions, reference accuracy, latency, and peak scratch. A working kernel alone is insufficient evidence to change the default.

Whisper's Core ML output backing and decoder-only weight loading have no current Canary counterpart. The [build integration](../vendor/transcribe-cpp-sys/src/CMakeLists.txt#L142) and [conversion script](../scripts/build-whisper-coreml-encoder.sh) target Whisper. Canary would need a model export, matching frontend and variable-length contract, packaging, backend selection, fallback, and accuracy validation. It is a separate project with uncertain speed and memory results.

## Decoder and cache changes

### Existing work that should not be repeated

Standard Canary [computes cross-attention K/V once](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L980) before prompt decoding. It has no separate language-detection decoder pass. Language and task are prompt tokens supplied from caller settings. Whisper's restricted language-head projection and duplicate cross-projection removal do not transfer here.

Normal decoding already skips vocabulary softmax and reads a backend argmax result. GPU generation reuses one static graph within an utterance. CPU generation deliberately rebuilds with the current attention length. Decoder KV defaults to F16. Quantizing caches, changing output limits, or lowering precision further would be new accuracy tradeoffs.

### Retain only necessary graph outputs

The standard Canary [prompt/dynamic decoder builder](../vendor/transcribe-cpp-sys/src/arch/canary/decoder.cpp#L380) marks embeddings and selected sublayer intermediates as graph outputs even outside debugging. The [allocator](../vendor/transcribe-cpp-sys/ggml/src/ggml-alloc.c#L645) preserves output storage and restricts its reuse.

Use the existing conditional [`debug::mark_tensor_for_dump`](../vendor/transcribe-cpp-sys/src/transcribe-debug.cpp#L180) for debug-only values, while preserving real outputs and cache writes. This is a small implementation candidate whose actual saving depends on graph lifetimes. It mainly affects prompt and CPU dynamic graphs; the static GPU step graph is separate. Validate with debugging both enabled and disabled because memory reuse and backend fusion can change.

### Slice the prompt head, then consider final-layer work

The standard decoder [projects every prompt position and then selects the last logits row](../vendor/transcribe-cpp-sys/src/arch/canary/decoder.cpp#L477). Slice the final hidden position before the vocabulary head in non-debug runs. All prompt positions must still populate the transformer cache.

Prompt construction produces 5 tokens for Canary1, 9 for aggregate-tokenizer Canary2, and 10 for the installed single-SentencePiece Canary2. For its 16,384-word vocabulary, the logits payload would fall from 640 KiB to 64 KiB. That saves nine head rows, not 90% of total inference. Matrix shape changes can select different kernels and rounding, so exact token/output and reference-accuracy comparisons are required.

A later extension could slice after the final layer's self-KV writes, avoiding earlier prompt rows in that layer's cross-attention and FFN. The short prompt limits the possible benefit, while more kernel shapes change. Qwen already performs last-position head projection and non-debug final-layer FFN slicing, so those are not new Qwen wins.

### Shrink the attention read window without changing the context limit

The [single GPU loop](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L1190) uses a power-of-two attention window with a 1024-slot floor, clamped to the cache capacity. It reads masked positions beyond the generated prefix. The [batch loop](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L1714) already starts at 64 and rebuilds as the prefix grows.

Adapt that approach to single-recording decoding and measure several initial window sizes. Keep actual keys, values, masks, generation budget, context ceiling, and cancellation behavior intact. Smaller windows save attention work but introduce occasional graph rebuilds. The current source documents a dispatch advantage for some larger shapes, so the crossover must be measured on the target hardware.

This changes how much KV is read, not automatically how much KV is allocated. It also changes attention reduction shapes, which need accuracy checks. Test every growth boundary and long outputs, not just typical short dictation.

### Separate physical capacity from semantic limits

Standard Canary [allocates self-KV to the full context ceiling](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L929), while the [current generation budget](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L1096) is at most 512 new tokens. A physical capacity based on reachable positions, or a growing allocation, could save memory without lowering the public context limit. This requires separating capacity from the limit currently used to calculate output length, and ensuring every padded graph read fits the allocation.

For the installed model, F16 self-KV occupies 32 MiB at 1024 positions. Its cross-KV uses 32 KiB per encoder frame. A 30-second input produces approximately 376 encoder frames, or 11.75 MiB of cross-KV. F32 doubles these payloads. A power-of-two capacity covering a 10-token prompt plus 512-token generation still rounds to 1024, so simple rounding alone saves nothing here. Actual savings require a different capacity/growth plan coordinated with the read-window change.

Outside the GPU target, CPU also [reserves graph scratch for the full cache ceiling](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L1274). Bounding that reservation to the reachable worst step is a CPU-only candidate. On either backend, the current combined self/cross cache is freed when encoder length changes; splitting their lifetimes could help reused library sessions, but adds ownership complexity and needs a strict retained-memory bound.

For Canary-Qwen, [session initialization eagerly allocates 2048 positions](../vendor/transcribe-cpp-sys/src/arch/canary_qwen/model.cpp#L730). Its [run path calculates a requirement starting at 1024 and only grows an existing cache](../vendor/transcribe-cpp-sys/src/arch/canary_qwen/model.cpp#L931). Deferring initial allocation until the prompt is known is a stronger first cache candidate. Its prompt includes audio embeddings, so capacity and read windows must include audio plus text plus the existing 256-token generation reserve. Short text alone does not establish a small required cache.

### Remove encoder-output staging carefully

Standard Canary [downloads encoder output to `enc_host`](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L854) and uploads it again for cross-KV projection. The staging vector is `4 × decoder_width × encoder_frames` bytes, about 1.47 MiB at 30 seconds for the installed model.

Two designs are possible: a persistent backend output tensor shared across the stage boundary, or cross-KV projection/cache writes appended to the encoder graph. Passing the existing tensor pointer directly is unsafe: the next stage frees its compute context and resets scheduler allocations. A combined graph may increase simultaneous scratch lifetimes, so removal of the copy must not raise peak memory. Shared Metal still needs correct ownership and synchronization even though CPU/GPU share physical RAM.

Qwen has a similar host round-trip, but its encoder output is injected into a causal prompt rather than a cross-attention cache. Preserve that distinction when designing a bridge.

### Sampling and smaller decoder candidates

Whisper's CPU probability and timestamp processing has no matching standard-Canary hot path. Qwen does read its first vocabulary vector to the CPU and [select the first strict maximum](../vendor/transcribe-cpp-sys/src/arch/canary_qwen/model.cpp#L1061). Later steps use backend argmax. Moving the first selection to the GPU could remove a transfer, but the existing [ggml CPU](../vendor/transcribe-cpp-sys/ggml/src/ggml-cpu/vec.h#L1558) and [Metal](../vendor/transcribe-cpp-sys/ggml/src/ggml-metal/ggml-metal.metal#L3033) argmax tie behavior does not generally match that first-index scan. Any replacement needs explicit tie and NaN tests; ordinary speech samples cannot prove equivalence.

The standard GPU loop also uploads the entire F16 mask each token, although only one entry becomes valid. An incremental mask update is possible only if scheduler input storage persists across runs. At 1024 positions the current upload is about 2 KiB per step, so dispatch/synchronization may dominate the saved bytes. Give this lower priority than the attention window itself.

## Batching and the application path

Canary already has parallel mel processing, serial encoder execution, and batched autoregressive decoding. Increasing batch size consumes more KV and padded intermediate storage. It improves throughput under some workloads, not necessarily interactive latency at the same memory budget.

Within the existing [batch path](../vendor/transcribe-cpp-sys/src/arch/canary/model.cpp#L1556), potential savings include releasing each mel buffer after encoding, releasing individual host embeddings after packing, and avoiding duplicate padded host staging. Grouping similar lengths can reduce padding, but must preserve request/output order and latency requirements. The shared batch decoder also evaluates heads for prompt positions whose predictions are discarded; a specialized prompt path could avoid those heads, at the cost of extra graph setup and effects on other families using the helper.

Cross-run graph caching and separate self/cross cache lifetimes matter mainly for long-lived sessions. The app [creates and drops a session for each transcription](../src/transcription.rs#L108), releasing KV and scheduler scratch. Retaining sessions would change idle memory and cannot be assumed to satisfy the original constraint.

There is another app-specific cost. [Language detection](../src/transcription.rs#L193) uses Whisper Tiny, while [`ensure_loaded`](../src/transcription.rs#L205) retains only one model. With multiple allowed languages for standard Canary, or multiple language routes, a recording can load the detector and then reload Canary. With an explicit single language and a stable selection, Canary can remain loaded. This makes mmap's loading behavior potentially relevant to every recording in routed use.

Benchmark those modes separately. Keeping both models loaded could reduce switching latency but increase idle memory; it is not part of the proposed same-memory transfer. Skipping language detection or guessing from the previous utterance would change routing accuracy. Empty language hints also need attention in benchmarks: standard Canary's engine defaults to English, rather than doing Whisper-style internal language detection.

## Validation plan and acceptance criteria

The existing [`whisper_bench` example](../examples/whisper_bench.rs#L134) explicitly rejects non-Whisper models and supplies Whisper family options. It cannot establish a Canary baseline unchanged. A future benchmark adaptation should use generic Canary run options and report the actual architecture, quantization, backend, thread count, KV type, language, PNC, translation, and session lifetime. No benchmark adaptation was implemented for this investigation.

1. **Freeze the baseline.** Record the source commit, build flags, SDK, model checksum, native libraries and Metal shader identity. Begin with installed Canary 1B v2 Q5_K_M on Metal. Verify actual encoder/decoder operation placement so CPU fallback cannot masquerade as GPU performance. Add Canary1 and aggregate Canary2 coverage before generalizing across the family; validate Qwen independently. Use CPU for focused shared-code and fallback correctness checks, not as a required performance campaign.
2. **Separate workloads.** Measure model loading, first inference, warm inference with fresh sessions, reused sessions, and the app's detector/model-switch sequence. Include short dictation, longer supported inputs, silence/noise, supported languages, PNC modes, and supported translation directions. Use each variant's reported limits and actual input gates; do not reuse Whisper's two-hour chunked test as though Canary had the same long-form engine.
3. **Measure memory during each stage.** Capture peak process physical footprint, RSS, model buffers, transformed weights, KV bytes, scheduler allocations, and retained memory after session/model release. Include first-touch page-ins for mmap. A model file's byte count is not a measurement of global RAM saved. Sample fast enough to observe short allocation peaks, or supplement sampling with allocator measurements.
4. **Isolate timing comparisons.** Run baseline/candidate/candidate/baseline sequentially with identical inputs and settings, without compilation or competing inference. Report per-stage timings, p50/p95 latency and total time. Keep debug/profiling output disabled for ordinary timing. Treat cold file-cache behavior separately without purging system caches during normal use.
5. **Prove behavioral preservation.** For storage-only changes, compare loaded tensor bytes and complete recognition outputs. For frontend copy/layout work, require bitwise mel comparisons against the unchanged frontend, including short boundaries, reflection, constant padding for shared callers, aliasing, reused buffers, silence, nonfinite inputs, and thread settings. For kernel-shape or attention-policy changes, compare token decisions, text/raw text, result status, partial results, and fixed-reference WER. Investigate every changed output; unchanged weights alone do not establish unchanged accuracy.
6. **Exercise lifetimes and limits.** Focus on shared Metal and its copied-loading fallback, debug on/off, F16/F32 KV, context and read-window boundaries, model load/unload, cancellation followed by reuse, short after long, and batch/single transitions. Keep focused CPU/private-Metal compatibility fixtures for shared loader changes. mmap checks need malformed/truncated metadata fixtures, incompatible layouts, buffer-wrap failure cleanup, and file replacement/unlink while loaded. Do not truncate a real mapped model as a lifecycle test.
7. **Accept per change.** Require no recognition or accuracy regression and no peak or retained-memory increase on its target workloads. Investigate every changed output; reject behavior changes outside the agreed scope. Keep an optimization only when its measured benefit justifies the code and validation burden. Run shared-frontend and Whisper checks when changing shared code.

## Suggested sequence and effort

These are rough engineering estimates from static inspection, not scheduled commitments. Hardware/model validation can take longer than the edits.

| Work package | Approximate effort | Decision gate |
| --- | --- | --- |
| Adapt measurement tools and capture Metal Canary baselines | Half to one day | Actual GPU placement and reliable stage, output and memory comparisons |
| Shared mmap helper plus standard Canary Metal ownership/fallback | Several hours of edits, one to two days with focused backend/corpus validation | Actual backend compatibility and lifecycle checks pass |
| Duplicate PCM/mel storage and conditional debug retention | One to two days with isolated validation | Bitwise frontend results and lower measured allocations |
| Final prompt head on Metal | Half to one day | Same recognition decisions; repeatable benefit |
| Growing attention window and physical cache design | One to three days | Boundary correctness, latency crossover and memory proven |
| Encoder-output bridge and coordinated mel layout | One to three days per path | Copy savings survive peak-scratch and accuracy checks |
| Qwen-specific mmap/transforms, lazy KV and debug branch | Separate one-to-three-day pass once a model baseline is available | Qwen storage, prompt and output contracts validated |
| Flash policy, tiled frontend, deeper final-layer pruning | Several days of experiments, outcome uncertain | Accuracy and memory constraints survive changed kernels |
| Canary Core ML backend | Separate design and estimation required | Export feasibility, accuracy and real device measurements |

Start with the installed model's shared Metal mmap path and GPU measurement support. Evaluate smaller decoder attention windows and prompt projection for latency, then address verified duplicate storage and stage transfers. CPU-only conversion and reservation work is outside this target. Keep attention-policy and backend changes isolated so their numerical effects cannot be hidden inside a broad optimization patch.
