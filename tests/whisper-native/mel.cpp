// Bitwise comparison against a separately compiled baseline MelFrontend.
// Link mel-baseline.cpp and baseline transcribe-mel.cpp compiled with
// -DMelFrontend=BaselineMelFrontend. --full adds 30/65-second cases;
// --bench [repeats] measures padded 30-second windows with 1/8 threads.
#include "transcribe-mel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace transcribe {
// Every case passes an explicit thread count. Keep this model-free test
// independent of the ggml-backed batch helpers.
int default_n_threads(int cap) { return std::min(cap, 8); }
}

extern "C" void * mel_baseline_create(const transcribe::MelConfig * cfg);
extern "C" void mel_baseline_destroy(void * frontend);
extern "C" transcribe_status mel_baseline_compute(void * frontend, const float * pcm, size_t n_samples,
                                                 std::vector<float> * mel, int * n_mels, int * n_frames,
                                                 int n_threads);

using Clock = std::chrono::steady_clock;

transcribe::MelConfig whisper_config() {
    transcribe::MelConfig cfg;
    cfg.n_fft = cfg.win_length = 400;
    cfg.pre_emphasis = 0.0f;
    cfg.window_type = "hann_periodic";
    cfg.normalize = "per_utterance";
    return cfg;
}

std::vector<float> waveform(size_t length, size_t voiced, float scale = 0.7f) {
    std::vector<float> pcm(length, 0.0f);
    std::mt19937 random(1977);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (size_t i = 0; i < std::min(length, voiced); ++i) {
        pcm[i] = scale * (0.5f * std::sin(static_cast<float>(i) * 0.13f) + 0.2f * uniform(random));
    }
    return pcm;
}

bool compare(const char * name, const transcribe::MelConfig & cfg, const std::vector<float> & pcm,
             int threads, int repeats = 1) {
    auto baseline = std::unique_ptr<void, decltype(&mel_baseline_destroy)>(
        mel_baseline_create(&cfg), &mel_baseline_destroy);
    transcribe::MelFrontend candidate(cfg);
    std::vector<float> expected, actual;
    int expected_mels = 0, expected_frames = 0, actual_mels = 0, actual_frames = 0;
    std::vector<double> old_times, new_times;
    for (int round = 0; round < repeats; ++round) {
        transcribe_status before_status, after_status;
        auto run_old = [&] {
            const auto start = Clock::now();
            before_status = mel_baseline_compute(baseline.get(), pcm.data(), pcm.size(), &expected,
                                                  &expected_mels, &expected_frames, threads);
            old_times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        };
        auto run_new = [&] {
            const auto start = Clock::now();
            after_status = candidate.compute(pcm.data(), pcm.size(), actual, actual_mels, actual_frames, threads);
            new_times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        };
        if (round & 1) { run_new(); run_old(); } else { run_old(); run_new(); }
        if (before_status != after_status || expected_mels != actual_mels || expected_frames != actual_frames ||
            expected.size() != actual.size()) {
            std::fprintf(stderr, "FAIL %s: status or dimensions differ\n", name);
            return false;
        }
        if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) != 0) {
            for (size_t i = 0; i < expected.size(); ++i) {
                if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "FAIL %s: index=%zu baseline=%a candidate=%a\n", name, i, expected[i], actual[i]);
                    return false;
                }
            }
        }
    }
    std::sort(old_times.begin(), old_times.end());
    std::sort(new_times.begin(), new_times.end());
    std::printf("PASS %-34s samples=%zu threads=%d floats=%zu baseline_us=%.1f candidate_us=%.1f speedup=%.2fx\n",
                name, pcm.size(), threads, expected.size(), old_times[repeats / 2], new_times[repeats / 2],
                old_times[repeats / 2] / new_times[repeats / 2]);
    return true;
}

bool compare_frame_major_padding() {
    const auto cfg = whisper_config();
    auto baseline = std::unique_ptr<void, decltype(&mel_baseline_destroy)>(
        mel_baseline_create(&cfg), &mel_baseline_destroy);
    transcribe::MelFrontend candidate(cfg);
    struct Case { const char * name; std::vector<float> pcm; size_t min_samples; };
    auto nan_input = waveform(513, 513);
    nan_input[257] = std::numeric_limits<float>::quiet_NaN();
    auto infinite_input = waveform(513, 513);
    infinite_input[1] = std::numeric_limits<float>::infinity();
    infinite_input.back() = -std::numeric_limits<float>::infinity();
    const std::vector<Case> cases{
        {"one_sample", {0.625f}, 480000},
        {"signed_zero", std::vector<float>(513, -0.0f), 480000},
        {"short_signal", waveform(16000, 16000), 480000},
        {"full_window", waveform(480000, 480000), 480000},
        {"partial_final_window", waveform(480321, 480321), 0},
        {"nan_input", nan_input, 480000},
        {"infinite_input", infinite_input, 480000},
        {"reuse_after_nonfinite", {0.0f}, 480000},
    };
    // Reuse one output across different audio, growing/shrinking frame
    // counts, and a finite input after NaN/Inf inputs.
    std::vector<float> expected, actual;
    int expected_mels = 0, expected_frames = 0, actual_mels = 0, actual_frames = 0;
    for (const auto & item : cases) {
        std::vector<float> padded(std::max(item.pcm.size(), item.min_samples), 0.0f);
        std::copy(item.pcm.begin(), item.pcm.end(), padded.begin());
        const auto before = mel_baseline_compute(baseline.get(), padded.data(), padded.size(), &expected,
                                                 &expected_mels, &expected_frames, 8);
        const auto after = candidate.compute(item.pcm.data(), item.pcm.size(), actual, actual_mels, actual_frames,
                                              8, transcribe::MelLayout::FrameMajor, item.min_samples);
        if (before != TRANSCRIBE_OK || after != before || expected_mels != actual_mels ||
            expected_frames != actual_frames || expected.size() != actual.size()) {
            std::fprintf(stderr, "FAIL frame_major_%s: status or dimensions differ\n", item.name);
            return false;
        }
        for (int t = 0; t < actual_frames; ++t) {
            for (int m = 0; m < actual_mels; ++m) {
                const float reference = expected[static_cast<size_t>(m) * expected_frames + t];
                const float value = actual[static_cast<size_t>(t) * actual_mels + m];
                if (std::memcmp(&reference, &value, sizeof(float))) {
                    std::fprintf(stderr, "FAIL frame_major_%s: frame=%d mel=%d baseline=%a candidate=%a\n",
                                 item.name, t, m, reference, value);
                    return false;
                }
            }
        }
        std::printf("PASS frame_major_%s samples=%zu pad_to=%zu output_reused\n", item.name, item.pcm.size(), item.min_samples);
    }

    for (size_t offset : {size_t(0), size_t(17)}) {
        auto aliased = waveform(4096 + offset, 4096 + offset);
        const auto before = mel_baseline_compute(baseline.get(), aliased.data() + offset, 4096, &expected,
                                                 &expected_mels, &expected_frames, 8);
        const auto after = candidate.compute(aliased.data() + offset, 4096, aliased, actual_mels, actual_frames,
                                              8, transcribe::MelLayout::FrameMajor);
        if (before != TRANSCRIBE_OK || after != before || expected_mels != actual_mels ||
            expected_frames != actual_frames || expected.size() != aliased.size()) return false;
        for (int t = 0; t < actual_frames; ++t) {
            for (int m = 0; m < actual_mels; ++m) {
                const float reference = expected[static_cast<size_t>(m) * expected_frames + t];
                if (std::memcmp(&reference, &aliased[static_cast<size_t>(t) * actual_mels + m], sizeof(float))) {
                    std::fprintf(stderr, "FAIL frame_major_alias offset=%zu\n", offset);
                    return false;
                }
            }
        }
        std::printf("PASS frame_major_alias offset=%zu\n", offset);
    }

    auto no_pad = cfg;
    no_pad.pad_mode = "none";
    const auto one_frame = waveform(400, 400);
    const auto saved_output = actual;
    const int saved_mels = actual_mels, saved_frames = actual_frames;
    if (transcribe::MelFrontend(no_pad).compute(one_frame.data(), one_frame.size(), actual, actual_mels, actual_frames,
                                               8, transcribe::MelLayout::FrameMajor) != TRANSCRIBE_ERR_INVALID_ARG ||
        actual.size() != saved_output.size() || actual_mels != saved_mels || actual_frames != saved_frames ||
        std::memcmp(actual.data(), saved_output.data(), actual.size() * sizeof(float))) {
        std::fprintf(stderr, "FAIL frame_major_no_output_frames: error changed existing output\n");
        return false;
    }
    std::puts("PASS frame_major_no_output_frames preserves existing output");
    return true;
}

bool compare_pow2_frontend() {
    struct Case { std::vector<float> pcm; size_t min_samples = 0; };
    std::vector<Case> cases;
    for (size_t length : {size_t(160), size_t(256), size_t(257), size_t(399), size_t(400), size_t(479),
                          size_t(480), size_t(511), size_t(512), size_t(559), size_t(560), size_t(561),
                          size_t(16001), size_t(257)}) {
        cases.push_back({waveform(length, length)});
    }
    cases.push_back({{0.625f}, 640});
    cases.push_back({waveform(257, 257), 800});
    cases.push_back({std::vector<float>(1024, -0.0f)});
    cases.push_back({waveform(1024, 641, std::numeric_limits<float>::denorm_min())});
    auto nonfinite = waveform(1024, 1024);
    nonfinite[0] = std::numeric_limits<float>::infinity();
    nonfinite[512] = std::numeric_limits<float>::quiet_NaN();
    cases.push_back({nonfinite});
    cases.push_back({waveform(640, 320)});

    for (const char * padding : {"reflect", "constant", "none"}) {
        for (float emphasis : {0.97f, 0.0f}) {
            transcribe::MelConfig cfg;
            cfg.pad_mode = padding;
            cfg.pre_emphasis = emphasis;
            auto baseline = std::unique_ptr<void, decltype(&mel_baseline_destroy)>(
                mel_baseline_create(&cfg), &mel_baseline_destroy);
            transcribe::MelFrontend candidate(cfg);
            for (auto layout : {transcribe::MelLayout::ChannelMajor, transcribe::MelLayout::FrameMajor}) {
                std::vector<float> expected, actual;
                int expected_mels = 0, expected_frames = 0, actual_mels = 0, actual_frames = 0;
                auto matches = [&] {
                    if (expected_mels != actual_mels || expected_frames != actual_frames ||
                        expected.size() != actual.size()) return false;
                    for (int m = 0; m < actual_mels; ++m) {
                        for (int t = 0; t < actual_frames; ++t) {
                            const size_t reference = static_cast<size_t>(m) * actual_frames + t;
                            const size_t output = layout == transcribe::MelLayout::ChannelMajor ? reference :
                                static_cast<size_t>(t) * actual_mels + m;
                            if (std::memcmp(&expected[reference], &actual[output], sizeof(float))) return false;
                        }
                    }
                    return true;
                };
                for (size_t index = 0; index < cases.size(); ++index) {
                    const auto & item = cases[index];
                    auto padded = item.pcm;
                    padded.resize(std::max(padded.size(), item.min_samples), 0.0f);
                    const int threads = index % 2 ? 1 : 4;
                    const auto before = mel_baseline_compute(baseline.get(), padded.data(), padded.size(), &expected,
                                                             &expected_mels, &expected_frames, threads);
                    const auto after = candidate.compute(item.pcm.data(), item.pcm.size(), actual, actual_mels,
                                                          actual_frames, threads, layout, item.min_samples);
                    if (before != after || !matches()) {
                        std::fprintf(stderr, "FAIL pow2_sequence pad=%s emphasis=%g layout=%d case=%zu\n",
                                     padding, emphasis, static_cast<int>(layout), index);
                        return false;
                    }
                }
                // Shrinking and growing outputs may alias PCM, including an
                // interior pointer. All input reads must precede output resize.
                for (size_t length : {size_t(480), size_t(16001)}) {
                    for (size_t offset : {size_t(0), size_t(17)}) {
                        actual = waveform(length + offset, length + offset);
                        const auto before = mel_baseline_compute(baseline.get(), actual.data() + offset, length,
                                                                 &expected, &expected_mels, &expected_frames, 4);
                        const auto after = candidate.compute(actual.data() + offset, length, actual, actual_mels,
                                                              actual_frames, 4, layout);
                        if (before != TRANSCRIBE_OK || before != after || !matches()) {
                            std::fprintf(stderr, "FAIL pow2_alias pad=%s emphasis=%g layout=%d length=%zu offset=%zu\n",
                                         padding, emphasis, static_cast<int>(layout), length, offset);
                            return false;
                        }
                    }
                }
                std::printf("PASS pow2_sequence pad=%s emphasis=%g layout=%d cases=%zu aliases=4\n",
                            padding, emphasis, static_cast<int>(layout), cases.size());
            }
        }
    }
    // The FP64 padding change also serves frontends without per-feature
    // normalization; keep their existing output and frame-mask behavior.
    for (const char * normalization : {"none", "global", "per_utterance"}) {
        transcribe::MelConfig cfg;
        cfg.normalize = normalization;
        if (!compare((std::string("pow2_") + normalization).c_str(), cfg, waveform(4096, 2049), 4)) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool full = argc > 1 && std::string(argv[1]) == "--full";
    const auto cfg = whisper_config();
    bool okay = true;
    if (bench) {
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 21;
        if (repeats <= 0) {
            std::fprintf(stderr, "repeats must be positive\n");
            return 2;
        }
        for (int threads : {1, 8}) {
            for (size_t voiced : {size_t(0), size_t(16000), size_t(80000), size_t(240000), size_t(480000)}) {
                const std::string name = "30s_" + std::to_string(voiced / 16000) + "s_voiced";
                okay &= compare(name.c_str(), cfg, waveform(480000, voiced), threads, repeats);
            }
        }
        return okay ? 0 : 1;
    }
    for (size_t length : {size_t(201), size_t(399), size_t(400), size_t(479), size_t(480), size_t(16001)}) {
        const std::string name = "silence_length_" + std::to_string(length);
        okay &= compare(name.c_str(), cfg, std::vector<float>(length), 1);
        okay &= compare(("signal_" + name).c_str(), cfg, waveform(length, length), 1);
    }
    for (float zero : {0.0f, -0.0f}) {
        std::vector<float> pcm(1024, zero);
        okay &= compare(std::signbit(zero) ? "negative_zero" : "positive_zero", cfg, pcm, 1);
    }
    std::vector<float> alternating_zeros(2048);
    for (size_t i = 0; i < alternating_zeros.size(); ++i) alternating_zeros[i] = (i & 1) ? -0.0f : 0.0f;
    okay &= compare("alternating_signed_zeros", cfg, alternating_zeros, 8);
    for (float scale : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(), 1e-18f, 1e-8f}) {
        okay &= compare(("tiny_" + std::to_string(scale)).c_str(), cfg, waveform(2048, 1537, scale), 1);
    }
    for (size_t sample : {size_t(0), size_t(159), size_t(160), size_t(199), size_t(200), size_t(201),
                          size_t(319), size_t(320), size_t(321), size_t(2046), size_t(2047)}) {
        std::vector<float> pcm(2048);
        pcm[sample] = 0.625f;
        okay &= compare(("impulse_" + std::to_string(sample)).c_str(), cfg, pcm, 1);
    }
    for (const char * normalization : {"none", "per_feature", "global"}) {
        auto variation = cfg;
        variation.normalize = normalization;
        okay &= compare((std::string(normalization) + "_silence").c_str(), variation, std::vector<float>(2048), 1);
        okay &= compare((std::string(normalization) + "_signal").c_str(), variation, waveform(2048, 641), 1);
    }
    auto emphasis = cfg;
    emphasis.pre_emphasis = 0.97f;
    okay &= compare("pre_emphasis_trailing_zeros", emphasis, waveform(2048, 1000), 1);
    auto constant = cfg;
    constant.pad_mode = "constant";
    okay &= compare("constant_padding", constant, waveform(2048, 1344), 1);
    auto nonfinite = cfg;
    nonfinite.filterbank = transcribe::MelFrontend(cfg).filterbank();
    nonfinite.filterbank[10] = std::numeric_limits<float>::infinity();
    okay &= compare("nonfinite_filterbank", nonfinite, std::vector<float>(2048), 1);
    nonfinite = cfg;
    nonfinite.window.assign(400, 1.0f);
    nonfinite.window[100] = std::numeric_limits<float>::infinity();
    okay &= compare("nonfinite_window", nonfinite, std::vector<float>(2048), 1);
    okay &= compare_frame_major_padding();
    okay &= compare_pow2_frontend();
    if (full) {
        for (int threads : {1, 8}) {
            okay &= compare("30sec_short_signal", cfg, waveform(480000, 25341), threads);
            okay &= compare("30sec_continuous_signal", cfg, waveform(480000, 480000), threads);
            okay &= compare("65sec_trailing_zeros", cfg, waveform(1040000, 735111), threads);
            okay &= compare("65sec_continuous_signal", cfg, waveform(1040000, 1040000), threads);
        }
        transcribe::MelConfig pow2;
        okay &= compare("pow2_long_signal", pow2, waveform(1040000, 735111), 8);
    }
    return okay ? 0 : 1;
}
