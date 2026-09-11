// Compare native Canary builds using the same 16 kHz mono PCM16 WAV manifest.
// Each line is a WAV path, optionally followed by a tab and language code.
#include "transcribe.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

using Clock = std::chrono::steady_clock;
using Model = std::unique_ptr<transcribe_model, decltype(&transcribe_model_free)>;
using Session = std::unique_ptr<transcribe_session, decltype(&transcribe_session_free)>;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string quoted(const char * text) {
    if (!text) return "null";
    std::ostringstream out;
    out << '"';
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        if (*p == '"' || *p == '\\') out << '\\' << *p;
        else if (*p < 32) {
            constexpr char hex[] = "0123456789abcdef";
            out << "\\u00" << hex[*p >> 4] << hex[*p & 15];
        } else out << *p;
    }
    out << '"';
    return out.str();
}

std::string memory() {
#ifdef __APPLE__
    rusage_info_v0 usage{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V0, reinterpret_cast<rusage_info_t *>(&usage)) == 0) {
        return "{\"rss_bytes\":" + std::to_string(usage.ri_resident_size) +
               ",\"physical_footprint_bytes\":" + std::to_string(usage.ri_phys_footprint) + "}";
    }
#endif
    return "null";
}

void check(transcribe_status status, const char * operation) {
    if (status != TRANSCRIBE_OK) {
        throw std::runtime_error(std::string(operation) + ": status " + std::to_string(status));
    }
}

uint32_t read_le(std::istream & stream, int bytes) {
    uint32_t result = 0;
    for (int i = 0; i < bytes; ++i) {
        const int byte = stream.get();
        if (byte < 0) throw std::runtime_error("truncated WAV");
        result |= static_cast<uint32_t>(byte) << (8 * i);
    }
    return result;
}

std::string read_tag(std::istream & stream) {
    char tag[4];
    if (!stream.read(tag, sizeof(tag))) throw std::runtime_error("missing WAV chunk");
    return std::string(tag, sizeof(tag));
}

std::vector<float> read_wav(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input || read_tag(input) != "RIFF") throw std::runtime_error("invalid WAV: " + path);
    read_le(input, 4);
    if (read_tag(input) != "WAVE") throw std::runtime_error("invalid WAVE header");
    bool valid_format = false;
    while (input.peek() != EOF) {
        const auto tag = read_tag(input);
        const auto size = read_le(input, 4);
        const auto next = input.tellg() + static_cast<std::streamoff>(size) + static_cast<std::streamoff>(size & 1);
        if (tag == "fmt ") {
            if (size < 16) throw std::runtime_error("short WAV format");
            const auto format = read_le(input, 2);
            const auto channels = read_le(input, 2);
            const auto rate = read_le(input, 4);
            read_le(input, 4);
            const auto alignment = read_le(input, 2);
            const auto bits = read_le(input, 2);
            valid_format = format == 1 && channels == 1 && rate == 16000 && alignment == 2 && bits == 16;
        } else if (tag == "data") {
            if (!valid_format || size == 0 || size % 2 || size / 2 > std::numeric_limits<int>::max()) {
                throw std::runtime_error("expected nonempty mono PCM16 WAV at 16 kHz: " + path);
            }
            std::vector<float> samples(size / 2);
            for (auto & sample : samples) {
                const auto value = read_le(input, 2);
                const int signed_value = value >= 32768 ? static_cast<int>(value) - 65536 : value;
                sample = static_cast<float>(signed_value) / 32767.0f;
            }
            return samples;
        }
        input.seekg(next);
    }
    throw std::runtime_error("missing WAV audio: " + path);
}

int main(int argc, char ** argv) {
    try {
        if (argc < 4) throw std::runtime_error(
            "usage: canary_bench MODEL.gguf MANIFEST.tsv REPEATS [--cpu] [--threads N] "
            "[--kv-f32] [--n-ctx N] [--reuse-session] [--language CODE] [--translate] [--pnc-off] [--reload-model]");
        const std::string model_path = argv[1];
        const int repeats = std::stoi(argv[3]);
        if (repeats < 1) throw std::runtime_error("repeats must be positive");
        transcribe_model_load_params load_params;
        transcribe_model_load_params_init(&load_params);
        load_params.backend = TRANSCRIBE_BACKEND_METAL;
        transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);
        transcribe_run_params run_params;
        transcribe_run_params_init(&run_params);
        run_params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
        bool reuse = false, reload = false;
        std::string language = "en";
        for (int i = 4; i < argc; ++i) {
            const std::string option = argv[i];
            auto value = [&]() -> const char * {
                if (++i >= argc) throw std::runtime_error("missing value for " + option);
                return argv[i];
            };
            if (option == "--cpu") load_params.backend = TRANSCRIBE_BACKEND_CPU;
            else if (option == "--threads") session_params.n_threads = std::stoi(value());
            else if (option == "--n-ctx") session_params.n_ctx = std::stoi(value());
            else if (option == "--language") language = value();
            else if (option == "--kv-f32") session_params.kv_type = TRANSCRIBE_KV_TYPE_F32;
            else if (option == "--reuse-session") reuse = true;
            else if (option == "--reload-model") reload = true;
            else if (option == "--pnc-off") run_params.pnc = TRANSCRIBE_PNC_MODE_OFF;
            else if (option == "--translate") {
                run_params.task = TRANSCRIBE_TASK_TRANSLATE;
                run_params.target_language = "en";
            } else throw std::runtime_error("unknown option: " + option);
        }
        struct Input { std::string path, language; };
        std::vector<Input> inputs;
        std::ifstream manifest(argv[2]);
        if (!manifest) throw std::runtime_error("cannot open manifest");
        for (std::string line; std::getline(manifest, line);) {
            if (line.empty()) continue;
            const auto tab = line.find('\t');
            inputs.push_back({line.substr(0, tab), tab == std::string::npos ? language : line.substr(tab + 1)});
        }
        if (inputs.empty()) throw std::runtime_error("empty manifest");
        std::cout << std::unitbuf;
        check(transcribe_init_backends_default(), "init backends");
        Model model(nullptr, transcribe_model_free);
        Session session(nullptr, transcribe_session_free);
        for (int repetition = 0; repetition < repeats; ++repetition) {
            for (const auto & input : inputs) {
                if (!model) {
                    const auto start = Clock::now();
                    transcribe_model * raw = nullptr;
                    check(transcribe_model_load_file(model_path.c_str(), &load_params, &raw), "load model");
                    model.reset(raw);
                    const std::string arch = transcribe_model_arch_string(model.get());
                    if (arch != "canary" && arch != "canary_qwen") throw std::runtime_error("expected Canary architecture");
                    std::cout << "{\"event\":\"load\",\"model\":" << quoted(model_path.c_str())
                              << ",\"backend\":" << quoted(transcribe_model_backend(model.get()))
                              << ",\"load_ms\":" << elapsed(start) << ",\"memory\":" << memory() << "}\n";
                }
                const auto pcm = read_wav(input.path);
                const auto start = Clock::now();
                if (!session) {
                    transcribe_session * raw = nullptr;
                    check(transcribe_session_init(model.get(), &session_params, &raw), "init session");
                    session.reset(raw);
                }
                run_params.language = input.language.c_str();
                const auto status = transcribe_run(session.get(), pcm.data(), static_cast<int>(pcm.size()), &run_params);
                const double inference_ms = elapsed(start);
                transcribe_timings timings;
                transcribe_timings_init(&timings);
                check(transcribe_get_timings(session.get(), &timings), "timings");
                std::cout << "{\"event\":\"run\",\"file\":" << quoted(input.path.c_str())
                          << ",\"repetition\":" << repetition << ",\"status\":" << status
                          << ",\"audio_ms\":" << pcm.size() / 16.0 << ",\"elapsed_ms\":" << inference_ms
                          << ",\"mel_ms\":" << timings.mel_ms << ",\"encode_ms\":" << timings.encode_ms
                          << ",\"decode_ms\":" << timings.decode_ms
                          << ",\"language\":" << quoted(transcribe_detected_language(session.get()))
                          << ",\"text\":" << quoted(transcribe_full_text(session.get()))
                          << ",\"raw_text\":" << quoted(transcribe_raw_text(session.get()))
                          << ",\"truncated\":" << (transcribe_was_truncated(session.get()) ? "true" : "false")
                          << ",\"segments\":[";
                for (int i = 0; i < transcribe_n_segments(session.get()); ++i) {
                    transcribe_segment segment;
                    transcribe_segment_init(&segment);
                    check(transcribe_get_segment(session.get(), i, &segment), "segment");
                    if (i) std::cout << ',';
                    std::cout << "{\"start_ms\":" << segment.t0_ms << ",\"end_ms\":" << segment.t1_ms
                              << ",\"text\":" << quoted(segment.text) << '}';
                }
                std::cout << "],\"memory_during_session\":" << memory();
                if (!reuse || reload) session.reset();
                std::cout << ",\"memory_after_session\":" << memory() << "}\n";
                check(status, "transcribe");
                if (reload) {
                    model.reset();
                    std::cout << "{\"event\":\"unload\",\"memory\":" << memory() << "}\n";
                }
            }
        }
        session.reset();
        model.reset();
        std::cout << "{\"event\":\"unload\",\"memory\":" << memory() << "}\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
