// Real-model lifecycle checks. Inputs are native float32 mono PCM at 16 kHz.
// Run this same fixture against baseline and candidate libraries and compare
// JSONL results; batch kernels may already differ from single-utterance kernels.
#include "transcribe.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Model = std::unique_ptr<transcribe_model, decltype(&transcribe_model_free)>;
using Session = std::unique_ptr<transcribe_session, decltype(&transcribe_session_free)>;

namespace {
int failures = 0;

void check(bool condition, const std::string & message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL " << message << '\n';
    }
}

void require_ok(transcribe_status status, const char * operation) {
    if (status != TRANSCRIBE_OK) {
        throw std::runtime_error(std::string(operation) + ": status " + std::to_string(status));
    }
}

std::string quoted(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else out << c;
    }
    out << '"';
    return out.str();
}

std::string copied(const char * value) { return value ? value : ""; }

struct Result {
    transcribe_status status;
    std::string text, raw_text, language;
    bool truncated, aborted;

    bool operator==(const Result & other) const {
        return status == other.status && text == other.text && raw_text == other.raw_text &&
               language == other.language && truncated == other.truncated && aborted == other.aborted;
    }
};

void emit(const std::string & name, const Result & result, const Result * reference = nullptr) {
    std::cout << "{\"case\":" << quoted(name) << ",\"status\":" << result.status
              << ",\"text\":" << quoted(result.text) << ",\"raw_text\":" << quoted(result.raw_text)
              << ",\"language\":" << quoted(result.language)
              << ",\"truncated\":" << (result.truncated ? "true" : "false")
              << ",\"aborted\":" << (result.aborted ? "true" : "false");
    if (reference) std::cout << ",\"matches_fresh_single\":" << (result == *reference ? "true" : "false");
    std::cout << "}\n";
}

std::vector<float> read_pcm(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = input.tellg();
    if (!input || bytes <= 0 || bytes % sizeof(float) != 0 ||
        bytes / sizeof(float) > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid float32 PCM file: ") + path);
    }
    std::vector<float> pcm(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(pcm.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error(std::string("short PCM read: ") + path);
    }
    return pcm;
}

Model load_model(const char * path, const transcribe_model_load_params & params) {
    transcribe_model * raw = nullptr;
    require_ok(transcribe_model_load_file(path, &params, &raw), "load model");
    Model model(raw, transcribe_model_free);
    if (!model || copied(transcribe_model_arch_string(model.get())) != "canary") {
        throw std::runtime_error("fixture requires a standard Canary model");
    }
    return model;
}

Session open_session(transcribe_model * model, transcribe_session_params params, int n_ctx = 0) {
    params.n_ctx = n_ctx;
    transcribe_session * raw = nullptr;
    require_ok(transcribe_session_init(model, &params, &raw), "init session");
    if (!raw) throw std::runtime_error("init session returned a null handle");
    return Session(raw, transcribe_session_free);
}

Result run(transcribe_session * session, const std::vector<float> & pcm,
           const transcribe_run_params & params, const std::string & name,
           const Result * reference = nullptr, bool expect_abort = false) {
    const auto status = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &params);
    Result result{status, copied(transcribe_full_text(session)), copied(transcribe_raw_text(session)),
                  copied(transcribe_detected_language(session)), transcribe_was_truncated(session),
                  transcribe_was_aborted(session)};
    emit(name, result, reference);
    if (expect_abort) {
        check(status == TRANSCRIBE_ERR_ABORTED && result.aborted, name + ": expected cancellation");
    } else {
        check(status == TRANSCRIBE_OK || status == TRANSCRIBE_ERR_OUTPUT_TRUNCATED,
              name + ": unexpected run status");
        check(!result.aborted, name + ": stale aborted flag");
        check(result.truncated == (status == TRANSCRIBE_ERR_OUTPUT_TRUNCATED), name + ": truncation flag mismatch");
    }
    if (reference) check(result == *reference, name + ": stale or changed session result");
    return result;
}

void batch(transcribe_session * session, const std::vector<float> & first, const std::vector<float> & second,
           const Result & first_reference, const Result & second_reference,
           const transcribe_run_params & params, const std::string & name) {
    const float * pcm[] = {first.data(), second.data()};
    const int sizes[] = {static_cast<int>(first.size()), static_cast<int>(second.size())};
    const auto status = transcribe_run_batch(session, pcm, sizes, 2, &params);
    std::cout << "{\"case\":" << quoted(name + "/dispatch") << ",\"status\":" << status
              << ",\"count\":" << transcribe_batch_n_results(session) << "}\n";
    check(status == TRANSCRIBE_OK, name + ": whole-batch failure");
    check(transcribe_batch_n_results(session) == 2, name + ": wrong result count");
    const Result * references[] = {&first_reference, &second_reference};
    for (int i = 0; i < 2; ++i) {
        const auto row_status = transcribe_batch_status(session, i);
        Result result{row_status, copied(transcribe_batch_full_text(session, i)),
                      copied(transcribe_batch_raw_text(session, i)),
                      copied(transcribe_batch_detected_language(session, i)),
                      row_status == TRANSCRIBE_ERR_OUTPUT_TRUNCATED, row_status == TRANSCRIBE_ERR_ABORTED};
        // Compare these rows to the same named baseline rows. Do not fail just
        // because batching selected different kernels from single inference.
        emit(name + "/" + std::to_string(i), result, references[i]);
        check(row_status == TRANSCRIBE_OK || row_status == TRANSCRIBE_ERR_OUTPUT_TRUNCATED,
              name + ": unexpected row status");
    }
}

bool abort_after_five_polls(void * opaque) {
    return ++*static_cast<int *>(opaque) >= 5;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 4) throw std::runtime_error(
            "usage: lifecycle MODEL.gguf SHORT.f32 LONG.f32 [--language CODE] [--threads N] [--cpu] [--kv-f32]");
        transcribe_model_load_params load_params;
        transcribe_model_load_params_init(&load_params);
        load_params.backend = TRANSCRIBE_BACKEND_METAL;
        transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);
        transcribe_run_params run_params;
        transcribe_run_params_init(&run_params);
        run_params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
        std::string language = "en";
        for (int i = 4; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--cpu") load_params.backend = TRANSCRIBE_BACKEND_CPU;
            else if (option == "--kv-f32") session_params.kv_type = TRANSCRIBE_KV_TYPE_F32;
            else if (option == "--language" && i + 1 < argc) language = argv[++i];
            else if (option == "--threads" && i + 1 < argc) session_params.n_threads = std::stoi(argv[++i]);
            else throw std::runtime_error("invalid option: " + option);
        }
        run_params.language = language.c_str();
        const auto short_pcm = read_pcm(argv[2]);
        const auto long_pcm = read_pcm(argv[3]);
        if (long_pcm.size() <= short_pcm.size()) throw std::runtime_error("LONG must contain more samples than SHORT");
        std::cout << std::unitbuf;
        require_ok(transcribe_init_backends_default(), "init backends");
        Result fresh_short{};
        {
            auto model = load_model(argv[1], load_params);
            {
                auto session = open_session(model.get(), session_params);
                fresh_short = run(session.get(), short_pcm, run_params, "fresh/short");
            }
            Result fresh_long{};
            {
                auto session = open_session(model.get(), session_params);
                fresh_long = run(session.get(), long_pcm, run_params, "fresh/long");
            }
            {
                auto session = open_session(model.get(), session_params);
                run(session.get(), short_pcm, run_params, "reuse/short-before", &fresh_short);
                run(session.get(), long_pcm, run_params, "reuse/long", &fresh_long);
                run(session.get(), short_pcm, run_params, "reuse/short-after", &fresh_short);
                batch(session.get(), short_pcm, long_pcm, fresh_short, fresh_long, run_params, "batch/short-long");
                batch(session.get(), long_pcm, short_pcm, fresh_long, fresh_short, run_params, "batch/long-short");
                run(session.get(), short_pcm, run_params, "after-batch/short", &fresh_short);
            }
            {
                auto session = open_session(model.get(), session_params);
                int polls = 0;
                transcribe_set_abort_callback(session.get(), abort_after_five_polls, &polls);
                run(session.get(), long_pcm, run_params, "cancel/long", nullptr, true);
                transcribe_set_abort_callback(session.get(), nullptr, nullptr);
                run(session.get(), short_pcm, run_params, "after-cancel/short", &fresh_short);
            }
            for (int limit : {63, 64, 65, 127, 128, 129}) {
                auto session = open_session(model.get(), session_params, limit);
                run(session.get(), long_pcm, run_params, "context/" + std::to_string(limit));
            }
        }
        // The first model and all its sessions are gone before each reload.
        // Kept Result objects own their strings and borrow no library storage.
        for (int cycle = 0; cycle < 5; ++cycle) {
            auto model = load_model(argv[1], load_params);
            auto session = open_session(model.get(), session_params);
            run(session.get(), short_pcm, run_params, "reload/" + std::to_string(cycle), &fresh_short);
        }
        std::cout << "{\"case\":\"summary\",\"failures\":" << failures << "}\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
