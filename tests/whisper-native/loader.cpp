// Small, model-free check of direct and staged GGUF weight loading.
// Run separately with: cpu | metal-shared | metal-private
#include "transcribe-load-common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct FixtureFile {
    std::string path;

    FixtureFile() {
        path = (std::filesystem::temp_directory_path() / "pyzhscriber-loader-XXXXXX").string();
        const int fd = mkstemp(path.data());
        require(fd >= 0, "creating temporary GGUF failed");
        close(fd);
    }

    ~FixtureFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

void check_bytes(const ggml_tensor * actual, const ggml_tensor * expected) {
    require(actual != nullptr, "missing target tensor");
    require(ggml_nbytes(actual) == ggml_nbytes(expected), "tensor size changed");
    std::vector<uint8_t> bytes(ggml_nbytes(actual));
    ggml_backend_tensor_get(actual, bytes.data(), 0, bytes.size());
    require(std::memcmp(bytes.data(), expected->data, bytes.size()) == 0, "tensor payload changed");
}

void run(const std::string & mode) {
    const bool metal = mode != "cpu";
    const bool shared = mode == "metal-shared";
    if (metal) {
        // Metal captures these flags at device initialization.
        unsetenv(shared ? "GGML_METAL_SHARED_BUFFERS_DISABLE" : "GGML_METAL_SHARED_BUFFERS_ENABLE");
        setenv(shared ? "GGML_METAL_SHARED_BUFFERS_ENABLE" : "GGML_METAL_SHARED_BUFFERS_DISABLE", "1", 1);
    }

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        metal ? ggml_backend_metal_init() : ggml_backend_cpu_init(), ggml_backend_free);
    require(backend != nullptr, "backend initialization failed");

    std::unique_ptr<ggml_context, decltype(&ggml_free)> source(
        ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    require(source != nullptr, "source context allocation failed");
    auto * f32 = ggml_new_tensor_2d(source.get(), GGML_TYPE_F32, 257, 131);
    auto * q8 = ggml_new_tensor_2d(source.get(), GGML_TYPE_Q8_0, 256, 513);
    require(f32 != nullptr && q8 != nullptr, "source tensor allocation failed");
    ggml_set_name(f32, "test.f32");
    ggml_set_name(q8, "test.q8");
    for (ggml_tensor * tensor : {f32, q8}) {
        auto * bytes = static_cast<uint8_t *>(tensor->data);
        for (size_t i = 0; i < ggml_nbytes(tensor); ++i) {
            bytes[i] = static_cast<uint8_t>((i * 37 + i / 251 + 19) & 0xff);
        }
    }

    FixtureFile file;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> written(gguf_init_empty(), gguf_free);
    require(written != nullptr, "GGUF allocation failed");
    gguf_add_tensor(written.get(), f32);
    gguf_add_tensor(written.get(), q8);
    require(gguf_get_tensor_offset(written.get(), 1) > ggml_nbytes(f32), "fixture needs alignment padding");
    require(gguf_write_to_file(written.get(), file.path.c_str(), false), "writing fixture failed");

    ggml_context * catalog_raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> catalog(
        gguf_init_from_file(file.path.c_str(), {true, &catalog_raw}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> target(catalog_raw, ggml_free);
    require(catalog != nullptr && target != nullptr, "reading GGUF metadata failed");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(target.get(), backend.get()), ggml_backend_buffer_free);
    require(buffer != nullptr, "target buffer allocation failed");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    require(!ggml_backend_metal_buffer_is_shared(nullptr), "null buffer reported shared");
    require(ggml_backend_metal_buffer_is_shared(buffer.get()) == shared, "incorrect Metal shared predicate");
    require(ggml_backend_buffer_is_host(buffer.get()) == !metal, "unexpected host buffer classification");
    require(transcribe::load_common::buffer_allows_direct_io(buffer.get()) == (!metal || shared),
            "incorrect direct-I/O policy");

    require(transcribe::load_common::stream_tensor_data(file.path, catalog.get(), target.get(), "loader-test") ==
                TRANSCRIBE_OK,
            "complete fixture failed to load");
    check_bytes(ggml_get_tensor(target.get(), "test.f32"), f32);
    check_bytes(ggml_get_tensor(target.get(), "test.q8"), q8);

    // Keep the valid catalog, then interrupt the second payload. The loader
    // must report an I/O error; the buffer can be partly filled and is freed
    // by its owner when this function returns.
    ggml_backend_buffer_clear(buffer.get(), 0xcc);
    const size_t partial_size = gguf_get_data_offset(catalog.get()) +
        gguf_get_tensor_offset(catalog.get(), 1) + ggml_nbytes(q8) / 2;
    std::filesystem::resize_file(file.path, partial_size);
    require(transcribe::load_common::stream_tensor_data(file.path, catalog.get(), target.get(), "loader-test") ==
                TRANSCRIBE_ERR_GGUF,
            "truncated payload was accepted");
    check_bytes(ggml_get_tensor(target.get(), "test.f32"), f32);

    std::printf("PASS %s: exact F32/Q8_0 bytes, shared-buffer predicate, truncated payload\n", mode.c_str());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "cpu") != 0 && std::strcmp(argv[1], "metal-shared") != 0 &&
                      std::strcmp(argv[1], "metal-private") != 0)) {
        std::fprintf(stderr, "usage: %s cpu|metal-shared|metal-private\n", argv[0]);
        return 2;
    }
    try {
        run(argv[1]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s: %s\n", argv[1], error.what());
        return 1;
    }
}
