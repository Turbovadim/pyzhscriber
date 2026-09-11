// Small macOS checks for immutable weight mappings and copied fallback.
// Run separately with: cpu | metal-shared | metal-private
#include "transcribe-weights-mmap.h"

#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"

#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using transcribe::load_common::WeightMapping;
using transcribe::load_common::try_map_weights;

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Catalog = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&transcribe::safe_buffer_free)>;

void * refused_mapping = nullptr;
ggml_backend_buffer_t refuse_mapping(ggml_backend_dev_t, void * address, size_t, size_t) {
    refused_mapping = address;
    return nullptr;
}

int protection_at(void * pointer) {
    mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(pointer);
    const auto requested = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto status = mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
                                        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    return status == KERN_SUCCESS && address <= requested && requested - address < size ? info.protection : -1;
}

struct Fixture {
    std::string path = "/tmp/whisper-mmap-test-XXXXXX";
    Context source{ggml_init({1024 * 1024, nullptr, false}), ggml_free};
    Catalog catalog{gguf_init_empty(), gguf_free};

    explicit Fixture(bool interleave = false) {
        int fd = mkstemp(path.data());
        require(fd >= 0, "mkstemp failed");
        close(fd);
        require(source && catalog, "fixture allocation failed");
        auto * encoder = ggml_new_tensor_1d(source.get(), GGML_TYPE_F32, 32768);
        auto * f32 = ggml_new_tensor_2d(source.get(), GGML_TYPE_F32, 257, 3);
        auto * q8 = ggml_new_tensor_2d(source.get(), GGML_TYPE_Q8_0, 256, 17);
        auto * f16 = ggml_new_tensor_3d(source.get(), GGML_TYPE_F16, 1, 32, 16);
        auto * q5 = ggml_new_tensor_2d(source.get(), GGML_TYPE_Q5_K, 256, 17);
        auto * q6 = ggml_new_tensor_2d(source.get(), GGML_TYPE_Q6_K, 256, 17);
        auto * bias = ggml_new_tensor_1d(source.get(), GGML_TYPE_F32, 129);
        ggml_set_name(encoder, "enc.unused");
        ggml_set_name(f32, "dec.f32");
        ggml_set_name(q8, "dec.q8");
        ggml_set_name(f16, "dec.f16");
        ggml_set_name(q5, "dec.q5");
        ggml_set_name(q6, "dec.q6");
        ggml_set_name(bias, "dec.bias");
        for (auto * tensor : {encoder, f32, q8, f16, q5, q6, bias}) {
            auto * bytes = static_cast<unsigned char *>(tensor->data);
            for (size_t i = 0; i < ggml_nbytes(tensor); ++i) bytes[i] = (i * 37 + i / 251 + 19) & 255;
        }
        auto * half = static_cast<ggml_fp16_t *>(f16->data);
        for (int64_t i = 0; i < ggml_nelements(f16); ++i) {
            half[i] = ggml_fp32_to_fp16(static_cast<float>(i % 257 - 128) * 0.125f);
        }
        if (interleave) {
            for (auto * tensor : {f32, encoder, q8, f16, q5, q6, bias}) gguf_add_tensor(catalog.get(), tensor);
        } else {
            for (auto * tensor : {encoder, f32, q8, f16, q5, q6, bias}) gguf_add_tensor(catalog.get(), tensor);
        }
        require(gguf_write_to_file(catalog.get(), path.c_str(), false), "fixture write failed");
        ggml_context * raw = nullptr;
        catalog.reset(gguf_init_from_file(path.c_str(), {true, &raw}));
        Context metadata(raw, ggml_free);
        require(catalog && metadata, "fixture metadata parse failed");
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    Context tensors(bool include_encoder = false) const {
        Context ctx(ggml_init({7 * ggml_tensor_overhead() + 256, nullptr, true}), ggml_free);
        require(ctx != nullptr, "decoder context allocation failed");
        for (const char * name : {"enc.unused", "dec.f32", "dec.q8", "dec.f16", "dec.q5", "dec.q6", "dec.bias"}) {
            if (!include_encoder && std::strcmp(name, "enc.unused") == 0) continue;
            auto * original = ggml_get_tensor(source.get(), name);
            auto * copy = ggml_dup_tensor(ctx.get(), original);
            ggml_set_name(copy, name);
        }
        return ctx;
    }
};

void check_unbound(ggml_context * context) {
    for (auto * tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
        require(tensor->buffer == nullptr && tensor->data == nullptr, "failure left a tensor bound");
    }
}

void check_bytes(ggml_context * context, const Fixture & fixture) {
    for (auto * tensor = ggml_get_first_tensor(context); tensor; tensor = ggml_get_next_tensor(context, tensor)) {
        auto * original = ggml_get_tensor(fixture.source.get(), tensor->name);
        std::vector<unsigned char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        require(std::memcmp(bytes.data(), original->data, bytes.size()) == 0, "mapped bytes differ");
    }
}

void check_cpu_promotion(ggml_context * context, ggml_backend_t backend) {
    auto * source = ggml_get_tensor(context, "dec.f16");
    auto * promoted = source;
    transcribe::BackendPlan plan;
    plan.primary = backend;
    plan.primary_kind = transcribe::BackendKind::Cpu;
    ggml_context * raw_context = nullptr;
    ggml_backend_buffer_t raw_buffer = nullptr;
    const auto status = transcribe::load_common::promote_conv_pw_f16_to_f32_on_cpu(
        plan, {{&promoted, source}}, "mapped F16 test", &raw_context, &raw_buffer);
    Context replacement_context(raw_context, ggml_free);
    Buffer replacement_buffer(raw_buffer, transcribe::safe_buffer_free);
    require(status == TRANSCRIBE_OK && promoted != source && replacement_context && replacement_buffer,
            "mapped F16 promotion failed");
    require(promoted->type == GGML_TYPE_F32 && promoted->buffer != source->buffer,
            "promotion did not allocate separate F32 storage");
    std::vector<float> actual(ggml_nelements(source));
    ggml_backend_tensor_get(promoted, actual.data(), 0, actual.size() * sizeof(float));
    const auto * half = static_cast<const ggml_fp16_t *>(source->data);
    for (size_t i = 0; i < actual.size(); ++i) {
        require(actual[i] == ggml_fp16_to_fp32(half[i]), "mapped F16 promotion changed values");
    }
}

void rejected_then_copy(Fixture & fixture, ggml_backend_t backend, bool stream_ok, bool include_encoder = false) {
    auto context = fixture.tensors(include_encoder);
    ggml_backend_buffer_t raw = nullptr;
    std::unique_ptr<WeightMapping> mapping;
    require(!try_map_weights(fixture.path, fixture.catalog.get(), context.get(), backend, raw, mapping),
            "unsafe fixture accepted");
    require(!raw && !mapping, "failure published outputs");
    check_unbound(context.get());
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), transcribe::safe_buffer_free);
    require(buffer != nullptr, "copied fallback allocation failed");
    auto status = transcribe::load_common::stream_tensor_data(fixture.path, fixture.catalog.get(), context.get(),
                                                             "mmap-test fallback");
    require(status == (stream_ok ? TRANSCRIBE_OK : TRANSCRIBE_ERR_GGUF), "copied fallback status changed");
    if (stream_ok) check_bytes(context.get(), fixture);
}

void run(const std::string & mode) {
    const bool metal = mode != "cpu";
    const bool shared = mode == "metal-shared";
    if (metal) {
        unsetenv(shared ? "GGML_METAL_SHARED_BUFFERS_DISABLE" : "GGML_METAL_SHARED_BUFFERS_ENABLE");
        setenv(shared ? "GGML_METAL_SHARED_BUFFERS_ENABLE" : "GGML_METAL_SHARED_BUFFERS_DISABLE", "1", 1);
    }
    std::unique_ptr<ggml_backend, decltype(&transcribe::safe_backend_free)> backend(
        metal ? ggml_backend_metal_init() : ggml_backend_cpu_init(), transcribe::safe_backend_free);
    require(backend != nullptr, "backend allocation failed");
    require(!ggml_backend_metal_buffer_type_is_shared(nullptr), "null buffer type reports shared");
    require(ggml_backend_metal_buffer_type_is_shared(ggml_backend_get_default_buffer_type(backend.get())) == shared,
            "incorrect shared buffer type predicate");

    if (mode == "metal-private") {
        Fixture fixture;
        rejected_then_copy(fixture, backend.get(), true);
        rejected_then_copy(fixture, backend.get(), true, true);
        printf("PASS metal-private: mapped path rejected, exact copied fallback bytes\n");
        return;
    }
    {
        Fixture fixture;
        auto context = fixture.tensors();
        auto device = ggml_backend_get_device(backend.get());
        auto original = device->iface.buffer_from_host_ptr;
        device->iface.buffer_from_host_ptr = refuse_mapping;
        ggml_backend_buffer_t raw = nullptr;
        std::unique_ptr<WeightMapping> mapping;
        const bool accepted = try_map_weights(fixture.path, fixture.catalog.get(), context.get(),
                                                       backend.get(), raw, mapping);
        device->iface.buffer_from_host_ptr = original;
        require(!accepted && !raw && !mapping, "backend refusal published outputs");
        require(refused_mapping && protection_at(refused_mapping) == -1, "backend refusal leaked mapping");
        check_unbound(context.get());
    }
    for (const bool include_encoder : {false, true}) {
        Fixture fixture;
        auto context = fixture.tensors(include_encoder);
        ggml_backend_buffer_t raw = nullptr;
        std::unique_ptr<WeightMapping> mapping;
        require(try_map_weights(fixture.path, fixture.catalog.get(), context.get(), backend.get(), raw,
                                        mapping), "valid decoder mapping failed");
        Buffer buffer(raw, transcribe::safe_buffer_free);
        require(mapping && buffer, "mapping success omitted ownership");
        require(protection_at(mapping->base) == VM_PROT_READ, "mapping is not read-only");
        if (!include_encoder) {
            require(mapping->size < std::filesystem::file_size(fixture.path), "decoder mapping retained encoder pages");
        }
        require(ggml_backend_metal_buffer_is_shared(buffer.get()) == shared, "wrong mapped buffer interface");
        check_bytes(context.get(), fixture);
        if (!metal) {
            check_cpu_promotion(context.get(), backend.get());
            check_bytes(context.get(), fixture);
        }
        require(unlink(fixture.path.c_str()) == 0, "fixture unlink failed");
        check_bytes(context.get(), fixture);
        auto * first = ggml_get_first_tensor(context.get());
        const auto * original = ggml_get_tensor(fixture.source.get(), first->name);
        buffer.reset();
        require(std::memcmp(first->data, original->data, ggml_nbytes(first)) == 0,
                "buffer release invalidated externally owned mapping");
        void * address = mapping->base;
        mapping.reset();
        require(protection_at(address) == -1, "mapping owner did not unmap");
    }
    {
        Fixture fixture;
        auto context = fixture.tensors();
        ggml_set_name(ggml_get_first_tensor(context.get()), "missing.weight");
        ggml_backend_buffer_t raw = nullptr;
        std::unique_ptr<WeightMapping> mapping;
        require(!try_map_weights(fixture.path, fixture.catalog.get(), context.get(), backend.get(), raw,
                                         mapping), "unknown weight accepted");
        check_unbound(context.get());
    }
    {
        Fixture interleaved(true);
        rejected_then_copy(interleaved, backend.get(), true);
    }
    {
        Fixture truncated;
        const int64_t id = gguf_find_tensor(truncated.catalog.get(), "dec.bias");
        const size_t final_end = gguf_get_data_offset(truncated.catalog.get()) +
            gguf_get_tensor_offset(truncated.catalog.get(), id) + gguf_get_tensor_size(truncated.catalog.get(), id);
        std::filesystem::resize_file(truncated.path, final_end - 1);
        rejected_then_copy(truncated, backend.get(), false);
    }
    {
        Fixture missing_payload;
        std::filesystem::resize_file(missing_payload.path, gguf_get_data_offset(missing_payload.catalog.get()));
        rejected_then_copy(missing_payload, backend.get(), false);
    }
    printf("PASS %s: exact F32/F16/Q8_0/Q5_K/Q6_K bytes, separate CPU F32 promotion, "
           "decoder and full spans, unlink/buffer/unmap lifetimes, "
           "backend-refusal cleanup, unknown-weight/interleaved rejection, "
           "truncated/out-of-file rejection and copied fallback\n", mode.c_str());
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "cpu") != 0 && std::strcmp(argv[1], "metal-shared") != 0 &&
                      std::strcmp(argv[1], "metal-private") != 0)) {
        fprintf(stderr, "usage: %s cpu|metal-shared|metal-private\n", argv[0]);
        return 2;
    }
    try {
        run(argv[1]);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL %s: %s\n", argv[1], error.what());
        return 1;
    }
}
