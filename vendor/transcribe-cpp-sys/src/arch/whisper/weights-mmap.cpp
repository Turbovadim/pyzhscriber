#include "weights-mmap.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-backend.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(__APPLE__) && TARGET_OS_OSX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(TRANSCRIBE_HAS_METAL) && !defined(TRANSCRIBE_GGML_BACKEND_DL)
#include "ggml-metal.h"
#endif
#endif

namespace transcribe::whisper {

WeightMapping::~WeightMapping() {
#if defined(__APPLE__) && TARGET_OS_OSX
    if (base != nullptr) {
        munmap(base, size);
    }
#endif
}

bool try_map_weights(const std::string & path, const gguf_context * gguf,
                     ggml_context * tensors, ggml_backend_t backend,
                     ggml_backend_buffer_t & buffer,
                     std::unique_ptr<WeightMapping> & mapping) noexcept {
#if defined(__APPLE__) && TARGET_OS_OSX
    try {
        if (!gguf || !tensors || !backend || buffer || mapping) {
            return false;
        }
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        bool compatible = buft == ggml_backend_cpu_buffer_type();
#if defined(TRANSCRIBE_HAS_METAL) && !defined(TRANSCRIBE_GGML_BACKEND_DL)
        compatible |= ggml_backend_is_metal(backend) && ggml_backend_metal_buffer_type_is_shared(buft);
#endif
        if (!compatible) {
            return false;
        }
        const auto device = ggml_backend_get_device(backend);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        if (!props.caps.mmap_support || !props.caps.buffer_from_host_ptr) {
            return false;
        }

        struct TensorRange {
            ggml_tensor * tensor;
            size_t offset;
            size_t size;
        };
        std::vector<TensorRange> ranges;
        const size_t alignment = gguf_get_alignment(gguf);
        // GGUF's default 32-byte tensor alignment is supported by both mapped
        // backends. CPU's stricter buffer-base alignment is provided by mmap.
        if (alignment < 32 || alignment % 32 != 0) {
            return false;
        }
        size_t max_tensor_size = 0;
        for (auto * tensor = ggml_get_first_tensor(tensors); tensor;
             tensor = ggml_get_next_tensor(tensors, tensor)) {
            if (tensor->buffer || tensor->data || tensor->view_src || tensor->op != GGML_OP_NONE || !ggml_is_contiguous(tensor)) {
                return false;
            }
            const int64_t id = gguf_find_tensor(gguf, tensor->name);
            if (id < 0 || gguf_get_tensor_type(gguf, id) != tensor->type ||
                std::memcmp(gguf_get_tensor_ne(gguf, id), tensor->ne, sizeof(tensor->ne)) != 0) {
                return false;
            }
            const size_t size = ggml_nbytes(tensor);
            const size_t offset = gguf_get_tensor_offset(gguf, id);
            if (size == 0 || size != gguf_get_tensor_size(gguf, id) || offset % alignment != 0 ||
                size != ggml_backend_buft_get_alloc_size(buft, tensor) ||
                offset > std::numeric_limits<size_t>::max() - size) {
                return false;
            }
            ranges.push_back({tensor, offset, size});
            max_tensor_size = std::max(max_tensor_size, size);
        }
        if (ranges.empty()) {
            return false;
        }
        std::sort(ranges.begin(), ranges.end(), [](const TensorRange & a, const TensorRange & b) {
            return a.offset < b.offset;
        });
        size_t end = ranges.front().offset;
        for (const auto & range : ranges) {
            const size_t padding = (alignment - end % alignment) % alignment;
            if (end > std::numeric_limits<size_t>::max() - padding || range.offset != end + padding) {
                return false;
            }
            end = range.offset + range.size;
        }

        const size_t data_offset = gguf_get_data_offset(gguf);
        if (data_offset > std::numeric_limits<size_t>::max() - end || data_offset % alignment != 0) {
            return false;
        }
        const size_t file_begin = data_offset + ranges.front().offset;
        const size_t file_end = data_offset + end;
        const long page_size_raw = sysconf(_SC_PAGESIZE);
        if (page_size_raw <= 0) {
            return false;
        }
        const size_t page_size = static_cast<size_t>(page_size_raw);
        const size_t map_offset = file_begin - file_begin % page_size;
        const size_t map_padding = (page_size - file_end % page_size) % page_size;
        if (file_end > std::numeric_limits<size_t>::max() - map_padding ||
            map_offset > static_cast<uintmax_t>(std::numeric_limits<off_t>::max())) {
            return false;
        }
        const size_t map_size = file_end + map_padding - map_offset;
        // Keep a single Metal view, avoiding its overlapping-view limits.
        if (map_size > ggml_backend_buft_get_max_size(buft)) {
            return false;
        }
        auto owner = std::make_unique<WeightMapping>();
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) {
            return false;
        }
        struct stat stat_buf{};
        const bool valid_file = fstat(fd, &stat_buf) == 0 && S_ISREG(stat_buf.st_mode) &&
                                stat_buf.st_size >= 0 &&
                                static_cast<uintmax_t>(stat_buf.st_size) >= file_end;
        void * base = valid_file ? mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd,
                                       static_cast<off_t>(map_offset)) : MAP_FAILED;
        close(fd);
        if (base == MAP_FAILED) {
            return false;
        }
        owner->base = base;
        owner->size = map_size;

        // Free backend references before the file mapping on every failure.
        std::unique_ptr<ggml_backend_buffer, decltype(&safe_buffer_free)> mapped_buffer(
            ggml_backend_dev_buffer_from_host_ptr(device, base, map_size, max_tensor_size), safe_buffer_free);
        if (!mapped_buffer || ggml_backend_buffer_get_base(mapped_buffer.get()) != base) {
            return false;
        }
        for (const auto & range : ranges) {
            if (ggml_backend_buffer_get_alloc_size(mapped_buffer.get(), range.tensor) != range.size) {
                return false;
            }
        }
        ggml_backend_buffer_set_usage(mapped_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & range : ranges) {
            void * address = static_cast<char *>(base) + (data_offset + range.offset - map_offset);
            if (ggml_backend_tensor_alloc(mapped_buffer.get(), range.tensor, address) != GGML_STATUS_SUCCESS) {
                for (const auto & reset : ranges) {
                    reset.tensor->buffer = nullptr;
                    reset.tensor->data = nullptr;
                }
                return false;
            }
        }
        buffer = mapped_buffer.release();
        mapping = std::move(owner);
        return true;
    } catch (...) {
        // All supported mapped buffer types have no tensor-init callback.
        // Allocations and driver calls that can throw precede tensor binding.
        return false;
    }
#else
    (void) path; (void) gguf; (void) tensors; (void) backend; (void) buffer; (void) mapping;
    return false;
#endif
}

}  // namespace transcribe::whisper
