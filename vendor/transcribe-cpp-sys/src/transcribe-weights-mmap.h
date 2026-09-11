#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <memory>
#include <string>

struct gguf_context;

namespace transcribe::load_common {

// Non-owning backend buffers must be released before this mapping is destroyed.
struct WeightMapping {
    void * base = nullptr;
    size_t size = 0;

    WeightMapping() = default;
    WeightMapping(const WeightMapping &) = delete;
    WeightMapping & operator=(const WeightMapping &) = delete;
    ~WeightMapping();
};

// Bind an unallocated weight context directly to an immutable GGUF file.
// The file must not be modified in place while mapped; unlink/rename is safe.
// Failure leaves tensors and outputs untouched so the caller can stream them.
// The caller owns both outputs and must free buffer before destroying mapping.
bool try_map_weights(const std::string & path, const gguf_context * gguf,
                     ggml_context * tensors, ggml_backend_t backend,
                     ggml_backend_buffer_t & buffer,
                     std::unique_ptr<WeightMapping> & mapping) noexcept;

}  // namespace transcribe::load_common
