// Optional Core ML Whisper encoder bridge for Apple platforms.

#pragma once

#include <cstddef>
#include <cstdint>

namespace transcribe::whisper {

struct WhisperCoreMlContext;

WhisperCoreMlContext * coreml_encoder_init(const char * path,
                                           int64_t      n_mel_frames,
                                           int64_t      n_mels,
                                           size_t       output_elements);
void coreml_encoder_free(WhisperCoreMlContext * context);

// mel is frame-major [n_mel_frames, n_mels]. output is the conventional
// Whisper encoder layout [n_encoded_frames, d_model], with d_model contiguous.
bool coreml_encoder_encode(const WhisperCoreMlContext * context,
                           int64_t                      n_mel_frames,
                           int64_t                      n_mels,
                           const float *                mel,
                           float *                      output,
                           size_t                       output_elements);

}  // namespace transcribe::whisper
