// Compile against the baseline header, with the same class-name macro as
// its translation unit, so later compute() arguments keep their old ABI.
#define MelFrontend BaselineMelFrontend
#include "transcribe-mel.h"
#undef MelFrontend

extern "C" void * mel_baseline_create(const transcribe::MelConfig * cfg) {
    return new transcribe::BaselineMelFrontend(*cfg);
}

extern "C" void mel_baseline_destroy(void * frontend) {
    delete static_cast<transcribe::BaselineMelFrontend *>(frontend);
}

extern "C" transcribe_status mel_baseline_compute(void * frontend, const float * pcm, size_t n_samples,
                                                 std::vector<float> * mel, int * n_mels, int * n_frames,
                                                 int n_threads) {
    return static_cast<transcribe::BaselineMelFrontend *>(frontend)->compute(
        pcm, n_samples, *mel, *n_mels, *n_frames, n_threads);
}
