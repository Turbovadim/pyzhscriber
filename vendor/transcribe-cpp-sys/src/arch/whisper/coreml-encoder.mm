#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "coreml-encoder.h"
#include "transcribe-log.h"

#include <cstring>

namespace transcribe::whisper {

struct WhisperCoreMlContext {
    const void * model;
};

namespace {

bool predict(MLModel *     model,
             int64_t       n_mel_frames,
             int64_t       n_mels,
             const float * mel,
             float *       output,
             size_t        output_elements) {
    NSError * error = nil;

    // transcribe.cpp stores mel frames contiguously. Core ML sees the same
    // allocation as [mel_bin, frame] through custom strides, avoiding a
    // 1.5 MB transpose for every utterance.
    MLMultiArray * input = [[MLMultiArray alloc]
        initWithDataPointer:const_cast<float *>(mel)
                      shape:@[@1, @(n_mels), @(n_mel_frames)]
                   dataType:MLMultiArrayDataTypeFloat32
                    strides:@[@(n_mels * n_mel_frames), @1, @(n_mels)]
                deallocator:nil
                      error:&error];
    if (input == nil) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "whisper Core ML: input creation failed: %s",
                error.description.UTF8String);
        return false;
    }

    MLDictionaryFeatureProvider * provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"logmel_data": [MLFeatureValue featureValueWithMultiArray:input]}
                     error:&error];
    if (provider == nil) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "whisper Core ML: input provider failed: %s",
                error.description.UTF8String);
        return false;
    }

    id<MLFeatureProvider> features = [model predictionFromFeatures:provider error:&error];
    if (features == nil) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "whisper Core ML: prediction failed: %s",
                error.description.UTF8String);
        return false;
    }

    MLMultiArray * encoded = [features featureValueForName:@"output"].multiArrayValue;
    if (encoded == nil || encoded.dataType != MLMultiArrayDataTypeFloat32 ||
        static_cast<size_t>(encoded.count) != output_elements) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "whisper Core ML: incompatible output (type=%ld, elements=%ld, expected=%zu)",
                static_cast<long>(encoded.dataType), static_cast<long>(encoded.count), output_elements);
        return false;
    }

    if (output != nullptr) {
        std::memcpy(output, encoded.dataPointer, output_elements * sizeof(float));
    }
    return true;
}

}  // namespace

WhisperCoreMlContext * coreml_encoder_init(const char * path,
                                           int64_t      n_mel_frames,
                                           int64_t      n_mels,
                                           size_t       output_elements) {
    @autoreleasepool {
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        // Let Core ML choose the fastest partition. On M5 Pro this beats
        // CPUAndNeuralEngine because some encoder operations favor the GPU.
        config.computeUnits = MLComputeUnitsAll;

        NSError * error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model == nil) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "whisper Core ML: load failed for '%s': %s", path,
                    error.description.UTF8String);
            return nullptr;
        }

        // Compile and specialize the Core ML execution plan while the app's
        // background model preload is running, not on the first dictation.
        NSMutableData * zero_mel = [NSMutableData dataWithLength:
            static_cast<NSUInteger>(n_mel_frames * n_mels * sizeof(float))];
        if (!predict(model, n_mel_frames, n_mels, static_cast<const float *>(zero_mel.bytes), nullptr,
                     output_elements)) {
            return nullptr;
        }

        auto * context = new WhisperCoreMlContext;
        context->model = CFBridgingRetain(model);
        return context;
    }
}

void coreml_encoder_free(WhisperCoreMlContext * context) {
    if (context == nullptr) {
        return;
    }
    CFRelease(context->model);
    delete context;
}

bool coreml_encoder_encode(const WhisperCoreMlContext * context,
                           int64_t                      n_mel_frames,
                           int64_t                      n_mels,
                           const float *                mel,
                           float *                      output,
                           size_t                       output_elements) {
    if (context == nullptr || mel == nullptr || output == nullptr) {
        return false;
    }
    @autoreleasepool {
        return predict((__bridge MLModel *) context->model, n_mel_frames, n_mels, mel, output,
                       output_elements);
    }
}

}  // namespace transcribe::whisper
