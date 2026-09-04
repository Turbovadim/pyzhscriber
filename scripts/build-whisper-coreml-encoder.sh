#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/whisper-large-v3-turbo-QUANT.gguf" >&2
    exit 2
fi

MODEL_PATH=$1
MODEL_NAME=$(basename "$MODEL_PATH")
if [[ ! -f "$MODEL_PATH" || "$MODEL_NAME" != whisper-large-v3-turbo-*.gguf ]]; then
    echo "expected a whisper-large-v3-turbo GGUF model" >&2
    exit 2
fi

for tool in git uv xcrun ditto; do
    if ! command -v "$tool" >/dev/null; then
        echo "missing required tool: $tool" >&2
        exit 1
    fi
done

MODEL_DIR=$(cd "$(dirname "$MODEL_PATH")" && pwd)
DESTINATION="$MODEL_DIR/whisper-large-v3-turbo-encoder.mlmodelc"
if [[ -e "$DESTINATION" ]]; then
    echo "Core ML encoder already exists: $DESTINATION" >&2
    exit 1
fi

WORK_DIR=$(mktemp -d /tmp/pyzhscriber-coreml.XXXXXX)
cleanup() {
    rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT

WHISPER_CPP="$WORK_DIR/whisper.cpp"
VENV="$WORK_DIR/venv"
WHISPER_COMMIT=c4ac0012a8f5a2082dfca6aad4ddfd8b2c02b337

git clone --filter=blob:none --no-checkout https://github.com/ggml-org/whisper.cpp.git "$WHISPER_CPP"
git -C "$WHISPER_CPP" fetch --depth 1 origin "$WHISPER_COMMIT"
git -C "$WHISPER_CPP" checkout --detach FETCH_HEAD

uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" \
    'torch==2.5.0' \
    'coremltools==8.3' \
    'openai-whisper==20250625'
# The converter imports this package even though the faster standard graph
# does not use its ANE-specific rewrite. Its published Torch cap is obsolete.
uv pip install --python "$VENV/bin/python" --no-deps 'ane-transformers==0.1.3'

(
    cd "$WHISPER_CPP"
    "$VENV/bin/python" models/convert-whisper-to-coreml.py \
        --model large-v3-turbo \
        --encoder-only True \
        --optimize-ane False \
        --quantize True
)

SOURCE_MLPACKAGE="$WHISPER_CPP/models/coreml-encoder-large-v3-turbo.mlpackage"
INT8_MLPACKAGE="$WORK_DIR/coreml-encoder-large-v3-turbo-int8.mlpackage"
SOURCE_MLPACKAGE="$SOURCE_MLPACKAGE" INT8_MLPACKAGE="$INT8_MLPACKAGE" \
    "$VENV/bin/python" - <<'PY'
import os

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

model = ct.models.MLModel(os.environ["SOURCE_MLPACKAGE"])
config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
    )
)
linear_quantize_weights(model, config=config).save(os.environ["INT8_MLPACKAGE"])
PY

COMPILED_DIR="$WORK_DIR/compiled"
mkdir "$COMPILED_DIR"
xcrun coremlc compile "$INT8_MLPACKAGE" "$COMPILED_DIR"
ditto "$COMPILED_DIR/coreml-encoder-large-v3-turbo-int8.mlmodelc" "$DESTINATION"

echo "Core ML encoder installed: $DESTINATION"
du -sh "$DESTINATION"
