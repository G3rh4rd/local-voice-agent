#!/usr/bin/env bash
# =============================================================================
# download_models.sh — Pre-download all required models to ./models/
# Run once before starting the agent.
# Uses huggingface_hub Python API directly — no reliance on huggingface-cli
# being on PATH.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

# Resolve the venv Python (project venv preferred, then whatever is on PATH)
if [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PYTHON="${SCRIPT_DIR}/venv/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo "ERROR: python3 not found. Activate your venv: source venv/bin/activate"
    exit 1
fi

echo "======================================================================"
echo "  Model Download Script — Local Voice Agent"
echo "======================================================================"
echo "  Target directory : ${MODELS_DIR}"
echo "  Python           : ${PYTHON}"
echo "======================================================================"
echo ""

# Helper: download via huggingface_hub Python API
# Usage: hf_download <repo_id> <local_dir> [filename]
hf_download() {
    local repo_id="$1"
    local local_dir="$2"
    local filename="${3:-}"   # optional — download single file if set

    "${PYTHON}" - <<PYEOF
from huggingface_hub import snapshot_download, hf_hub_download
import sys, os

repo_id    = "${repo_id}"
local_dir  = "${local_dir}"
filename   = "${filename}" if "${filename}" else None

try:
    if filename:
        print(f"  Downloading {repo_id}/{filename} ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  Saved: {path}")
    else:
        print(f"  Downloading repo {repo_id} ...")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  Saved: {path}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ------------------------------------------------------------------------------
# 1. LLM — Qwen2.5-14B-Instruct-AWQ (quantized for RTX 4080, ~8 GB)
# ------------------------------------------------------------------------------
echo "[1/4] Downloading Qwen2.5-14B-Instruct-AWQ (LLM) — ~8 GB, be patient..."
hf_download "Qwen/Qwen2.5-14B-Instruct-AWQ" \
    "${MODELS_DIR}/models--Qwen--Qwen2.5-14B-Instruct-AWQ"
echo "[1/4] LLM downloaded OK."
echo ""

# ------------------------------------------------------------------------------
# 2. STT — Systran FasterWhisper Small (CPU int8 inference)
# ------------------------------------------------------------------------------
echo "[2/4] Downloading faster-whisper-small (STT)..."
hf_download "Systran/faster-whisper-small" \
    "${MODELS_DIR}/models--Systran--faster-whisper-small"
echo "[2/4] FasterWhisper downloaded OK."
echo ""

# ------------------------------------------------------------------------------
# 3. TTS — Kokoro ONNX (ultra-low latency CPU TTS)
#    Files come from the kokoro-onnx GitHub releases, not HuggingFace.
#    Model:  kokoro-v1_0.onnx
#    Voices: voices-v1.0.bin  (single npz-like binary, all voices bundled)
# ------------------------------------------------------------------------------
echo "[3/4] Downloading Kokoro TTS model + voices (GitHub releases)..."
mkdir -p "${MODELS_DIR}/kokoro"

KOKORO_RELEASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

# Download int8-quantized ONNX model (~88 MB) — fastest on Intel i9 CPU
echo "  Downloading kokoro-v1.0.int8.onnx ..."
curl -L --progress-bar \
    "${KOKORO_RELEASE}/kokoro-v1.0.int8.onnx" \
    -o "${MODELS_DIR}/kokoro/kokoro-v1.0.int8.onnx"

# Download all-voices binary (~100 MB, contains every voice)
echo "  Downloading voices-v1.0.bin ..."
curl -L --progress-bar \
    "${KOKORO_RELEASE}/voices-v1.0.bin" \
    -o "${MODELS_DIR}/kokoro/voices-v1.0.bin"

echo "[3/4] Kokoro TTS downloaded OK."
echo ""

# ------------------------------------------------------------------------------
# 4. VAD — Silero VAD (pre-cache torch.hub to avoid runtime download)
# ------------------------------------------------------------------------------
echo "[4/4] Pre-caching Silero VAD (torch.hub)..."
"${PYTHON}" - <<'PYEOF'
import torch, sys
print("  Pulling snakers4/silero-vad from torch.hub ...")
try:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    print(f"  Cached at: {torch.hub.get_dir()}")
except Exception as e:
    print(f"ERROR caching Silero VAD: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
echo "[4/4] Silero VAD cached OK."
echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo "======================================================================"
echo "  All models downloaded successfully!"
echo ""
echo "  Directory layout:"
find "${MODELS_DIR}" -maxdepth 3 -type d | sort | sed 's|^|    |'
echo ""
echo "  Next steps:"
echo "    1. ./start_vllm.sh      (in a separate terminal)"
echo "    2. python main.py"
echo "======================================================================"
