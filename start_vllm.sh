#!/usr/bin/env bash
# =============================================================================
# start_vllm.sh — Launch vLLM OpenAI-compatible server in Docker
# Serves Qwen2.5-14B-Instruct-AWQ on port 8000
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL_REPO_DIR="${MODELS_DIR}/models--Qwen--Qwen2.5-14B-Instruct-AWQ"

# ------------------------------------------------------------------------------
# Locate the model directory.
# download_models.sh uses snapshot_download(local_dir=...) which places files
# directly in MODEL_REPO_DIR (no snapshots/<hash>/ subdirectory).
# ------------------------------------------------------------------------------
if [ ! -f "${MODEL_REPO_DIR}/config.json" ]; then
    echo "ERROR: Model not found at ${MODEL_REPO_DIR}"
    echo "Run download_models.sh first."
    exit 1
fi

# Path as seen *inside* the container (./models is mounted to /app/model)
CONTAINER_MODEL_PATH="/app/model/models--Qwen--Qwen2.5-14B-Instruct-AWQ"

echo "======================================================================"
echo "  Starting vLLM — Qwen2.5-14B-Instruct-AWQ"
echo "======================================================================"
echo "  Host models dir : ${MODELS_DIR}"
echo "  Container path  : ${CONTAINER_MODEL_PATH}"
echo "  Endpoint        : http://localhost:8000/v1"
echo ""

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker not found. Install Docker with GPU support."
    exit 1
fi

# Verify NVIDIA GPU is visible to Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "WARNING: Could not confirm GPU visibility in Docker."
    echo "         Ensure nvidia-container-toolkit is installed."
fi

echo "Launching Docker container (this will pull the image if not cached)..."
echo "Press Ctrl+C to stop the server."
echo ""

docker run --rm \
    --name vllm-qwen-agent \
    --gpus all \
    --ipc=host \
    -p 8000:8000 \
    -v "${MODELS_DIR}:/app/model" \
    vllm/vllm-openai:latest \
        --model "${CONTAINER_MODEL_PATH}" \
        --quantization awq_marlin \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.95 \
        --dtype half \
        --enforce-eager \
        --trust-remote-code \
        --served-model-name "Qwen2.5-14B-Instruct-AWQ"
