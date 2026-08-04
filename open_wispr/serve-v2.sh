#!/bin/bash
#SBATCH --account=hpc-llm
#SBATCH --job-name=open_wispr-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/hpc-llm/logs/open_wispr-v2_%j.out
#SBATCH --error=/shared/project/hpc-llm/logs/open_wispr-v2_%j.err

# =============================================================================
# OpenWispr v2 — Unified Model-Agnostic Server
# Single process on L40S (48 GB):
#   - SGLang (Qwen3.6-35B-A3B-FP8)  → port 8100 (~18 GB)
#   - Unified ASR+TTS server         → port 8200 (~3 GB total)
#     - ASR: faster-whisper large-v3-turbo int8 (~0.8 GB)
#     - TTS: F5-TTS (~1.5-2 GB)
#   Total: ~21 GB / 48 GB (down from ~34 GB)
# =============================================================================

set -e

PROJECT=/shared/project/hpc-llm
OPEN_WISPR=$PROJECT/model_training/open_wispr
NODE=$(hostname)
PIDS=()

echo "============================================"
echo "OpenWispr v2 — Unified Server"
echo "Node: $NODE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

# --- Load environment ---
module purge
module load hpc-cluster/2509
module load cli-tools

# Activate Python environment
source $PROJECT/venv/bin/activate 2>/dev/null || {
    echo "Creating venv..."
    cd $PROJECT
    uv venv venv --python 3.11
    source venv/bin/activate
}

# Install dependencies for new backends
echo "Installing dependencies..."
uv pip install torch torchaudio --quiet 2>/dev/null || true
uv pip install fastapi uvicorn python-multipart pyyaml --quiet 2>/dev/null || true
uv pip install faster-whisper --quiet 2>/dev/null || true
uv pip install f5-tts --quiet 2>/dev/null || true
uv pip install soundfile librosa --quiet 2>/dev/null || true
uv pip install sglang[all] --quiet 2>/dev/null || true

# --- 1. SGLang LLM (port 8100) ---
echo ""
echo ">>> Starting SGLang (Qwen3.6-35B-A3B-FP8) on port 8100..."
python -m sglang.launch_server \
    --model-path $PROJECT/models/Qwen3.6-35B-A3B-FP8 \
    --port 8100 \
    --host 0.0.0.0 \
    --tp 1 \
    --quantization fp8 \
    --context-length 32768 \
    --mem-fraction-static 0.35 \
    --disable-cuda-graph &
PIDS+=($!)
echo "SGLang PID: ${PIDS[-1]}"

# Wait for SGLang to be ready
echo "Waiting for SGLang to load model..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "SGLang ready after ${i}s"
        break
    fi
    sleep 1
done

# --- 2. Unified ASR+TTS Server (port 8200) ---
echo ""
echo ">>> Starting unified OpenWispr server on port 8200..."
cd $OPEN_WISPR
python -m uvicorn server:app --host 0.0.0.0 --port 8200 &
PIDS+=($!)
echo "OpenWispr PID: ${PIDS[-1]}"

# Wait for server to load models
echo "Waiting for ASR+TTS models to load..."
for i in $(seq 1 180); do
    if curl -s http://localhost:8200/health > /dev/null 2>&1; then
        echo "OpenWispr ready after ${i}s"
        break
    fi
    sleep 1
done

# --- All services ready ---
echo ""
echo "============================================"
echo "ALL SERVICES RUNNING"
echo "============================================"
echo "LLM:       http://${NODE}:8100  (Qwen3.6-35B-A3B-FP8)"
echo "ASR+TTS:   http://${NODE}:8200  (unified server)"
echo ""
echo "Endpoints:"
echo "  POST /transcribe   — speech-to-text"
echo "  POST /synthesize   — text-to-speech"
echo "  POST /clone        — zero-shot voice cloning"
echo "  GET  /health       — aggregate status"
echo "  POST /admin/swap-model — hot-swap backends"
echo "  WS   /ws/transcribe — streaming ASR"
echo ""
echo "TUNNEL COMMAND (run on Windows):"
echo "  ssh -L 8100:${NODE}:8100 -L 8200:${NODE}:8200 -N user@hpc-cluster.example.com"
echo ""
echo "VRAM usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "============================================"

# Write state file for tunnel reconnection
cat > $PROJECT/.open_wispr-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "version": "v2",
    "services": {
        "llm": {"port": 8100, "model": "Qwen3.6-35B-A3B-FP8"},
        "unified": {"port": 8200, "asr": "faster-whisper", "tts": "f5-tts"}
    },
    "tunnel_cmd": "ssh -L 8100:${NODE}:8100 -L 8200:${NODE}:8200 -N user@hpc-cluster.example.com",
    "started_at": "$(date -Iseconds)"
}
EOF

# --- Keep alive ---
cleanup() {
    echo "Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    echo "All services stopped."
}
trap cleanup SIGTERM SIGINT

wait
