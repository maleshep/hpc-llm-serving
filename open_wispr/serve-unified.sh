#!/bin/bash
#SBATCH --account=hpc-llm
#SBATCH --job-name=open_wispr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/hpc-llm/logs/open_wispr_%j.out
#SBATCH --error=/shared/project/hpc-llm/logs/open_wispr_%j.err

# Unified OpenWispr speech server.
# Starts a single FastAPI process on port 8200 which loads ASR/TTS backends
# from model_registry.yaml.

set -euo pipefail

PROJECT=/shared/project/hpc-llm
OPEN_WISPR=$PROJECT/model_training/open_wispr
NODE=$(hostname)
PORT=8200

echo "============================================"
echo "OpenWispr unified speech server"
echo "Node: $NODE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "============================================"

module purge
module load hpc-cluster/2509
module load cli-tools

source $PROJECT/venv/bin/activate 2>/dev/null || {
    echo "Creating venv..."
    cd $PROJECT
    uv venv venv --python 3.11
    source venv/bin/activate
}

echo "Installing OpenWispr dependencies..."
uv pip install torch torchaudio --quiet 2>/dev/null || true
uv pip install fastapi uvicorn python-multipart pyyaml --quiet 2>/dev/null || true
uv pip install faster-whisper --quiet 2>/dev/null || true
uv pip install f5-tts --quiet 2>/dev/null || true
uv pip install soundfile librosa --quiet 2>/dev/null || true

cd $OPEN_WISPR

echo ""
echo "Starting unified server on port $PORT ..."
python -m uvicorn server:app --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

echo "Waiting for server readiness..."
READY=false
for i in $(seq 1 180); do
    sleep 1
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        READY=true
        echo "Server ready after ${i}s"
        break
    fi
done

if [ "$READY" != "true" ]; then
    echo "Server failed to start in time"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "============================================"
echo "OPEN_WISPR READY"
echo "============================================"
echo "API:    http://${NODE}:$PORT"
echo "Tunnel: ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com"
echo "VRAM:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "============================================"

cat > $PROJECT/.open_wispr-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "server": "open_wispr-unified",
    "asr_backend": "faster-whisper",
    "tts_backend": "f5-tts",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com",
    "started_at": "$(date -Iseconds)"
}
EOF

cleanup() {
    echo "Shutting down OpenWispr..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT

wait $SERVER_PID
