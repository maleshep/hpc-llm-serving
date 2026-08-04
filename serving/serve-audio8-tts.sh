#!/bin/bash
#SBATCH --account=hpc-llm
#SBATCH --job-name=audio8-tts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=1d
#SBATCH --time=1-00:00:00
#SBATCH --output=/shared/project/hpc-llm/llm/logs/audio8_tts_%j.out
#SBATCH --error=/shared/project/hpc-llm/llm/logs/audio8_tts_%j.err

# =============================================================================
# Audio8-TTS-Preview-0.6b — text-to-speech + zero-shot voice cloning engine
# 1x L40S (48GB), gpu partition, port 8310, QoS=1d (cheap boot, <2min).
#
# What this is:
#   Audio8/Audio8-TTS-Preview-0.6b — 0.6B DualAR TTS model (Apache 2.0).
#   Beats CosyVoice3-1.5B / Fish S2 Pro 4.6B / Higgs Audio v2 4.7B on English
#   WER (1.506) and CV3 zh/en multilingual error rates. 11 languages.
#   Zero-shot voice cloning from a single reference clip + transcript.
#   Bundled 44.1 kHz neural audio codec (no separate codec checkpoint).
#
# Why SGLang Omni (not vLLM, not transformers):
#   Audio8 ships a SGLang Omni adapter that exposes an OpenAI-compatible
#   /v1/audio/speech endpoint with paged attention + dynamic batching.
#   vLLM is not supported per the model card. Transformers works for
#   single-request inference but lacks batching — fine for dev, not for
#   serving. SGLang Omni is the production path.
#
# API (OpenAI-compatible):
#   POST /v1/audio/speech    — text -> WAV/PCM audio (OpenAI shape)
#     { "model": "audio8-tts", "input": "text...", "voice": "default"|"alloy"|...,
#       "response_format": "wav"|"pcm", "speed": 1.0 }
#   Reference-audio voice cloning: pass voice as a structured reference
#   (see Audio8 SGLang Omni adapter docs — the /v1/audio/speech schema accepts
#   a reference_clip + reference_transcript pair for zero-shot cloning).
#   GET  /v1/models          — list served models
#   GET  /health             — SGLang health probe
#
# No proxy/logger: standalone OpenAI-compatible endpoint, not part of the
# claude-* launcher fleet. Added to fleet dashboard for visibility only
# (tag: A8TTS). Eval target for open_wispr/ TTS backend swap — see
# serving/bench_tts.py.
#
# Verified 2026-08-04. See docs/ops/audio8-tts-serve.md (to be written).
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/hpc-llm
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/audio8-tts-0.6b
PORT=8310
NODE=$(hostname)

module purge
module load hpc-cluster/2509-update
module load cuda/12.9.0
module load apptainer/1.4.1

[ ! -d "$MODEL" ] && { echo "ERROR: $MODEL not found — run: sbatch serving/download-audio8-tts.sh"; exit 1; }

# SGLang Omni container — pull on first boot if missing.
# NOTE: SGLang Omni adapter for Audio8 pins a specific SGLang revision
# (see model card "Validated against a pinned SGLang Omni revision").
# Use the latest SGLang Omni nightly container; the adapter's API contract
# is stable across nightly bumps.
SIF=$LLM_DIR/containers/sglang-omni-latest.sif
if [ ! -f "$SIF" ]; then
    echo "Pulling SGLang Omni container..."
    apptainer pull "$SIF" docker://lmsysorg/sglang:omni-latest
fi

echo "=== Audio8-TTS-Preview-0.6b (SGLang Omni container, 1x L40S) ==="
echo "NODE=$NODE PORT=$PORT JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"

LOCAL_TMP=/tmp/audio8-$SLURM_JOB_ID
mkdir -p $LOCAL_TMP/triton $LOCAL_TMP/torch-ext $LOCAL_TMP/hf-cache $LOCAL_TMP/pip-cache

cat > $LLM_DIR/.serve-state-audio8.json << EOF
{"job_id":"$SLURM_JOB_ID","node":"$NODE","port":$PORT,"model":"audio8-tts","engine":"sglang-omni-container","context_length":2048,"vision":false,"asr":false,"tts":true,"started_at":"$(date -Iseconds)","status":"loading","tunnel_cmd":"ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com"}
EOF

# Audio8 SGLang Omni serve command. Key flags:
#   --served-model-name audio8-tts: matches the OpenAI /v1/audio/speech
#     "model" field the proxy/clients will send.
#   --trust-remote-code: Audio8 ships custom modeling code (DualAR arch).
#   --dtype bfloat16: per model card (BF16 is the inference precision).
#   --tensor-parallel-size 1: single L40S, model is ~1.2 GB — TP unnecessary.
#   --max-num-seqs 32: codec frame generation is sequential per request,
#     but multiple requests batch head; 32 is a comfortable ceiling for 48GB.
#   --max-model-len 2048: Audio8 context is "up to 2048 packed text/audio
#     positions" per card. Going higher buys nothing.
#   --gpu-memory-utilization 0.70: model + codec is tiny (~3 GB), but SGLang
#     pre-allocates KV cache aggressively on first request; 0.70 leaves
#     headroom for the audio codec decode workspace.
#   --chunked-prefill: helps with long reference clips for voice cloning.
#   --host 0.0.0.0 --port $PORT: bind on all interfaces (tunnel reaches it).
apptainer exec --nv --cleanenv \
    --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env HOME=/tmp \
    --env HF_HOME=$LOCAL_TMP/hf-cache \
    --env PIP_CACHE_DIR=$LOCAL_TMP/pip-cache \
    --env TRITON_CACHE_DIR=$LOCAL_TMP/triton \
    --env TORCH_EXTENSIONS_DIR=$LOCAL_TMP/torch-ext \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --bind $MODEL:/models/audio8-tts-0.6b \
    --bind $LOCAL_TMP:$LOCAL_TMP \
    $SIF \
    bash -c "
        set -e
        echo '[A] launching SGLang Omni serve for Audio8...'
        exec python3 -m sglang.launch_server \
            --model-path /models/audio8-tts-0.6b \
            --served-model-name audio8-tts \
            --trust-remote-code \
            --dtype bfloat16 \
            --tensor-parallel-size 1 \
            --max-num-seqs 32 \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.70 \
            --chunked-prefill \
            --host 0.0.0.0 \
            --port $PORT
    " &
SERVER_PID=$!

READY=false
for i in $(seq 1 60); do
    sleep 2
    if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "SERVER READY after $((i*2))s"; READY=true; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER CRASHED"; exit 1; fi
    [ $((i % 6)) -eq 0 ] && echo "  ...loading ($((i*2))s)"
done

[ "$READY" != "true" ] && { echo "Timeout waiting for /health"; kill $SERVER_PID 2>/dev/null; exit 1; }

cat > $LLM_DIR/.serve-state-audio8.json << EOF2
{"job_id":"$SLURM_JOB_ID","node":"$NODE","port":$PORT,"model":"audio8-tts","engine":"sglang-omni-container","context_length":2048,"vision":false,"asr":false,"tts":true,"started_at":"$(date -Iseconds)","status":"serving","tunnel_cmd":"ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com"}
EOF2

echo "SERVING audio8-tts on $NODE:$PORT"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
wait $SERVER_PID
