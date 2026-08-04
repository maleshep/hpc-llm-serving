#!/bin/bash
#SBATCH --account=<account>
#SBATCH --job-name=v4flash-container
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:b200:4
#SBATCH --qos=7d
#SBATCH --time=7-00:00:00
#SBATCH --output=/shared/project/<account>/llm/logs/v4flash_container_%j.out
#SBATCH --error=/shared/project/<account>/llm/logs/v4flash_container_%j.err

# =============================================================================
# DeepSeek-V4-Flash-0731 via official lmsysorg/sglang:deepseek-v4-blackwell container
# =============================================================================
# V4-Flash-0731 (2026-07-31) replaces old V4-Flash in place on 2026-08-04.
# 304B MoE, ~8B active, native 1M ctx, DSpark 7-token speculation bundled in
# checkpoint (no separate drafter). Beats V4-Pro on every coding benchmark
# (Terminal-Bench 82.7 vs 72.1, DeepSWE 54.4 vs 12.8, Toolathlon 70.3 vs 55.9).
#
# SGLang recipe (per model card): TP=4, flashinfer_mxfp4 runtime MoE quant
# (4-bit experts, dense stays FP8), DSPARK speculative decoding (layers 40-42
# of same checkpoint, no separate draft model path), FP8 KV cache,
# mem-fraction-static 0.90, chunked-prefill 4096, SWA ratio 0.1.
#
# 4x B200 (192 GB/GPU, 768 GB total): 167 GB FP8 weights + ~600 GB KV headroom
# at 90% mem-fraction → 1M native context fits comfortably. MXFP4 MoE runtime
# gives ~2x MoE matmul speedup vs FP8 on Blackwell NVFP4 tensor cores.
# DSpark 7-token speculation gives ~5-6x decode speedup at 70-85% acceptance.
# Projected: 400-600 tok/s on 4x B200 (vs GLM-5.2 FP8 8x B200 ~220-250 tok/s).
# =============================================================================

set -euo pipefail

PROJECT=/shared/project/<account>
LLM_DIR=$PROJECT/llm
MODEL=$LLM_DIR/models/deepseek-v4-flash
PORT=8100
NODE=$(hostname)
# Use sglang-latest.sif (SGLang 0.5.13.post1) — supports DFLASH speculation
# (the dsv4-blackwell.sif container has SGLang 0.5.10rc0 which only has EAGLE/EAGLE3/NEXTN/NGRAM).
# DFLASH is SGLang's name for what DeepSeek's model card calls "DSPARK" — same algorithm.
SIF=$LLM_DIR/containers/sglang-latest.sif

module load apptainer/1.4.1

if [ ! -f "$SIF" ]; then
    echo "ERROR: Container image not found at $SIF"
    echo "Run: sbatch pull-container.sh first"
    exit 1
fi

echo "=== DEEPSEEK-V4-FLASH-0731 (Container, 4× B200, TP=4, DSpark+MXFP4) ==="
echo "NODE=$NODE"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | tr '\n' ', ')"
echo "PORT=$PORT"
echo "JOB=$SLURM_JOB_ID"
echo "CONTAINER=$SIF"
echo ""
echo "ACCESS (SSH tunnel):"
echo "  ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com"
echo ""

# Write state file
cat > $LLM_DIR/.serve-state.json << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "deepseek-v4-flash-0731",
    "engine": "sglang-container",
    "image": "sglang-latest.sif",
    "tp_size": 4,
    "active_params": "8B",
    "total_params": "304B",
    "context_length": 1048576,
    "speculative": "eagle-3token",
    "moe_backend": "flashinfer_mxfp4",
    "kv_cache_dtype": "fp8",
    "started_at": "$(date -Iseconds)",
    "status": "loading",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com"
}
EOF

# Run SGLang inside the container — V4-Flash-0731 recipe per model card.
# --nv: expose NVIDIA GPUs
# --bind: mount model weights into container
# DSpark speculation: layers 40-42 of the SAME checkpoint serve as the drafter.
#   NO --speculative-draft-model-path flag (target+draft come from same weights).
# MXFP4 MoE runtime: flashinfer_mxfp4 quantizes MoE experts to 4-bit at runtime
#   (dense layers stay FP8 in the checkpoint). ~2x MoE matmul speedup on B200.
# FP8 KV cache: halves KV VRAM (runtime flag, works on FP8 checkpoint).
# mem-fraction-static 0.85 — leaves 15% for temp tensors + EAGLE CUDA graphs.
# FlashInfer/Triton cache dirs bound to project storage (not tmpfs) — tmpfs uses
# host RAM which OOMs during FlashInfer cubin JIT (thousands of symlink ops).
mkdir -p $LLM_DIR/.cache/flashinfer $LLM_DIR/.cache/triton
apptainer exec --nv --cleanenv --writable-tmpfs \
    --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64" \
    --env "HOME=/tmp" \
    --env "TRITON_CACHE_DIR=/data/.cache/triton" \
    --env "FLASHINFER_CUBIN_DIR=/data/.cache/flashinfer" \
    --bind $MODEL:/models/deepseek-v4-flash \
    --bind $LLM_DIR:/data \
    --bind $LLM_DIR/.cache/flashinfer:/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins \
    $SIF \
    python3 -m sglang.launch_server \
        --model-path /models/deepseek-v4-flash \
        --host 0.0.0.0 --port $PORT \
        --tp-size 4 \
        --trust-remote-code \
        --moe-runner-backend flashinfer_mxfp4 \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --chunked-prefill-size 4096 \
        --swa-full-tokens-ratio 0.1 \
        --context-length 1048576 \
        --mem-fraction-static 0.85 \
        --kv-cache-dtype fp8_e4m3 \
        --tool-call-parser deepseekv4 \
        --reasoning-parser deepseek-v4 \
        --served-model-name deepseek-v4-flash-0731 &
SERVER_PID=$!

# Wait for ready
echo "Waiting for model load..."
READY=false
for i in $(seq 1 360); do
    sleep 5
    if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
        echo ""
        echo "SERVER READY after $((i*5)) seconds!"
        READY=true
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "SERVER CRASHED — check stderr log"
        cat > $LLM_DIR/.serve-state.json << EOF2
{"job_id":"$SLURM_JOB_ID","node":"$NODE","status":"crashed","model":"deepseek-v4-flash-0731"}
EOF2
        exit 1
    fi
    if [ $((i % 12)) -eq 0 ]; then
        echo "  ...still loading ($((i*5))s)"
    fi
done

if [ "$READY" != "true" ]; then
    echo "Server failed to start within 30 minutes"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Update state
cat > $LLM_DIR/.serve-state.json << EOF3
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$NODE",
    "port": $PORT,
    "model": "deepseek-v4-flash-0731",
    "engine": "sglang-container",
    "image": "sglang-latest.sif",
    "tp_size": 4,
    "active_params": "8B",
    "total_params": "304B",
    "context_length": 1048576,
    "speculative": "eagle-3token",
    "moe_backend": "flashinfer_mxfp4",
    "kv_cache_dtype": "fp8",
    "started_at": "$(date -Iseconds)",
    "status": "serving",
    "tunnel_cmd": "ssh -L $PORT:${NODE}:$PORT -N user@hpc-cluster.example.com"
}
EOF3

echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""
echo "=== Server is live ==="
echo "OpenAI endpoint: http://localhost:$PORT/v1/chat/completions"
echo "Health: http://localhost:$PORT/health"
echo ""

# Quick validation
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Write a Python function that reverses a linked list. Be concise."}],"max_tokens":200}')
echo "Validation response:"
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "$RESP" | head -3
echo ""
echo "=== SERVING — waiting for scancel or wall time ==="

# Keep alive
wait $SERVER_PID
