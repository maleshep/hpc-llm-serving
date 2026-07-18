#!/bin/bash
#SBATCH --account=<account>
#SBATCH --job-name=glm52-reap
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:b200:4
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --exclude=fat-node-07
#SBATCH --output=/shared/project/<account>/llm/logs/glm52-reap_%j.out
#SBATCH --error=/shared/project/<account>/llm/logs/glm52-reap_%j.err

# GLM-5.2 REAP-504B (0xSero/GLM-5.2-504B): REAP-pruned (168/256 experts, 34%) + NVFP4 modelopt.
# ~309GB, 504B, native 1M context. Fits 1M KV on 4×B200 because the pruned 300GB
# footprint leaves ~1.46M tokens of fp8 KV headroom.
#
# Fleet placement: the long-context-on-half-hardware niche (1M ctx on 4 GPUs).
# 70.5% Terminal-Bench vs 81% FP8 primary — NOT a coding-model replacement.
#
# STANDALONE JOB (no fleet wiring in this reference). Verified boot + /health +
# /v1/models + 1M ctx only.
#
# The load-bearing boot fixes (vs a naive GLM NVFP4 script):
#   1. --enforce-disable-flashinfer-allreduce-fusion (Case B B200 wedge fix,
#      SGLang #29073 + PR #30945). Without it SGLang auto-enables AllReduce
#      Fusion on SM10X/B200, hangs in torch._symmetric_memory.rendezvous
#      post-KV-alloc. All GLM-on-B200 scripts need this.
#   2. DROP all --speculative-* (EAGLE). No pruned drafter exists for the 504B
#      expert set; two open crash bugs #30209 + #31093 for NVFP4+EAGLE.
#      Cost ~2.5x TPOT — acceptable for a long-ctx model.
#   3. in-container pip transformers>=5.3.0,<6 (NVIDIA req for glm_moe_dsa on
#      NVFP4; PEP-668 --break-system-packages). Pinned <6 to avoid 6.x break.
#   4. --kv-cache-dtype fp8_e4m3 (NOT bare 'fp8' — the SGLang container rejects it).
#   5. --exclude on any draining node.
#
# Sanitized reference copy. Replace <account>, login host, model path, SIF path.

set -euo pipefail

PROJECT=/shared/project/<account>
LLM_DIR=$PROJECT/llm
MODEL=/shared/scratch/<muid>/models/glm-5.2-reap-504b
PORT=8109
NODE=$(hostname)
SIF=$LLM_DIR/containers/sglang-latest.sif

module load hpc-env/2509-fat     # replace with your cluster's module stack
module load cuda/12.9.0
module load apptainer/1.4.1

[ ! -f "$SIF" ] && { echo "ERROR: $SIF not found"; exit 1; }
[ ! -d "$MODEL" ] && { echo "ERROR: REAP model $MODEL not found"; exit 1; }

echo "=== GLM-5.2 REAP-504B (NVFP4-pruned, 4×B200, 1M context, port $PORT, NO EAGLE) ==="
echo "NODE=$NODE PORT=$PORT JOB=$SLURM_JOB_ID"

LOCAL_TMP=/tmp/glm52-reap-$SLURM_JOB_ID
mkdir -p $LOCAL_TMP/triton $LOCAL_TMP/torch-ext $LOCAL_TMP/flashinfer-cubins $LOCAL_TMP/sglang-cache

cat > $LLM_DIR/.serve-state-glm52-reap.json << EOF
{"job_id":"$SLURM_JOB_ID","node":"$NODE","port":$PORT,"model":"glm-5.2-reap","engine":"sglang-reap-504b","context_length":1048576,"started_at":"$(date -Iseconds)","status":"loading","tunnel_cmd":"ssh -L $PORT:${NODE}:$PORT -N user@hpc-login.example.com"}
EOF

# In-container: install transformers>=5.3.0,<6 (NVIDIA req for glm_moe_dsa on NVFP4),
# then launch sglang. The `&` backgrounds the whole apptainer exec (so bash -c, pip,
# and sglang all live under SERVER_PID); `wait $SERVER_PID` keeps the job alive.
apptainer exec --nv --cleanenv --writable-tmpfs \
    --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64 \
    --env HOME=/tmp \
    --env TRITON_CACHE_DIR=$LOCAL_TMP/triton \
    --env TORCH_EXTENSIONS_DIR=$LOCAL_TMP/torch-ext \
    --env SGLANG_CACHE_DIR=$LOCAL_TMP/sglang-cache \
    --bind $MODEL:/models/glm-5.2-reap-504b \
    --bind $LLM_DIR:/data \
    --bind $LOCAL_TMP:$LOCAL_TMP \
    --bind $LOCAL_TMP/flashinfer-cubins:/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins \
    $SIF \
    bash -c "pip install -q -U --break-system-packages 'transformers>=5.3.0,<6' 2>&1 | tail -3 && \
        python3 -c 'import transformers; print(\"transformers=\", transformers.__version__)' && \
        python3 -m sglang.launch_server \
            --model-path /models/glm-5.2-reap-504b \
            --host 0.0.0.0 --port $PORT \
            --tp 4 \
            --quantization modelopt_fp4 \
            --trust-remote-code \
            --mem-fraction-static 0.85 \
            --max-running-requests 4 \
            --context-length 1048576 \
            --chunked-prefill-size 8192 \
            --kv-cache-dtype fp8_e4m3 \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --disable-flashinfer-autotune \
            --enforce-disable-flashinfer-allreduce-fusion \
            --served-model-name glm-5.2-reap" &
SERVER_PID=$!

READY=false
for i in $(seq 1 720); do
    sleep 5
    if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "SERVER READY after $((i*5))s"; READY=true; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER CRASHED"; exit 1; fi
    [ $((i % 12)) -eq 0 ] && echo "  ...loading ($((i*5))s)"
done

[ "$READY" != "true" ] && { echo "Timeout"; kill $SERVER_PID 2>/dev/null; exit 1; }

cat > $LLM_DIR/.serve-state-glm52-reap.json << EOF2
{"job_id":"$SLURM_JOB_ID","node":"$NODE","port":$PORT,"model":"glm-5.2-reap","engine":"sglang-reap-504b","context_length":1048576,"started_at":"$(date -Iseconds)","status":"serving","tunnel_cmd":"ssh -L $PORT:${NODE}:$PORT -N user@hpc-login.example.com"}
EOF2

echo "SERVING glm-5.2-reap on $NODE:$PORT (REAP-504B NVFP4, 1M context, max_running=4, NO EAGLE)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
wait $SERVER_PID
