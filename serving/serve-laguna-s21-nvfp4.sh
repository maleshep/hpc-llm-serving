#!/bin/bash
#SBATCH --account=<account>
#SBATCH --job-name=laguna-s21
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:b200:2
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00
#SBATCH --exclude=fat002,fat007
#SBATCH --output=/shared/project/<account>/llm/logs/laguna-s21_%j.out
#SBATCH --error=/shared/project/<account>/llm/logs/laguna-s21_%j.err

# Poolside Laguna S 2.1 NVFP4 + DFlash block-diffusion speculation.
# 118B total / 8.5B active MoE, 256 experts (top-10) + 1 shared, 48 layers
# (12 global attn + 36 sliding-window, window 512), GQA 8 KV heads dim 128.
# NVFP4 weights ~72GB. Native 1M context (ships at 256K, restore 1M via config.json
# rope_parameters — see "EXTEND TO 1M" below).
#
# SIDECAR placement (2×B200 — fits trivially; SWA makes KV cache ~6GB/GPU at 1M):
#   claude-glm      → primary FP8      → 8×B200 → 8103 → 1M → 7d QoS  (81% TB, gold standard)
#   glm52-reap      → REAP+DFlash 504B → 4×B200 → 8109 → 1M → 3d QoS  (70.5% TB, ~2.3x decode)
#   laguna-s21      → Laguna+DFlash     → 2×B200 → 8114 → 1M → 3d QoS  (70.2% TB, sidecar) ← THIS
#
# Why 2×B200: NVFP4 weights 72GB / 2 = 36GB/GPU; KV at 1M is ~1GB/GPU (only 12 of 48
# layers hold full-context KV; 36 are SWA window=512). ~50GB/GPU used, ~140GB free.
# TP=2 divides evenly (8 KV heads / 2 = 4 KV heads/GPU). PR #24204 tested TP=1 + TP=4
# on H200 (XS.2); TP=2 is untested but architecturally sound.
#
# DFlash drafter: poolside/Laguna-S-2.1-DFlash-NVFP4 (2.2GB, 6-layer, quantization-matched).
# Lossless (verify rejects wrong drafts) → quality stays at 70.2% TB. Expect ~2.5-3x
# decode tok/s per Poolside's DFlash card (measured at TP=2, num_speculative_tokens=15).
#
# NVFP4 serve notes (from HF card, adapted for SGLang/B200):
#   - Sampling defaults (temp 0.7, top_p 0.95) are NOT set server-side — the
#     container SGLang rejects --override-generation-config (unrecognized arg,
#     crashes at boot). The proxy/client sends these per-request instead.
#   - Card targets vLLM 0.25.1 + cu130 + FlashInfer nightly on GB10/sm_121. We run SGLang
#     in sglang-latest.sif on B200/sm_100 + CUDA 12.9 — different path. Flag any kernel
#     issues in logs and fall back to FP8 (serve-laguna-s21-fp8.sh) if NVFP4 misbehaves.
#   - Quantization auto-detected from config.json (no --quantization flag needed).
#
# B200 wedge flags (Case A + B, proven on GLM/REAP — kept as belt-and-suspenders):
#   --disable-flashinfer-autotune, --enforce-disable-flashinfer-allreduce-fusion.
# KV cache: fp8_e4m3 (not bare fp8 — SGLang container rejects bare fp8).
#
# Controlling reasoning: Laguna uses `enable_thinking` in chat_template_kwargs (NOT
# reasoning_effort). Server default ON here; proxy maps thinking toggle per-request.
# The poolside_v1 reasoning parser emits reasoning_content (same field as GLM) so
# existing proxy reasoning extraction works unchanged.
#
# UNVERIFIED assumptions (test on first boot):
#   1. poolside_v1 parser exists in sglang-latest.sif (PR #24204 merged May 2026).
#      If boot crashes with "unknown parser", drop --tool-call-parser/--reasoning-parser
#      and rely on --trust-remote-code chat template.
#   2. DFlash drafter loads cleanly on SGLang at TP=2 (we validated TP=4 on REAP).
#      --speculative-draft-model-quantization unquant kept from REAP; may be unneeded
#      since the DFlash-NVFP4 drafter is already NVFP4-matched. Try without if it errors.
#   3. Laguna softplus attention gating + SWA KV path on SM100 — no published B200 test.
#      Monitor for silent hang past "KV Cache is allocated" (Case B pattern).

set -euo pipefail

PROJECT=/shared/project/<account>
LLM_DIR=$PROJECT/llm
MODEL=/shared/scratch/<user>/models/laguna-s-2.1-nvfp4
DRAFTER=/shared/scratch/<user>/models/laguna-s-2.1-dflash-nvfp4
PORT="${PORT:-8114}"
NODE=$(hostname)
SIF=$LLM_DIR/containers/sglang-latest.sif

module load <hpc-module>/2509-fat
module load cuda/12.9.0
module load apptainer/1.4.1

[ ! -f "$SIF" ] && { echo "ERROR: $SIF not found"; exit 1; }
[ ! -d "$MODEL" ] && { echo "ERROR: Laguna model $MODEL not found — run download-laguna-s21-nvfp4.sh first"; exit 1; }
[ ! -d "$DRAFTER" ] && { echo "ERROR: DFlash drafter $DRAFTER not found — run download-laguna-s21-nvfp4.sh first"; exit 1; }

echo "=== Laguna S 2.1 NVFP4 + DFlash (2×B200, TP=2, port 8114) ==="
echo "NODE=$NODE PORT=$PORT JOB=$SLURM_JOB_ID MODEL=$MODEL DRAFTER=$DRAFTER"

LOCAL_TMP=/tmp/laguna-s21-$SLURM_JOB_ID
mkdir -p $LOCAL_TMP/triton $LOCAL_TMP/torch-ext $LOCAL_TMP/flashinfer-cubins $LOCAL_TMP/sglang-cache

cat > $LLM_DIR/.serve-state-laguna.json << EOF
{"job_id":"$SLURM_JOB_ID","node":"$NODE","port":$PORT,"model":"laguna-s-2.1","engine":"sglang-laguna-nvfp4-dflash","context_length":1048576,"started_at":"$(date -Iseconds)","status":"loading","tunnel_cmd":"ssh -L $PORT:${NODE}:$PORT -N <user>@<hpc-login>"}
EOF

# mem-fraction 0.80 reserves room for DFlash draft graph + draft KV.
# Context length env-driven for A/B: CTX_LEN (default 1048576 = native 1M).
#
# 1M context requires TWO fixes (CP2 of meta-harness-loop, problem: laguna-boot):
#   1. FLATTENED config.json — the pristine nested rope_parameters (full_attention/
#      sliding_attention keys) trips transformers v5 _validate_yarn_rope_parameters
#      (KeyError original_max_position_embeddings). The flattened 1M config merges
#      factor 32->128, attention_factor 1.346->1.485, sets max_position_embeddings=
#      1048576, AND hoists the full_attention rope block to the top level of
#      rope_parameters so the validator finds original_max_position_embeddings next
#      to rope_type=yarn. sliding_attention stays nested (validator ignores it).
#      Built offline from pristine + tmp_build_laguna_config.py; staged at
#      $PATCH_DIR/config_1m_flattened.json. Pristine preserved at config.json.pristine.
#   2. MAIN-branch sglang/srt/configs/laguna.py bind-mount — the container's
#      0.5.13.post1 laguna.py:160 has an UNGUARDED `self.layer_types.index(
#      "full_attention")` that raises in the server boot path (SGLang's
#      _CONFIG_REGISTRY routes model_type=laguna to its built-in LagunaConfig, not
#      the remote one). Main branch added `if "full_attention" in self.layer_types
#      else 0` + correct nested-rope conversion (full_rope_scaling/swa_rope_scaling).
#      Self-contained file (imports only transformers+typing). Same bind-mount
#      pattern as REAP DFlash PR#25943 isdir guard. Staged at $PATCH_DIR/laguna.py.main.
# 256K fallback: set CTX_LEN=262144 + use config.json.pristine (no flatten, no
#   bind-mount needed for the config, but the laguna.py guard bind-mount is still
#   required because the server path uses the built-in LagunaConfig regardless).
PATCH_DIR=/shared/project/<account>/llm/patches/laguna
CTX_LEN="${CTX_LEN:-1048576}"
# DFlash toggle: DFLASH_ON=1 = DFlash speculation; 0 (default) = plain serve.
# CP3 finding: container's modeling_laguna.py (0.5.13.post1) raises
#   "LagunaForCausalLM does not implement set_dflash_layers_to_capture" under DFlash
#   even though the method string is present — DFlash-on-Laguna integration is
#   incomplete in this container. DFlash is a follow-up trace (task #10). Default
#   OFF so resubmits boot reliably; opt in with --export=ALL,DFLASH_ON=1.
DFLASH_ON="${DFLASH_ON:-0}"
NGRAM_ON="${NGRAM_ON:-0}"
if [ "$DFLASH_ON" = "1" ] && [ "$NGRAM_ON" = "1" ]; then
    echo "ERROR: DFLASH_ON and NGRAM_ON are mutually exclusive"; exit 1
fi
if [ "$DFLASH_ON" = "1" ]; then
    DFLAGS="--speculative-algorithm DFLASH --speculative-draft-model-path /models/laguna-s-2.1-dflash-nvfp4 --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 16 --disable-cuda-graph --disable-piecewise-cuda-graph --moe-runner-backend triton --flashinfer-mxfp4-moe-precision bf16"
    DFLASH_PREFLIGHT="grep -q 'def set_dflash_layers_to_capture' /sgl-workspace/sglang/python/sglang/srt/models/laguna.py || { echo 'ERROR: DFLASH_ON=1 but modeling_laguna.py lacks set_dflash_layers_to_capture (container 0.5.13.post1 DFlash-on-Laguna incomplete). See meta-harness-loop DFlash trace task #10. Use DFLASH_ON=0.'; exit 1; } && "
    # CP1 (trace laguna-dflash): bind-mount surgically-patched modeling_laguna.py
    #   Container's models/laguna.py (787 lines) lacks DFlash capture logic (0 refs to
    #   layers_to_capture). Patched file (813 lines) ports main's DFlash hunks onto the
    #   container's import graph. A naive full-file main bind-mount crashes
    #   (ModuleNotFoundError: sglang.srt.runtime_context). Staged at $PATCH_DIR/modeling_laguna.py.dflash
    MODELING_LAGUNA_DFLASH="$PATCH_DIR/modeling_laguna.py.dflash"
    [ ! -f "$MODELING_LAGUNA_DFLASH" ] && { echo "ERROR: $MODELING_LAGUNA_DFLASH not found — stage patched modeling_laguna.py first"; exit 1; }
    # CP4: patched flashinfer_backend.py — guards prefix_lens=None in update_sliding_window
    #   (DFlash drafter forward has no prefix cache → prefix_lens is None → crash)
    FLASHINFER_PATCHED="$PATCH_DIR/flashinfer_backend_patched.py"
    [ ! -f "$FLASHINFER_PATCHED" ] && { echo "ERROR: $FLASHINFER_PATCHED not found"; exit 1; }
    DFLASH_DEBUG="$PATCH_DIR/dflash_debug.py"
    DFLASH_BIND="--bind $MODELING_LAGUNA_DFLASH:/sgl-workspace/sglang/python/sglang/srt/models/laguna.py --bind $FLASHINFER_PATCHED:/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py --bind $DFLASH_DEBUG:/sgl-workspace/sglang/python/sglang/srt/models/dflash.py"
    echo "DFlash ON (speculative-algorithm DFLASH, 16 draft tokens, with preflight + modeling_laguna.py.dflash bind-mount)"
elif [ "$NGRAM_ON" = "1" ]; then
    # CP9 (trace laguna-dflash): NGRAM speculation sidesteps the DFlash masked-MoE NaN.
    #   DFlash capture routes target hidden states through flashinfer_cutedsl_moe_masked
    #   (tree-mask cu_seqlens), which NaNs at the first full-attn layer on B200/SM100
    #   with NVFP4 expert weights (CP7/CP8 root cause). NGRAM needs no drafter, no
    #   target-hidden capture -> never enters the masked-MoE path -> reuses the plain
    #   causal prefill/decode kernels that already work at 113 tok/s. Lossless.
    #   Expert estimate: 1.3-1.5x on repetitive/code traffic -> ~150-170 tok/s.
    DFLAGS="--speculative-algorithm NGRAM --speculative-num-draft-tokens 4 --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 10"
    DFLASH_PREFLIGHT=""
    # CP9: NGRAM's tree-verify attention runs without prefix cache -> prefix_lens=None
    #   crashes update_sliding_window (same CP1/CP4 bug as DFlash). Bind the same
    #   flashinfer_backend_patched.py guard. Do NOT bind modeling_laguna.py.dflash
    #   (NGRAM captures no hidden states — the DFlash capture patch would change the
    #   return signature and break plain forward).
    FLASHINFER_PATCHED="$PATCH_DIR/flashinfer_backend_patched.py"
    [ ! -f "$FLASHINFER_PATCHED" ] && { echo "ERROR: $FLASHINFER_PATCHED not found"; exit 1; }
    DFLASH_BIND="--bind $FLASHINFER_PATCHED:/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py"
    echo "NGRAM ON (speculative-algorithm NGRAM, 4 draft tokens, no drafter — sidesteps DFlash masked-MoE NaN)"
else
    DFLAGS=""
    DFLASH_PREFLIGHT=""
    DFLASH_BIND=""
    echo "DFlash OFF (plain serve — base-model 1M, 113 tok/s baseline)"
fi
if [ "$CTX_LEN" = "1048576" ]; then
    cp "$PATCH_DIR/config_1m_flattened.json" "$MODEL/config.json"
    echo "Installed flattened 1M config.json (max_pos=1048576, rope flattened for v5 validator)"
else
    cp "$MODEL/config.json.pristine" "$MODEL/config.json"
    echo "Installed pristine 256K config.json (max_pos=262144)"
fi
# laguna.py guard bind-mount source (always needed — server path uses built-in LagunaConfig)
LAGUNA_PY_MAIN="$PATCH_DIR/laguna.py.main"
[ ! -f "$LAGUNA_PY_MAIN" ] && { echo "ERROR: $LAGUNA_PY_MAIN not found — stage main laguna.py first"; exit 1; }
apptainer exec --nv --cleanenv --writable-tmpfs \
    --env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64 \
    --env HOME=/tmp \
    --env TRITON_CACHE_DIR=$LOCAL_TMP/triton \
    --env TORCH_EXTENSIONS_DIR=$LOCAL_TMP/torch-ext \
    --env SGLANG_CACHE_DIR=$LOCAL_TMP/sglang-cache \
    --env SGLANG_ENABLE_SPEC_V2=1 \
    --env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    --bind $MODEL:/models/laguna-s-2.1-nvfp4 \
    --bind $DRAFTER:/models/laguna-s-2.1-dflash-nvfp4 \
    --bind $LLM_DIR:/data \
    --bind $LOCAL_TMP:$LOCAL_TMP \
    --bind $LOCAL_TMP/flashinfer-cubins:/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins \
    --bind $LAGUNA_PY_MAIN:/sgl-workspace/sglang/python/sglang/srt/configs/laguna.py \
    $DFLASH_BIND \
    $SIF \
    bash -c "$DFLASH_PREFLIGHT pip install -q -U --break-system-packages 'transformers>=5.3.0,<6' 2>&1 | tail -3 && \
        python3 -c 'import transformers; print(\"transformers=\", transformers.__version__)' && \
        python3 -m sglang.launch_server \
            --model-path /models/laguna-s-2.1-nvfp4 \
            --host 0.0.0.0 --port $PORT \
            --tp 2 \
            --trust-remote-code \
            $DFLAGS \
            --mem-fraction-static 0.80 \
            --cuda-graph-max-bs 32 \
            --max-running-requests 4 \
            --context-length $CTX_LEN \
            --chunked-prefill-size 8192 \
            --kv-cache-dtype fp8_e4m3 \
            --tool-call-parser poolside_v1 \
            --reasoning-parser poolside_v1 \
            --disable-flashinfer-autotune \
            --enforce-disable-flashinfer-allreduce-fusion \
            --served-model-name laguna-s-2.1" &
SERVER_PID=$!

READY=false
for i in $(seq 1 900); do
    sleep 5
    if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
        echo "SERVER READY after $((i*5))s"; READY=true; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER CRASHED"; exit 1; fi
    [ $((i % 12)) -eq 0 ] && echo "  ...loading ($((i*5))s)"
done

[ "$READY" != "true" ] && { echo "Timeout"; kill $SERVER_PID 2>/dev/null; exit 1; }

cat > $LLM_DIR/.serve-state-laguna.json << EOF2
{"job_id":"$SLURM_JOB_ID","node":"$NODE","port":$PORT,"model":"laguna-s-2.1","engine":"sglang-laguna-nvfp4-dflash","context_length":1048576,"started_at":"$(date -Iseconds)","status":"serving","tunnel_cmd":"ssh -L $PORT:${NODE}:$PORT -N <user>@<hpc-login>"}
EOF2

echo "SERVING laguna-s-2.1 on $NODE:$PORT (Laguna S 2.1 NVFP4, 2×B200, TP=2, ${CTX_LEN} ctx, DFlash=$DFLASH_ON)"
echo "=== Quick throughput probe ==="
curl -sf -X POST http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"Write a Python one-liner to reverse a list."}],"max_tokens":64,"temperature":0.7,"top_p":0.95}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); u=r.get('usage',{}); print('prompt=',u.get('prompt_tokens'),'completion=',u.get('completion_tokens'))" 2>/dev/null || echo "(probe skipped)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
wait $SERVER_PID
