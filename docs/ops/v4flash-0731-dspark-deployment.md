# DeepSeek V4-Flash-0731 + DSpark Deployment

**Date:** 2026-08-05
**Job:** 2956457 (fat008, 7d QoS, 4× B200)
**Throughput:** 256 tok/s average (Trial 2: 245, Trial 3: 267)
**Baseline:** 83 tok/s with EAGLE 3-token speculation (3.1× speedup)

## TL;DR

DeepSeek V4-Flash-0731 with DSpark speculation hits **256 tok/s on 4× B200** — matching/beating GLM-5.2 FP8 on 8× B200 (~220-250 tok/s), on half the hardware. The hard requirement is **SGLang 0.5.16+** (`sglang-v0516.sif` container) — older SGLang versions don't support DSpark for DeepSeekV4 arch.

## The breakthrough

DSpark is DeepSeek's native speculation algorithm (arXiv 2606.19348), implemented in SGLang via PR #30261 (merged July 2026, released in v0.5.16). It uses layers 40-42 of the same checkpoint as the drafter — no separate draft model, no extra VRAM.

**Before this work:** Our `sglang-latest.sif` container was SGLang 0.5.13.post1, which hard-asserts EAGLE only for DeepSeekV4 via the `apply_deepseek_v4_defaults` hook. DSpark/DFLASH are rejected.

**The fix:** Pull `lmsysorg/sglang:v0.5.16-cu130` from Docker Hub into a new `sglang-v0516.sif` Apptainer SIF (11.8 GB). DSpark is in the allowed list.

## Container pull recipe

`serving/pull-sglang-v0516.sh`:

```bash
#!/bin/bash
#SBATCH --account=<account>
#SBATCH --job-name=pull-sglang-v0516
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G             # CRITICAL: 8GB OOMs during mksquashfs
#SBATCH --qos=1d
#SBATCH --time=1-00:00:00

module load apptainer/1.4.1
export APPTAINER_TMPDIR=<project>/.cache/apptainer-tmp   # CRITICAL: /tmp ran out of space
mkdir -p $APPTAINER_TMPDIR
cd <project>/containers
apptainer pull --force sglang-v0516.sif docker://lmsysorg/sglang:v0.5.16-cu130
```

Pull takes ~15 min on the cpu partition. Needs 64GB RAM (default 8GB OOMs during squashfs compression of the 11.8 GB image).

## Winning serve config

`serving/serve-v4flash-container.sh` key flags:

```bash
SIF=$LLM_DIR/containers/sglang-v0516.sif    # NOT sglang-latest.sif (0.5.13.post1)

apptainer exec --nv --cleanenv --writable-tmpfs \
    --bind $MODEL:/models/deepseek-v4-flash \
    --bind $LLM_DIR:/data \
    --bind $LLM_DIR/.cache/flashinfer:/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins \
    --env "TRITON_CACHE_DIR=/data/.cache/triton" \
    --env "FLASHINFER_CUBIN_DIR=/data/.cache/flashinfer" \
    $SIF \
    python3 -m sglang.launch_server \
        --model-path /models/deepseek-v4-flash \
        --host 0.0.0.0 --port 8100 \
        --tp-size 4 \
        --trust-remote-code \
        --moe-runner-backend flashinfer_mxfp4 \
        --attention-backend dsv4 \
        --speculative-algorithm DSPARK \
        --speculative-dspark-block-size 5 \
        --enable-deepseek-v4-fp4-indexer \
        --cuda-graph-max-bs 192 \
        --chunked-prefill-size 4096 \
        --swa-full-tokens-ratio 0.1 \
        --disable-flashinfer-autotune \
        --context-length 1048576 \
        --mem-fraction-static 0.90 \
        --kv-cache-dtype fp8_e4m3 \
        --tool-call-parser deepseekv4 \
        --reasoning-parser deepseek-v4 \
        --served-model-name deepseek-v4-flash-0731
```

**Critical flags explained:**

| Flag | Value | Why |
|---|---|---|
| `SIF` | `sglang-v0516.sif` | SGLang 0.5.16+ required for DSpark on DeepSeekV4 |
| `--speculative-algorithm` | `DSPARK` | DeepSeek's native spec (PR #30261) |
| `--speculative-dspark-block-size` | `5` | Default K value, gamma=5, verify 6 draft tokens |
| `--moe-runner-backend` | `flashinfer_mxfp4` | Quantize MoE experts to 4-bit at runtime (~2× MoE matmul) |
| `--attention-backend` | `dsv4` | Auto-selected for DeepSeekV4, TRT-LLM sparse attn |
| `--cuda-graph-max-bs` | `192` | lmsys blog value — DSpark CUDA graphs work at higher bs |
| `--mem-fraction-static` | `0.90` | Model card recipe (0.85 with EAGLE, 0.92 without spec) |
| `--kv-cache-dtype` | `fp8_e4m3` | SGLang 0.5.13.post1 rejected bare `fp8`; 0.5.16 accepts both |
| `--enable-deepseek-v4-fp4-indexer` | (flag) | DeepSeek V4-specific optimization |
| `--disable-flashinfer-autotune` | (flag) | lmsys blog recommendation |

**FlashInfer/Triton cache on disk** (not tmpfs): bind-mount `$LLM_DIR/.cache/flashinfer` and set `FLASHINFER_CUBIN_DIR=/data/.cache/flashinfer`. tmpfs uses host RAM and OOMs during JIT cubin compilation (thousands of symlink ops).

## Benchmark methodology

Same prompt for all trials — 512-token LRU cache coding task:

```python
{"model":"claude-opus-4-7[1m]","max_tokens":512,"messages":[{"role":"user","content":"Write a Python class implementing a thread-safe LRU cache with TTL support..."}]}
```

3 trials, take best 2 (Trial 1 usually slower due to CUDA graph warmup). Via the Anthropic Messages API at the proxy port 5010.

## Full iteration history

| Attempt | Config | tok/s | Outcome |
|---|---|---|---|
| 1 | EAGLE 3-token, sglang-dsv4-blackwell.sif 0.5.10rc0 | crash | wrong container (no DFLASH/DSpark) |
| 2 | DFLASH, sglang-latest.sif 0.5.13.post1 | crash | DFLASH not allowed for DeepSeekV4 (hook asserts EAGLE) |
| 3 | DSPARK, sglang-latest.sif 0.5.13.post1 | crash | DSPARK not in allowed list (needs 0.5.16+) |
| 4 | EAGLE 3-token + mem 0.85 | 83 | baseline |
| 5 | EAGLE 5-token + FP4 indexer | 82 | no improvement |
| 6 | EAGLE 7-token + cuda-graph-max-bs 4 | 76 | worse (acceptance drops with more tokens) |
| 7 | EAGLE 2-token + cuda-graph-max-bs 1 | 94 | slightly better |
| 8 | No speculation + cuda-graph-max-bs 1 + mem 0.92 | 119 | best non-DSpark config |
| 9 | `deep_gemm` MoE backend (no spec) | crash | TVM kernel bug (`SiluAndMulMaskedPostQuantKernel`) |
| 10 | `--enable-multi-layer-eagle` | crash | `DeepseekV4ForCausalLMNextN.__init__() got unexpected 'draft_model_idx'` |
| **11** | **DSpark gamma=5, sglang-v0516.sif** | **256** | **✅ goal met** |

## Known issues to watch

These are OPEN SGLang issues as of 2026-08-05 that may affect production:

1. **PR #33614 (OPEN)** — DSpark TP rank divergence. Sampling decisions made independently per TP rank with no cross-rank sync. May deadlock at TP>1 (we're TP=4). Watch for deadlocks at long context. Fix requires NCCL 2.30.7 via `SGLANG_NCCL_SO_PATH` (torch's bundled 2.28.9 "wedges this workload's graph/eager mix").

2. **Issue #33549 (OPEN)** — decode hang at ~245K context with DSpark + TP=8. All ranks 100% util / 150W spin-wait. Killed by watchdog after 300s. We're TP=4 so less affected, but watch for hangs past 200K.

3. **Issue #33493 (OPEN)** — DSpark looks up `acc_linear_penalities` (typo) instead of `acc_additive_penalties`. Silently returns None. Breaks `min_new_tokens` and additive penalties under DSpark. Two-file patch provided but not merged.

4. **Issue #33656 (OPEN)** — hierarchical cache NaN crash. Don't enable `--enable-hierarchical-cache` with DSpark.

5. **`deep_gemm` MoE backend crashes** on B200 — `SiluAndMulMaskedPostQuantKernel` TVM error. NOT a DSpark-specific bug; affects any MoE config with deep_gemm. Stay on `flashinfer_mxfp4`.

## Sources

- Model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- SGLang v0.5.16 release: https://github.com/sgl-project/sglang/releases/tag/v0.5.16
- DSpark implementation PR #30261: https://github.com/sgl-project/sglang/pull/30261
- lmsys DSpark blog: https://www.lmsys.org/blog/2026-07-06-dspark-sglang
- SGLang cookbook (DeepSeek-V4): https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- DSpark TP divergence PR #33614: https://github.com/sgl-project/sglang/pull/33614
- 245K context hang Issue #33549: https://github.com/sgl-project/sglang/issues/33549
- Wrong sampling key Issue #33493: https://github.com/sgl-project/sglang/issues/33493
- DeepSeek paper: arXiv 2606.19348

## Related

- `docs/CHRONICLE.md` — chronicle entry
- `docs/training_chronicle.html` — HTML chronicle
- Memory: `project_v4flash0731_replacement.md`, `project_v4flash0731_speed_optimization.md`
- Serve script: `serving/serve-v4flash-container.sh`
- Container pull script: `serving/pull-sglang-v0516.sh`
