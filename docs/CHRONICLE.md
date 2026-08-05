# LLM Training Pipeline Chronicle

> **This is a historical archive.** It records what happened and why, in reverse-chronological order.
> For current operational knowledge, see the reference docs in [`docs/ops/`](ops/):
> - [proxy.md](ops/proxy.md) — Proxy reasoning rules, max_tokens caps, context safety net
> - [serving-tuning.md](ops/serving-tuning.md) — EAGLE, chunked prefill, mem-fraction, throughput data
> - [training.md](ops/training.md) — LoRA config, data pipeline, eval results, SFT/DPO/GRPO
> - [deployment-lessons.md](ops/deployment-lessons.md) — Container flags, TP math, OOM fixes, storage
> - [client-tooling.md](ops/client-tooling.md) — claude-*, proxy-ai, Zed, Factory config
> - [headroom.md](ops/headroom.md) — Context compression, HPC proxy setup, token savings
>
> When adding new entries: record the narrative here, but capture the reusable lesson in the appropriate ops/ file.

---

## GLM-5.2 400+ tok/s Investigation — EAGLE-3 Path (2026-07-08)

**Status**: Flag tuning ruled out via A/B experiment. EAGLE-3 self-training is the only remaining lever. Corpus generation in progress.

### What Was Done

Goal: match provider-quoted 400+ tok/s single-stream on GLM-5.2 (currently ~215 tok/s on NVFP4 4×B200).

**Ruled out via A/B (job 2399732 on fat003:8107 vs production glm-alt fat002:8106):**
- `--speculative-num-draft-tokens 4` (down from 6): -4.4%
- `--kv-cache-dtype fp8_e4m3`: no gain
- `--fp4-gemm-backend flashinfer_trtllm`: no gain (auto already picks the fast path)
- `--cuda-graph-max-bs 64` (up from 32): no gain

Both configs settled at ~215 tok/s. Experimental job cancelled after benchmark.

**Feasibility confirmed** for EAGLE-3 self-training (SpecForge PR #493 template + PR #658 proves GLM-5.2 already runs in SpecForge; no DSA blockers documented; AQ-MedAI/GLM-5.1-eagle3 is the reference recipe).

**Kicked off Stage 1** (corpus generation): dedicated NVFP4 instance on fat003:8108, `serving/serve-glm52-corpus.sh`, resumable `corpus_gen.py` client with 32 concurrent workers pulling from `bigcode/self-oss-instruct-sc2-exec-filter-50k`. Target: 10K (prompt, completion) pairs.

### Key Findings

1. **Fireworks 446 tok/s is not reproducible with SGLang flag tuning.** The gap is architecture/kernel-level, not config-level.
2. **NVFP4 4×B200 with EAGLE-2 hits its ceiling around 215-220 tok/s**, independent of the 4 tuning knobs we tried.
3. **No public GLM-5.2 EAGLE-3 draft exists on HF** — only GLM-5.1 and GLM-4.7 variants. Must train our own.
4. **Corpus gotcha**: `training/data/sft_train_v2.jsonl` is MMM-domain (pharma Bayesian optimization). Wrong distribution for a coding drafter — need coding-heavy corpus.
5. **Windows unicode gotcha**: `corpus_gen.py` initially crashed on `UnicodeEncodeError` printing progress. Fix: `PYTHONIOENCODING=utf-8` on launch.

### Isolation Discipline

Every pipeline job uses `--exclude=fat-node-002,fat-node-007`, its own state file, its own port (8107/8108/8109), and distinct `--served-model-name`. Production glm-alt and primary GLM are never touched. See [docs/ops/eagle3-training.md](ops/eagle3-training.md).

---

## Cost Dashboard — hpc-alt Traffic Silently Dropped (2026-07-08)

**Status**: Fixed. 11.4M tokens of hpc-alt traffic (1,838 rows over 7d) had been invisible in the dashboard.

### What Was Done

`claude-glm-alt` writes to `hpc-alt.jsonl` with `route: "hpc-alt"`. The dashboard's `parseLine` (in `serving/llm-cost-dashboard/lib/cost-aggregator.ts`) rejected any row whose route wasn't `palantir | hpc | uptimize` — dropping all hpc-alt rows at parse time. HPC filter tab showed 8M tokens; actual HPC total was 19.4M.

Five edits:
1. `cost-aggregator.ts::parseLine` — accept `hpc-alt`
2. `cost-types.ts::calcSpend` — treat hpc-alt as zero-cost via new `isHpc()` helper
3. `cost-types.ts::filterByRoute` — HPC filter now includes hpc-alt
4. `cost.tsx` — added color `#4CD9A5` for hpc-alt badge (distinct from hpc green)
5. `cost.tsx` — "HPC Share" KPI sums both routes

### Key Finding

**Parser allowlist was the silent killer.** The aggregator's daily rollups (spend, tokens) already had else-branches that would have caught unrecognized routes as "hpc". But `parseLine` filtered at the door — any new route name (uptimize, hpc-alt, future variants) is dropped unless explicitly whitelisted. Documented in [docs/ops/client-tooling.md](ops/client-tooling.md#cost-dashboard).

---

## Proxy Session Memory Leak — claude-glm-alt Poisoning (2026-07-08)

**Status**: Fixed by disabling `SESSION_MEMORY_ENABLED` on glm-alt only.

### What Was Done

`~/.local/bin/proxy-ai.cmd` glm-alt path now sets `SESSION_MEMORY_ENABLED=0` (in addition to existing `SUMMARIZE_ENABLED=1`). Other proxies untouched.

### Key Finding

`server.py` in `claude-code-proxy` has a **process-global `_session_memory: list`** (line 275) that accumulates facts from every Claude Code session hitting the process, then injects them into the next request's system message — regardless of session boundary. Cross-session poisoning was inevitable.

The Tier 4 fallback that fills `_session_memory` (line 542, `_session_memory.extend(facts)`) fires more often on glm-alt because summarization pressure is higher there. That's why glm-alt showed the leak first — but the underlying bug affects any proxy with `SESSION_MEMORY_ENABLED=1`.

**Followup deferred**: proper redesign of `server.py` (key `_summary_cache` by `metadata.user_id`, delete `_session_memory` entirely, add 30-min TTL) — out of scope for glm-alt-only fix.

---

## MiniMax M3 Multimodal Enabled (2026-06-22)

**Status**: Vision+video enabled on MiniMax M3 via hot-swap. Zero downtime. No SGLang needed — vLLM already supported M3 multimodal, just needed to remove `--language-model-only` flag.

### What Was Done

Hot-swapped MiniMax M3 from language-only to multimodal using a second fat node:
1. Launched new vLLM instance on fat008:8106 without `--language-model-only` (old instance on fat006:8105 kept running)
2. Verified: text generation works (117 tok/s), image input works (200 OK), old instance rejects images (400)
3. Switched traffic to new instance, cancelled old job on fat006
4. Updated canonical serve script to port 8105 with `--max-model-len 811648`

### Key Finding

vLLM's M3 container (`vllm/vllm-openai:minimax-m3`) already had full multimodal support — the model class `MiniMaxM3SparseForConditionalGeneration` is natively multimodal. The `--language-model-only` flag was the only thing blocking vision. No SGLang switch needed.

SGLang dev container pull (`lmsysorg/sglang:dev-minimax-m3`) was attempted but failed — `mksquashfs` crashed with illegal instruction on RHEL9. Moot point since vLLM works.

### Impact

| Metric | Before (language-only) | After (multimodal) |
|--------|----------------------|-------------------|
| Vision | Disabled (400 on image input) | **Enabled** (200 on image input) |
| Context | 1M | 811K (vision encoder cache takes ~20GB KV) |
| tok/s | 118.6 | 117.4 (noise — vision encoder skipped for text) |
| VRAM/GPU | 143GB | 152GB |
| Tool calling | Enabled | Enabled |
| Reasoning | Enabled | Enabled |

The 812K context is more than sufficient — Claude Code auto-compacts at ~170K.

### Apptainer Cache Cleanup

The failed SGLang container pull filled `~/.apptainer/cache` with 86GB, hitting the 100GB home directory soft limit. Cleared cache, home back to 13GB.

→ See [`docs/ops/client-tooling.md`](ops/client-tooling.md) for updated MiniMax M3 config and multimodal usage

---

## Headroom Context Compression — 26.2% Token Savings (2026-06-18)

**Status**: Headroom 0.27.0 deployed on HPC with SmartCrusher (Rust core). SSH tunnel from local to HPC:8787. Achieved 26.2% token reduction on Foundry tool_result content.

### What Was Done

Installed headroom-ai 0.27.0 on HPC (Rust core works — downloads succeed on HPC network). Local Windows stuck at 0.20.15 (pure Python, no SmartCrusher — corporate network blocks Rust toolchain SSL cert). Solution: run headroom on HPC, SSH tunnel local:8787 → HPC:8787.

### Results

| Metric | Value |
|--------|-------|
| Token savings | 9,443 → 6,965 (**26.2%**) |
| Pipeline overhead | 97ms |
| TTFT overhead | +1.9s (amortizes on long convos) |
| Decode tok/s impact | None (166-167 tok/s both paths) |

### Key Decisions

- **HPC as compression proxy**: Since local Windows can't run Rust core, HPC serves as the headroom endpoint. SSH tunnel makes it transparent to local clients.
- **Not enabled for 1M context models**: V4-Flash/Pro have 1M context — compression unnecessary.
- **Opt-in via `--headroom` flag**: `proxy-ai glm --headroom` enables compression. Without flag, direct tunnel as before.

### What's Next

1. Route Foundry/Anthropic API traffic through HPC headroom for automatic tool_result compression
2. Benchmark on full multi-turn Foundry sessions (not just sample content)
3. Compare token usage: Foundry Anthropic access vs HPC Chinese models (Task #33)
4. Explore SmartCrusher compression profiles for different content types

→ See [`docs/ops/headroom.md`](ops/headroom.md) for full setup details

---

## GLM-5.2 Deployment + Fleet Rationalization (2026-06-17)

**Status**: GLM-5.2-FP8 deployed, hot-swapped from GLM-5.1. Kimi K2.7 deprecated. Fleet is now GLM-5.2 (primary) + MiniMax M3 (multimodal/long-context).

### GLM-5.2 — Surprise Drop, Massive Leap

GLM-5.2 released Jun 16-17, 2026 — completely unexpected, just 3 weeks after 5.1. Same model size (756GB FP8, same `GlmMoeDsaForCausalLM` architecture), same hardware footprint (8×B200), but with two architectural improvements:

- **IndexShare**: reuses sparse attention index across every 4 layers → 2.9× less FLOPs at 1M context → faster decode at long contexts even without EAGLE
- **MTP improved**: +20% EAGLE acceptance length → measured ~220-250 tok/s (was ~175-185 on 5.1)
- **1M context native** (was 202K on 5.1) — closes the context gap with MiniMax M3

**Benchmark gains over GLM-5.1 (self-reported):**
- Terminal-Bench 2.1: 63.5% → **81.0%** (+17.5 pts — largest jump in the table)
- FrontierSWE: 30.5% → **74.4%** (+43.9 pts — 2.4× improvement)
- SWE-bench Pro: 58.4% → **62.1%** (+3.7 pts)
- AIME 2026: 95.3 → **99.2**

Download and hot-swap completed in ~2.5 hours total. SGLang upgraded from 0.5.12 → 0.5.13.post1 to support GLM-5.2.

### The MiniMax M3 Reality Check

MiniMax M3 was marketed on its MSA sparse attention enabling 1M context efficiently — that positioning was accurate. However, benchmarks (from GLM-5.2's own table, which includes M3 as a competitor) reveal where M3 actually sits:

| Benchmark | GLM-5.2 | MiniMax M3 | Gap |
|-----------|---------|------------|-----|
| SWE-bench Pro | 62.1% | 59.0% | GLM+3.1 |
| Terminal-Bench 2.1 | **81.0%** | 65.0% | **GLM+16** |
| HLE | 40.5 | 37.0 | GLM+3.5 |
| GPQA-Diamond | 91.2 | **93.0** | M3+1.8 |
| MCP-Atlas | 76.8% | 74.2% | GLM+2.6 |

**Key insight**: MiniMax M3's strength is its attention *architecture* (MSA for 1M ctx efficiency), not its raw benchmark performance. On coding tasks it's squarely mid-tier — better than GLM-5.1 but clearly below GLM-5.2. M3's GPQA-Diamond lead (93% vs 91.2%) shows stronger science/knowledge reasoning, but for software engineering tasks GLM-5.2 dominates.

M3's genuine unique value: **multimodal** (image+video inputs), **MSA efficiency at 1M tokens** (15× decode speedup vs GQA at 1M), **64ms TTFT** (vs 206ms for GLM). Keep it for those use cases.

### Fleet after rationalization

- Kimi K2.7 deprecated (was 3rd-best, cluster capacity better used elsewhere)
- GLM-5.2: ~220-250 tok/s, primary coding, 1M ctx, best open-source Terminal-Bench
- MiniMax M3: ~119 tok/s, 64ms TTFT, multimodal, 1M ctx efficiency

---

## Fleet Expansion: Kimi K2.7 + MiniMax M3 (2026-06-12 to 2026-06-15)

**Status**: Fleet expanded from 2 to 3 coding models. Kimi hot-swapped K2.6→K2.7. MiniMax M3 MXFP8 deployed after a multi-day debugging saga. Daily fleet now: GLM-5.1 (188 tok/s) + Kimi K2.7 (~112 tok/s) + MiniMax M3 (~75-80 tok/s).

### Kimi K2.7-Code Hot-Swap

Kimi K2.7-Code dropped Jun 12 — same architecture as K2.6 (drop-in replacement), +22% coding benchmarks, -30% thinking tokens. Downloaded via `snapshot_download` on fat/B200 node (595GB compressed-tensors format). Key lesson: `hf` CLI stalls on large shards on gpu/L40S nodes; always use fat/B200 for downloads. Hot-swapped K2.6 → K2.7 in-place (same port 8104, same vLLM serve script adjusted for new model name).

### MiniMax M3 MXFP8 — Deployment Saga

MiniMax M3 (428B MoE, 23B active, 1M context) dropped Jun 12. Required the `vllm/vllm-openai:minimax-m3` Docker image because M3 isn't in any stable vLLM pip release. Running Docker containers via Apptainer on HPC produced a cascade of failures over ~2 days:

1. **DeepGEMM JIT**: `/workspace/.deps/deepgemm-src/csrc/jit/compiler.hpp` — `nvcc` not found → fixed with `CUDA_HOME` + Spack bind
2. **gcc library mismatch**: Spack GCC (RHEL9) can't find Ubuntu glibc headers inside container → fixed with `CC=/usr/bin/gcc` (use container's Ubuntu gcc for Triton, not host Spack gcc)  
3. **libcudart.so.12**: `LD_LIBRARY_PATH` with GCC libs overrode singularity GPU driver libs → fixed by ordering `$CUDA_PREFIX/lib64` first
4. **VLLM_USE_BREAKABLE_CUDAGRAPH=0**: Appeared to be the AMD "use full cudagraphs" flag but in this container build disables cudagraphs entirely → removed, let vLLM auto-enable breakable cudagraphs
5. **MSA warmup**: `_gqa_sparse_decode_kernel` Triton JIT during first inference → spikes on first request, cached afterward in `/tmp/triton-cache`

Final config: TP=4, `--enable-expert-parallel`, breakable cudagraphs, `CC=/usr/bin/gcc`, `CUDA_HOME=$CUDA_PREFIX`. Achieves **~75-80 tok/s** — usable as a daily driver. Weights load in **46 seconds** (was 365s) once Triton + torch-extensions caches are warm.

**MiniMax M3 vs GLM/Kimi**: Not a speed competitor — its unique value is 1M context, multimodal (image+video), and MSA sparse attention which gives 15× decode speedup at 1M vs dense attention. Use for large codebase analysis, document ingestion, image/screenshot tasks.

### Benchmarks (Jun 2026, self-reported)

| Model | SWE-Bench Pro | MCP Atlas | Kimi Code Bench | Tok/s | GPUs |
|-------|--------------|-----------|-----------------|-------|------|
| GLM-5.1 | 73.8% | — | — | **188** | 8×B200 |
| Kimi K2.7 | — | 76.0% | 62.0% | **~112** | 4×B200 |
| MiniMax M3 | 59.0% | 74.2% | — | **~75-80** | 4×B200 |

---

## Fleet Simplification, Kimi 2× Speedup, Proxy Quality Fixes (2026-06-03 to 2026-06-05)

**Status**: Daily fleet simplified to GLM-5.1 + Kimi K2.6. Kimi throughput doubled. Proxy quality improvements deployed. Nemotron-3-Ultra researched and downloaded. Fine-tuning roadmap planned.

### Fleet Decision: GLM + Kimi Only

After extensive benchmarking and cluster analysis, simplified the daily serving fleet:
- **GLM-5.1 FP8** (8× B200): PRIMARY — SWE-bench 73.8%, 188 tok/s, full thinking control
- **Kimi K2.6 NVFP4** (4× B200): ALT — Terminal-bench 67.2%, always-thinking, 112 tok/s

V4-Flash/Pro remain on disk and launchable but not in daily rotation. GLM FP8 is production; NVFP4 copy deleted (post-hoc quantization, lower quality).

### Kimi TOKENSPEED_MLA — 2× Throughput

Added `--attention-backend TOKENSPEED_MLA --kv-cache-dtype fp8 --max-num-batched-tokens 32768` to `serve-kimi-k2.sh`. CuTe DSL kernel (vLLM PR #41778) specifically for MLA on SM100/B200. **Measured: 106-112 tok/s** (was 50-70 tok/s). 2.0-2.3× speedup at batch≥8. → See `docs/ops/serving-tuning.md`

Also fixed Kimi startup: `VLLM_DISABLE_COMPILE=1` + `VLLM_WORKER_MULTIPROC_METHOD=spawn` prevents the shm_broadcast hang caused by Ninja build failure on SM100 flashinfer cache. → See `docs/ops/deployment-lessons.md`

### Proxy Quality Improvements

Extended two Gemma-only proxy fixes to all HPC models:
1. **system-reminder stripping**: Claude Code injects skill blobs mid-conversation. These confuse open-source models (not trained on Claude's agentic protocol). Now stripped for GLM, Kimi, DeepSeek, Gemma, Nemotron, Qwen.
2. **Tool schema simplification**: 140 tools × verbose JSON schemas = ~100K tokens per request. Simplification (truncate descriptions, collapse anyOf) saves ~50K tokens. Extended to all HPC models.

Also fixed GLM max_tokens cap (16384→4096/8192 EAGLE values) which was causing 400 errors on long conversations. BACKEND_CONTEXT_LIMIT lowered to 186K for GLM.

### Cluster Learnings

Cluster was frequently full. Key learnings:
- Other teams' jobs are invisible — `squeue` only shows your account by default. What looked like ghost allocations were real jobs.
- `squeue -p fat --all` shows all users' jobs; `-w NODE` shows jobs on a specific node
- `1d` QOS (priority 80) always beats `3d` (40) with same fairshare cost
- Fairshare driven by GPU hours consumed, not QOS level

### Knowledge Reorganization

Restructured docs into three tiers: lean CLAUDE.md (ops runbook) + docs/ops/ (domain reference) + CHRONICLE.md (archive). Added `docs/ops/deploy-new-model.md` as 11-step agentic workflow.

### Nemotron-3-Ultra Research

Researched and downloaded Nemotron-3-Ultra-550B-A55B NVFP4 (329GB, June 2026). Key findings:
- Fits on 4× B200, 1M context, native MTP 5-step EAGLE, ~120-160 tok/s estimated
- SWE-bench 69.7% (NVFP4) vs GLM-5.1 73.8% — GLM wins on coding
- Genuine strength: 1M context (RULER 94%), Mamba-2 O(1) KV for long-context
- OmniScience "non-hallucination" advantage is a cherry-pick (abstention metric, not accuracy)
- Deferred deployment — not better than GLM+Kimi for current use cases

### Next: MoE Fine-Tuning

Documented full Gemma-4 fine-tuning pipeline (SFT+GRPO+DPO). SFT-only was best (59% vs Sonnet). Next plan: scale to MoE models starting with GLM-5.1. Requires DeepSpeed ZeRO-2, 8× B200 training allocation, FP8 mixed-precision training. → See `docs/superpowers/specs/` for implementation plan.

---

## DP+EP Research & Chunked Prefill TTFT Fix (2026-05-22)

**Status**: V4-Pro TTFT reduced from ~60s to 7.4s. GLM confirmed 188 tok/s. Kimi chunked prefill applied.

### The Problem

V4-Pro (`claude-dsp`) was taking **~1 full minute** per response in agentic workloads (job hunt pipeline with large conversation context). Meanwhile GLM-5.1 remained fast. Root cause investigation pointed to prefill time on 49B active params with large context windows.

### Research: SGLang DP+EP+DPA Optimization

From SGLang official docs (`dp_dpa_smg_guide.mdx`, `expert_parallelism.md`, Kimi K2 cookbook):

| Strategy | What It Does | Expected Gain |
|----------|-------------|---------------|
| `--dp-size N --enable-dp-attention` | Data Parallelism for attention — eliminates KV cache duplication across TP ranks | 2-4× TTFT on long context |
| `--ep N` | Expert Parallelism — distributes MoE experts across GPUs instead of replicating | ~30-50% throughput at high concurrency |
| `--moe-a2a-backend deepep` | DeepSeek's all-to-all kernel for expert dispatch | Lower EP communication overhead |
| `--chunked-prefill-size 4096` | Smaller prefill chunks interleave with decode, reducing head-of-line blocking | ~10-30% TTFT reduction |

**Critical finding**: DP+EP is a **throughput optimization for multi-user**, NOT a latency optimization for single-user agentic workloads. With DP=8, each request goes to 1/8th of GPUs — catastrophic for single-stream latency.

### What Was Tested & Results

| Model | Optimization | Result |
|-------|-------------|--------|
| **V4-Pro** | DP+EP+DPA (`--dp-size 8 --ep 8 --enable-dp-attention --moe-a2a-backend deepep`) | **CRASHED** — Container SGLang 0.5.10rc0 DeepEP assertion error during CUDA graph capture |
| **V4-Pro** | `--chunked-prefill-size 4096` (was 8192) | **SUCCESS** — 80K input: 60s → **7.4s** |
| **GLM-5.1** | DP+EP (`--dp-size 8 --ep 8 --enable-dp-attention`) | Server launched OK, but single-request throughput dropped 167→95 tok/s (request routed to 1/8 GPUs) |
| **GLM-5.1** | Reverted to TP=8 only | **188 tok/s** (improvement from SGLANG_ENABLE_SPEC_V2=1 applied in prior session) |
| **Kimi K2.6** | `--enable-chunked-prefill --max-num-batched-tokens 4096` | **70 tok/s**, 80K input in 21s |
| **Kimi K2.6** | `--num-scheduler-steps 8` | **FAILED** — flag doesn't exist in vLLM 0.21 |

### Benchmark Results (80K input tokens, 200 output, single request)

| Model | Time | tok/s (short) | Hardware | Active Params |
|-------|------|---------------|----------|--------------|
| **V4-Pro** | **7.4s** | ~163 | 8× B200 | 49B |
| **GLM-5.1** | **7.7s** | **188** | 8× B200 | 40B |
| **Kimi K2.6** | 21.0s | 70 | 4× B200 | 32B |

### Why Kimi Is Slowest Despite Fewest Active Params

Three compounding factors:
1. **4× B200 vs 8× B200** — half the memory bandwidth (1536GB/s vs 3072GB/s aggregate)
2. **No EAGLE/MTP speculation** — Moonshot never released MTP heads for K2.6. Only has ngram speculation (~10-30% boost vs EAGLE's ~2.8× speedup)
3. **vLLM vs SGLang** — SGLang with CUDA graphs + EAGLE is fundamentally faster for MoE decode

### Key Lessons

1. **Chunked prefill size is THE knob for single-user TTFT**. 8192→4096 on V4-Pro cut TTFT from ~60s to 7s on 80K token input. SGLang pipeline parallelism case study confirms 4K optimal for DeepSeek-V3/V4 architecture.

2. **DP+EP hurts single-user latency**. The optimization distributes work for concurrent users but routes each individual request to fewer GPUs. Only beneficial at high concurrency (8+ simultaneous requests).

3. **Container SGLang 0.5.10rc0 doesn't support full DP+EP pipeline**. DeepEP assertion fails during CUDA graph capture. SGLang pip (0.5.11+) supports it, but it's counterproductive for our use case anyway.

4. **The user's 60s latency had two causes**: (a) chunked prefill 8192 causing head-of-line blocking on long context, and (b) CUDA graph compilation on first requests after model load (one-time warmup cost).

5. **EAGLE acceptance rate is the decode throughput multiplier**. GLM's 188 tok/s comes from SpecV2 EAGLE (accept_len ~2.85-3.08). Without EAGLE (Kimi), decode speed is fundamentally limited to 1 token per forward pass.

### Final Configs Applied (Production)

**V4-Pro** (`serve-v4pro-container.sh`): Only change: `--chunked-prefill-size 4096` (was 8192). Everything else unchanged.

**GLM-5.1** (`serve-glm51.sh`): No changes this session (DP+EP tested and reverted). Still benefits from `SGLANG_ENABLE_SPEC_V2=1` from 2026-05-20.

**Kimi K2.6** (`serve-kimi-k2.sh`): Added `--enable-chunked-prefill --max-num-batched-tokens 4096`.

**V4-Flash** (`serve-v4flash-container.sh`): DP+EP removed (same container, same DeepEP incompatibility). Already had chunked prefill 4096.

---

## Optimized Configs Applied + V4-Pro OOM Fix + Proxy Reasoning Fix (2026-05-20/21)

**Status**: All optimizations deployed and verified. V4-Pro, GLM-5.1, Kimi K2.6 running with new configs. Proxy reasoning_effort bug fixed.

### What Was Applied

| Model | Changes | Result |
|-------|---------|--------|
| **V4-Pro** | chunked-prefill 8192, swa-full-tokens 0.1, mxfp4, `mem-fraction 0.82` | SERVING on fat007 ✓ |
| **GLM-5.1** | `SGLANG_ENABLE_SPEC_V2=1` | SERVING on fat003, accept_len 3.08 ✓ |
| **Kimi K2.6** | ngram speculation (tokens=4, lookup_max=5) | SERVING on fat008 ✓ |
| **V4-Flash** | Config updated (0.92, swa, reasoning-parser) | NOT restarted (shares port 8100 with V4-Pro) |

### V4-Pro OOM Discovery

`--mem-fraction-static 0.92` caused immediate OOM on first request:
- V4-Pro weighs 805GB → ~178GB/183GB per GPU for weights alone
- 0.92 × 183GB = 168GB reserved → only ~5GB free for KV cache
- flash_mla needed 2.02 GiB for a single attention op → `torch.OutOfMemoryError`
- **Fix**: reduced to `0.82` (leaves ~33GB per GPU for KV cache)
- **Lesson**: mem-fraction must account for model size. V4-Flash (147GB / 4 GPUs = 37GB/GPU) can handle 0.92; V4-Pro (805GB / 8 GPUs = 100GB/GPU) cannot.

### `--speculative-adaptive` Not Available

Container SGLang 0.5.10rc0 doesn't have this flag (it's in newer pip builds). Caused immediate crash:
```
launch_server.py: error: unrecognized arguments: --speculative-adaptive
```
Removed from both V4-Pro and V4-Flash configs.

### Proxy `reasoning_effort` Bug + Fix

**Bug**: Proxy sent `reasoning_effort="none"` to all models. Container SGLang rejects "none" (only accepts low/medium/high/max) → V4-Pro returned 400 Bad Request.

**Root cause**: V4-Flash's OLD config had no `--reasoning-parser` → SGLang didn't validate the field → "none" passed silently. V4-Pro deployed fresh WITH `--reasoning-parser deepseek-v4` → validation kicked in → rejected "none".

**Fix** (claude-code-proxy/server.py):
```python
if thinking_on:
    extra_body["reasoning_effort"] = "high"
elif "deepseek" not in BIG_MODEL.lower():
    extra_body["reasoning_effort"] = "none"
# DeepSeek: omit field entirely (doesn't think by default)
```

**Thinking control per model**:
| Model | Thinking OFF | Thinking ON | Controllable? |
|-------|-------------|-------------|---------------|
| V4-Pro/Flash | Omit field (doesn't think by default) | "high" (but reasons inline, no hidden `<think>`) | Partial — no hidden thinking mode |
| GLM-5.1 | "none" → suppressed | "high" → thinking block visible | **Full control** ✓ |
| Kimi K2.6 | "none" ignored by vLLM | "high" ignored by vLLM | **No control** — always thinks |

**Key insight**: Effort level granularity (low/medium/high/max) doesn't help for open models. GLM's thinking is binary — model self-regulates depth based on problem complexity. The binary ON/OFF mapping is the correct abstraction.

---

## Serving Optimization Research + 3-Model Benchmark (2026-05-20)

**Status**: Research complete, configs applied (see above).

### 3-Model Benchmark vs Sonnet 4.6 (10 tasks, Opus judge)

| Model | 1st Place | Strengths | Measured tok/s |
|-------|-----------|-----------|----------------|
| **Sonnet 4.6** | 4/10 (40%) | Async patterns, concurrency, debugging | ~69 |
| **V4-Pro** | 3/10 (30%) | Architecture, ML, distributed systems | ~73 |
| **GLM-5.1** | 3/10 (30%) | Refactoring, SQL, system design | ~120 |
| **Kimi K2.6** | 0/10 (0%) | (Empty responses — proxy restart mid-run) | **153** |

Kimi's 153 tok/s with ZERO speculation is remarkable — 32B active + MLA + NVFP4 = extremely efficient decode.

### Optimization Research (from SGLang docs.sglang.io + vLLM docs)

**Why**: Current configs use conservative defaults from initial deployment. Official docs (updated May 2026) recommend B200-specific optimizations we're not using.

**V4-Pro/Flash (SGLang container)**:
| Change | From | To | Why |
|--------|------|-----|-----|
| `--mem-fraction-static` | 0.85 | **0.92** | B200 Pro official rec; more KV cache = more concurrent sessions |
| `--chunked-prefill-size` | 4096 | **8192** (Pro only) | Pro-specific; faster prefill on long prompts |
| `--moe-runner-backend` | (not set) | **flashinfer_mxfp4** | Blackwell FP4 MoE kernel; direct compute speedup |
| `--swa-full-tokens-ratio` | (not set) | **0.1** | Blackwell sliding window attention optimization |
| `--disable-flashinfer-autotune` | (not set) | **yes** | Blackwell stability (autotuner can cause stalls) |
| `--speculative-adaptive` | (not set) | **yes** | Auto-adjusts EAGLE steps [1,3,7] based on acceptance EMA |

**GLM-5.1 (SGLang pip)**:
| Change | From | To | Why |
|--------|------|-----|-----|
| `SGLANG_ENABLE_SPEC_V2` | (not set) | **1** | Required env var for EAGLE on GLM architecture |

**Kimi K2.6 (vLLM pip)**:
| Change | From | To | Why |
|--------|------|-----|-----|
| `--speculative-config` | (none) | **ngram, tokens=4, lookup_max=5** | Could push 153→180-200 tok/s on code (high n-gram hit rate) |

**Expected impact**: V4-Pro from ~73→90+ tok/s, V4-Flash from 190→220+ tok/s, Kimi from 153→180+ tok/s.

**Risk**: `mem-fraction-static 0.92` may OOM under max concurrency — docs say "increment by 0.01 until OOM, back off." Will test incrementally.

### Repo Cleanup (same session)

- HPC: removed 312 stale logs, 7.6GB vllm-openai.sif container, stale glm51-container.sh
- Local: added .claude/.playwright-mcp/ to .gitignore, removed stale pull-vllm-container.sh

---

## V4-Pro Live, 1M Context, Compaction Fix (2026-05-20)

**Status**: V4-Pro serving on fat007 (8× B200), 1M context, EAGLE speculation. Claude Code compaction fixed — 1M models now use full context window.

### V4-Pro Serving

Downloaded (806GB, HF token auth) and now **live** on fat-node-007:
- Job 1938662, port 8100, 8× B200 TP=8, EAGLE 3-step MTP
- Context: **1,048,576 tokens** (was incorrectly set to 65K, fixed to 1M — same as V4-Flash)
- Benchmarks: LiveCodeBench 93.5%, SWE-bench Verified 80.6%, Codeforces 3206

Running simultaneously with GLM-5.1 (fat003) and Kimi K2.6 (fat002). 3 fat nodes occupied, 3 idle, 2 draining.

### Claude Code 1M Compaction Fix

**Problem**: All settings files used `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` (200K context). Claude Code derives its auto-compaction threshold from this model name → compacted at ~170K tokens, wasting 83% of V4-Pro/Flash's 1M context.

**Discovery** (from https://platform.claude.com/docs/en/docs/about-claude/models):
- `claude-sonnet-4-20250514` → 200K context (deprecated, retiring June 15, 2026)
- `claude-sonnet-4-6` → **1M context** (current)

**Fix applied:**
- `dsp-settings.json` (V4-Pro): `ANTHROPIC_MODEL` → `claude-sonnet-4-6` (1M)
- `ds-settings.json` (V4-Flash): `ANTHROPIC_MODEL` → `claude-sonnet-4-6` (1M)
- `glm-settings.json` (GLM-5.1): kept `claude-sonnet-4-20250514` (200K matches 202K backend)
- `kimi-settings.json` (Kimi K2.6): kept `claude-sonnet-4-20250514` (200K matches 196K backend)
- Also updated `ANTHROPIC_SMALL_FAST_MODEL` from deprecated `claude-haiku-3-5-20241022` to `claude-haiku-4-5-20251001`

**Result**: V4-Pro/Flash sessions now compact at ~850K+ instead of 170K — **5× more usable context**.

### Proxy Context Limits Updated

| Model | Server Context | `BACKEND_CONTEXT_LIMIT` | `ANTHROPIC_MODEL` |
|-------|---------------|------------------------|-------------------|
| V4-Pro | 1,048,576 | 1,000,000 | `claude-sonnet-4-6` (1M) |
| V4-Flash | 1,048,576 | 1,000,000 | `claude-sonnet-4-6` (1M) |
| GLM-5.1 | 202,752 | 202,752 | `claude-sonnet-4-20250514` (200K) |
| Kimi K2.6 | 196,608 | 196,608 | `claude-sonnet-4-20250514` (200K) |

### Model Recommendation Analysis

Compared all 4 coding models for agent swarm use. Recommendation: **V4-Pro + V4-Flash** (retire GLM-5.1 + Kimi K2.6 when ready):

| Metric | V4-Pro | V4-Flash | GLM-5.1 | Kimi K2.6 |
|--------|--------|----------|---------|-----------|
| SWE-bench Verified | **80.6%** | 48-52% | ~54% | ~52% |
| LiveCodeBench | **93.5%** | 65-70% | ~70% | ~68% |
| Codeforces | **3206** | ~1800 | ~1800 | ~1700 |
| tok/s (single) | ~50-80 | **190** | 167 | ~50-70 |
| tok/s (8 conc.) | ~200-300 | **782** | ~500 | ~200 |
| EAGLE | Yes | Yes | Yes | **No** |
| Context | **1M** | **1M** | 202K | 256K |
| GPUs | 8× B200 | 4× B200 | 8× B200 | 4× B200 |

V4-Pro for hard problems (SWE-bench 80.6% beats even Opus 4.6's 72.5%), V4-Flash for speed/concurrency. Both share port 8100, same container, same EAGLE — zero-friction switching.

GLM-5.1 drawback: Chinese thinking/tool-call output ("让我阅读我需要作为模板的关键文件" leaking into visible responses). Kimi drawback: no EAGLE speculation.

### V4-Pro Infrastructure (complete)

- `serving/download-deepseek-v4-pro.sh` — cpu partition, scratch output, HF_TOKEN auth
- `serving/serve-v4pro-container.sh` — 8× B200, TP=8, same container as V4-Flash, port 8100, EAGLE, 1M context
- `proxy-ai dsp` — tunnel:8100 + proxy:5009, BIG_MODEL=deepseek-v4-pro, ctx=1,000,000
- `claude-dsp.cmd` → `~/.claude/dsp-settings.json` (port 5009, `claude-sonnet-4-6`)
- `switch-model.sh v4pro` integrated

**Key design decisions:**
- V4-Pro and V4-Flash share port 8100 (only one at a time, switch via `switch-model.sh`)
- Separate proxy ports: ds=5005 (flash), dsp=5009 (pro) — different BIG_MODEL env var
- Model stored on scratch (805GB exceeds 1TB project quota shared with other models)

### Factory Droid Fix (Root Cause: 3-Layer Config Desync)

**Error**: `BYOK Error: 404 {"detail":"Not Found"}`

**Root cause** (THREE separate issues):
1. **Phantom `anthropic` provider entry** in settings.json — `provider: "anthropic"` makes Factory hit `/v1/messages` which doesn't exist on our OpenAI-format proxy → 404
2. **Index 5 collision** — Three entries shared index 5 (GPT Codex, Opus anthropic, Opus Bedrock)
3. **Mission-level model-settings.json** still referenced old `custom:Claude-Opus-4.6-Bedrock-0` which no longer existed after our earlier settings.json fix

**Fix:**
- Removed the `anthropic` provider Opus entry entirely (broken — proxy only serves `/v1/chat/completions`)
- Moved Opus Bedrock to index 6 (unique, no collision)
- Updated BOTH mission `model-settings.json`: `validationWorkerModel` → `custom:Claude-Opus-4.6-Bedrock-6`
- Updated BOTH mission `runtime-custom-models.json`: added/fixed Opus entry at index 6
- Updated global `missionModelSettings` and `missionOrchestratorModel` to match

**Lesson**: Factory has 3 config layers (config.json → settings.json → missions/*/). Mission-level OVERRIDES global. Must fix all three.

---

## V4-Pro Production Ready & ASR/TTS Research (2026-05-20)

**Status**: V4-Pro serve script hardened, client tooling confirmed compatible, ASR/TTS landscape mapped.

### DeepSeek-V4-Pro Deployment Readiness

All infrastructure verified and production-ready:

| Component | Status | Notes |
|-----------|--------|-------|
| `serving/download-deepseek-v4-pro.sh` | Ready | ~805GB download to `$LLM_DIR/models/deepseek-v4-pro` |
| `serving/serve-v4pro-container.sh` | **Fixed** | QoS 1d→3d, added `--mem-fraction-static 0.85` |
| `serving/switch-model.sh v4pro` | Ready | Cancels coding jobs, starts V4-Pro on 8× B200 |
| Client tooling (`claude-ds`) | Compatible | Same port 8100, same container, same proxy path |
| Container SIF | Shared | Same `sglang-dsv4-blackwell.sif` as V4-Flash |

**Fixes applied to `serve-v4pro-container.sh`**:
- `--qos=1d` / `--time=10:00:00` → `--qos=3d` / `--time=3-00:00:00` (8× B200 node = expensive, 3 days matches V4-Flash)
- Added `--mem-fraction-static 0.85` — EAGLE speculation needs ~2GB temp tensors per GPU; without this flag, OOM on large batches (learned from V4-Flash deployment)

**To deploy** (when ready for heavy coding tasks):
```bash
ssh user@hpc-cluster.example.com "sbatch /shared/project/<account>/llm/serving/download-deepseek-v4-pro.sh"
# Wait ~30-60 min for 805GB download, then:
./serving/switch-model.sh v4pro
```

### ASR/TTS/STT Model Landscape (May 2026)

Researched best open-source speech models for potential HPC self-hosting:

#### ASR (Speech-to-Text)

| Model | Params | Context | Languages | VRAM | Key Feature | License |
|-------|--------|---------|-----------|------|-------------|---------|
| **VibeVoice-ASR** | 9B | **60 min audio** | 50+ | ~18GB (1× L40S) | Diarization, timestamps, inverse text norm | Apache 2.0 |
| Whisper-large-v3-turbo | 0.8B | 30s chunks | 100+ | ~3GB | Battle-tested, fastest Whisper | MIT |
| **Qwen3-ASR-1.7B** | 1.7B | — | Multi | ~4GB | **Already on HPC** | Apache 2.0 |

**Recommendation**: VibeVoice-ASR is the clear winner — 60-minute context means no chunking for long meetings/recordings. Handles diarization (who spoke when) natively. Fits on a single L40S.

#### TTS (Text-to-Speech)

| Model | Params | Latency | Languages | VRAM | Key Feature | License |
|-------|--------|---------|-----------|------|-------------|---------|
| **Kokoro-82M** | 82M | ~50ms | EN, JA, ZH, FR, KO | <1GB | 54 voices, 10M+ downloads, tiny | Apache 2.0 |
| **CosyVoice3-0.5B** | 0.5B | **150ms** | 9 (EN/ZH/JA/KO/FR/DE/...) | ~2GB | Voice cloning, streaming | Apache 2.0 |
| VibeVoice-Realtime-0.5B | ~0.5B | 300ms | Multi | ~2GB | Real-time streaming, emotion | MIT |
| **Qwen3-TTS-1.7B** | 1.7B | — | Multi | ~4GB | **Already on HPC** | Apache 2.0 |

**Recommendation**: 
- **For quality + cloning**: CosyVoice3-0.5B (150ms latency, 9 languages including German, voice cloning in 5s of reference audio)
- **For minimal footprint**: Kokoro-82M (82M params = runs on CPU, 54 built-in voices, sub-50ms)
- **Already available**: Qwen3-TTS-1.7B at `/shared/project/<account>/llm/models/qwen3-tts-1.7b/`

#### Deployment Plan (if proceeding)

| Model | Hardware | Port | Use Case |
|-------|----------|------|----------|
| VibeVoice-ASR (9B) | 1× L40S | 8300 | Long-form transcription, meetings |
| CosyVoice3-0.5B | 1× L40S (shared) | 8301 | Real-time TTS with voice cloning |
| Kokoro-82M | CPU only | 8302 | Lightweight TTS, zero GPU cost |

Total additional cost: 1-2 L40S GPUs (174 available, 33 nodes idle). All models fit without touching B200 allocation.

---

## Proxy Hardening, Factory Fix & Model Landscape Update (2026-05-20)

**Status**: All proxy fixes live, Factory droid connected to Bedrock proxy, model research complete.

### Proxy Fixes (claude-code-proxy/server.py)

Three issues fixed in this session:

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| **422 from /pdf skill** | Anthropic document blocks (base64 PDF) forwarded raw to OpenAI-compat backend | Replace with text note: "[Document content not supported, use Read tool]" |
| **GLM/Kimi responding in Chinese** | No language instruction in system prompt for CJK-default models | Inject "CRITICAL LANGUAGE REQUIREMENT: English only" into system message for all OpenAI-routed models |
| **Content array as string error** | `tool_result.content` can be array of blocks, backend expects string | Flatten all content block arrays to text before forwarding |

**Code changes** (server.py):
- Added `ContentBlockDocument` and `ContentBlockUnknown` Pydantic types for robust parsing
- Document blocks → replaced with "[Document content ({media_type}) not supported by this model]"
- English instruction appended to system message for all non-Anthropic backends
- All content block arrays (user, assistant, tool_result) properly flattened

### Factory (Droid) Settings Fix

**Problem**: Factory BYOK error "400 status code (no body)" when using Bedrock proxy on localhost:9191.

**Root cause**: Factory auto-generates custom model IDs using pattern `custom:{DisplayName}-{index}`. The Bedrock entries had IDs ending in `-0` instead of their actual array index. Session default referenced `custom:Claude-Sonnet-4.6-Bedrock-4` (correct for index 4), but the entry's ID was `custom:Claude-Sonnet-4.6-Bedrock-0` — no match → broken request.

**Fix** (`.factory/settings.json`):
```
custom:Claude-Sonnet-4-Bedrock-0  → custom:Claude-Sonnet-4-Bedrock-2   (index 2)
custom:Claude-Sonnet-4.5-Bedrock-0 → custom:Claude-Sonnet-4.5-Bedrock-3 (index 3)
custom:Claude-Sonnet-4.6-Bedrock-0 → custom:Claude-Sonnet-4.6-Bedrock-4 (index 4)
custom:Claude-Opus-4.6-Bedrock-0  → custom:Claude-Opus-4.6-Bedrock-5   (index 5)
```

Mission orchestrator/worker models updated to `custom:Claude-Opus-4.6-Bedrock-5`.

### Open Source Model Landscape (May 2026)

**Arena Leaderboard (open-weight models, current rankings)**:

| Rank | Model | Organization | Active Params | Notes |
|------|-------|--------------|---------------|-------|
| #5 | GLM-5.1 | ZAI | 40B | Already deployed (8× B200, 167 tok/s) |
| #7 | Kimi K2.6 | Moonshot | 32B | Already deployed (4× B200, vLLM) |
| #20 | mimo-v2.5-pro | Xiaomi | ? | New entrant — unknown specs |
| #26 | Qwen3.5-397B-A17B | Alibaba | 17B | New MoE — smaller active than our Qwen3-235B |
| #50 | Qwen3-235B-A22B-2507 | Alibaba | 22B | Already have (backup) |
| ~Top 10 | DeepSeek-V4-Pro | DeepSeek | 49B | **Ready to deploy** (listed in CLAUDE.md) |

**DeepSeek-V4-Pro (confirmed specs from HuggingFace)**:

| Spec | Value |
|------|-------|
| Total params | **1.6T** (MoE) |
| Active per token | **49B** |
| Context | **1,000,000** (1M) |
| Precision | FP4 experts + FP8 attention (native, not post-quant) |
| Attention | CSA + HCA hybrid (27% FLOPs of V3.2, 10% KV cache) |
| LiveCodeBench | **93.5%** (best open-source) |
| Codeforces | **3206** (beating Claude) |
| SWE-bench Verified | 80.6% |
| License | MIT |
| GPUs needed | 8× B200 (full fat node) |
| Download size | ~862GB |

**Key finding**: V4-Pro is now confirmed as the strongest open-source coding model (LiveCodeBench 93.5%, Codeforces 3206). Already in our deployment plan (`serving/download-deepseek-v4-pro.sh` exists). Main trade-off vs V4-Flash: 3.8× slower per token (49B vs 13B active) but dramatically better on hard tasks.

**New model worth watching**: Qwen3.5-397B-A17B (Arena #26, 17B active) — smaller than our existing Qwen3-235B (22B active) but newer architecture. Lower priority since we already cover that tier with V4-Flash (13B) and GLM-5.1 (40B).

### Recommendations: What to Add to Our Stack

| Priority | Model | Why | Action |
|----------|-------|-----|--------|
| **HIGH** | DeepSeek-V4-Pro | LiveCodeBench 93.5%, already have download script | `sbatch serving/download-deepseek-v4-pro.sh` → 8× B200 |
| LOW | Qwen3.5-397B-A17B | Arena #26 but 17B active overlaps with V4-Flash tier | Monitor, don't deploy yet |
| LOW | mimo-v2.5-pro | Arena #20, unknown specs | Wait for HuggingFace release |

**Current fleet is well-positioned**: V4-Flash (speed), GLM-5.1 (deep reasoning + tools), Kimi K2.6 (cost-efficient). V4-Pro is the only clear upgrade — use for hard coding tasks where 93.5% LiveCodeBench matters.

---

## Context Overflow Protection & Kimi K2.6 Live (2026-05-19)

**Status**: All proxy models protected against 400 context overflow errors. Kimi K2.6 actively serving.

### The Problem

Claude Code assumes it's talking to a 200K-context Claude model and auto-compresses at ~170K tokens. When proxying to HPC models, this creates a mismatch — the proxy sends more tokens than the backend allows, causing hard 400 errors. Three separate issues discovered:

1. **max_completion_tokens vs max_tokens**: LiteLLM/OpenAI uses `max_completion_tokens`, but the proxy cap only checked `max_tokens`. A request with `max_completion_tokens: 32000` bypassed the cap entirely, pushing GLM-5.1 past its 202K limit.
2. **No input token awareness**: The proxy had no concept of backend context limits. It forwarded whatever Claude Code sent, even if the input alone exceeded the model's capacity.
3. **Kimi served at 65K**: Initial serve script used `--max-model-len 65536` (conservative), but Kimi K2.6 with MLA attention can handle much more on 4× B200.

### Three-Layer Fix

| Layer | What | Where |
|-------|------|-------|
| **1. Cap both keys** | Check `max_tokens` AND `max_completion_tokens` | `server.py` line ~1228 |
| **2. BACKEND_CONTEXT_LIMIT** | New env var per model; proxy estimates input tokens (chars/4) and trims oldest non-system messages if exceeding limit | `server.py` + `proxy-ai.cmd` |
| **3. Increased server context** | Kimi: 65K → 196K, V4-Flash: already 1M | `serve-kimi-k2.sh` |

### Context Limits per Model

| Model | Server --max-model-len | Proxy BACKEND_CONTEXT_LIMIT |
|-------|----------------------|---------------------------|
| V4-Flash | 1,048,576 (1M) | 1,000,000 |
| GLM-5.1 | 202,752 | 202,752 |
| Kimi K2.6 | 196,608 | 196,608 |
| Gemma-4 | 8,192 | 8,192 |

### proxy-ai Improvements

- **Per-model stop**: `proxy-ai stop glm` / `proxy-ai stop kimi` / `proxy-ai stop ds` kills only that model's tunnel + proxy (previously only `proxy-ai stop` existed, which killed everything)
- **Kimi fully integrated**: `proxy-ai kimi` sets up tunnel:8104 + proxy:5008 with `BACKEND_CONTEXT_LIMIT=196608`
- **V4-Flash context unlocked**: `BACKEND_CONTEXT_LIMIT` bumped from 65K to 1M to match server's actual capacity

### Kimi K2.6 Now Serving

Job 1932237 loaded successfully (~20 min). Currently at 131K context (submitted before 196K script update — next resubmit will use 196K). Full client stack working: `claude-kimi`, `proxy-ai kimi`, Zed IDE.

### Why This Matters

Claude Code's auto-compaction is hardcoded per model name — there's no env var or setting to override the 170K trigger point. When the backend has less context (Gemma at 8K) or when completion tokens push past the limit, the only defense is proxy-side trimming. The `BACKEND_CONTEXT_LIMIT` safety net drops old messages rather than summarizing (not as good as true compaction), but it prevents hard crashes.

For V4-Flash (1M server), Claude Code's 170K compaction triggers long before any server limit — no trimming needed.

---

## Kimi K2.6 NVFP4 Deployment — Ready to Serve (2026-05-18)

**Status**: Infrastructure complete, not actively serving (user preference — GLM-5.1 is primary alt-coding model)

### What Was Done

Deployed full Kimi K2.6 pipeline: download → vLLM pip venv → serve script → client tooling (claude-kimi, proxy-ai, Zed).

| Component | Status |
|-----------|--------|
| Model weights (555GB, 60 shards) | Downloaded to `/shared/scratch/user/models/kimi-k2.6-nvfp4/` |
| vLLM venv (0.21.0, Python 3.11) | Created at `/shared/project/<account>/llm/vllm-venv/` |
| Serve script (`serve-kimi-k2.sh`) | 4× B200, TP=4, port 8104, tool calling + reasoning |
| `claude-kimi.cmd` + `kimi-settings.json` | Proxy port 5008, same pattern as claude-ds/glm/gm |
| `proxy-ai kimi` | Tunnel:8104 + proxy:5008 |
| Zed `settings.json` | "HPC Kimi" provider at `localhost:8104/v1` |
| `switch-model.sh kimi` | Integrated |

### Key Technical Decisions

1. **vLLM pip over container**: Container approach (`vllm/vllm-openai:latest`) pulled successfully (7.6GB SIF) but crashed at runtime — `--writable-tmpfs` overlay ran out of memory when flashinfer tried to create cubin symlinks. Switched to pip-based vLLM in a separate `vllm-venv` to avoid conflicts with SGLang `venv`.

2. **No native MTP**: `num_nextn_predict_layers: 0` in ALL Kimi K2 variants (base, instruct, NVFP4). Moonshot never released MTP heads — not stripped by NVIDIA's quantization. Only speculation option is ngram (removed from initial deploy for stability).

3. **`--enable-auto-tool-choice` required**: vLLM rejects `tool_choice: "auto"` (sent by Claude Code) unless this flag is present alongside `--tool-call-parser`.

4. **Separate venv**: vLLM 0.21.0 conflicts with SGLang (different torch/flashinfer versions). Isolated in `vllm-venv` with its own `LD_LIBRARY_PATH` for `libnvrtc.so` on SM100/B200.

### Performance Expectation

~50-70 tok/s estimated (no MTP, 32B active, 4× B200 TP=4). GLM-5.1 is faster (167 tok/s with EAGLE) and higher Arena ELO (#5 vs #7). Kimi's advantage: half the GPU cost (4 vs 8 B200s), 256K context, vision support.

### To Launch

```bash
sbatch serving/serve-kimi-k2.sh   # ~20 min to load
# Then locally:
proxy-ai kimi                      # or: claude-kimi
```

---

## GLM-5.1 State File Fix (2026-05-18)

**Problem**: `proxy-ai glm` failed with "No node found" because `.serve-state-glm.json` didn't exist on HPC.

**Root cause**: The currently running GLM job (1929548) was submitted from an older version of `serve-glm51.sh` that predated the state file write. The serve script now creates `.serve-state-glm.json` automatically, but this job never did.

**Fix**: Manually created the state file pointing to `fat-node-003` (the node running the job). Future `sbatch` submissions will create it automatically via the current script.

---

## GLM-5.1 Benchmark, Proxy Fixes & Kimi K2.6 Research (2026-05-18)

**Status**: Fixes deployed — proxy caps max_tokens + auto-doubles for thinking mode

### Three-Way Comparison (8 tasks, Opus 4.6 judge)

| Model | 1st Place | Avg Score /40 | Avg tok/s | Avg Latency |
|-------|-----------|---------------|-----------|-------------|
| GLM-5.1 (thinking) | 1/7 (14%) | 32.9 | 105 | 36.3s |
| **GLM-5.1 (no-think)** | **3/7 (43%)** | **34.0** | 88 | **17.5s** |
| **Sonnet 4.6** | **3/7 (43%)** | 33.0 | 65 | 27.0s |

For comparison, **V4-Flash** (measured 2026-05-11): **207 tok/s**, 7.8s latency, **62% win rate** vs Sonnet.

**Findings**:
- GLM no-think ≈ Sonnet 4.6 quality, but **1.4× faster** (88 vs 65 tok/s)
- Thinking mode helps only on system design/architecture tasks (1/7 wins)
- Thinking hurts on coding (latency doubles, output truncation risk)
- V4-Flash remains speed king: **2.4× faster than GLM**, **3× faster than Sonnet**

### Critical Discovery: SGLang EAGLE max_tokens Scheduling Penalty

Benchmarking revealed a **10-26× slowdown** with non-standard max_tokens values:

| max_tokens | Speed | Relative |
|-----------|-------|----------|
| 2048 | **57 tok/s** | 1× (optimal) |
| 512 | 2.2 tok/s | **26× slower** |
| 8192 | 3.4 tok/s | **17× slower** |

**Root cause**: SGLang pre-allocates KV cache blocks sized to max_tokens. When this doesn't align with EAGLE's speculative draft buffer allocation, the scheduler thrashes.

### Thinking Budget Truncation Bug (Fixed)

SGLang counts reasoning + content tokens together in `completion_tokens`:
```
max_tokens=2048 + thinking: ~1200 reasoning + 848 content → TRUNCATED → 24/40 score
max_tokens=4096 + thinking: ~1200 reasoning + 1700 content → FULL → 33/40 score
```
+8 points recovered by simply providing enough token budget for thinking.

### Proxy Fix (server.py line 1224-1229)

```python
# Cap max_tokens to SGLang EAGLE-optimal range.
# Thinking mode needs 2× budget: reasoning tokens count toward completion_tokens.
optimal_max = 4096 if thinking_on else 2048
if litellm_request.get("max_tokens", 0) > optimal_max:
    litellm_request["max_tokens"] = optimal_max
```

This fix addresses both issues simultaneously:
1. Prevents max_tokens=8192/16384 from hitting the scheduling penalty
2. Gives thinking mode 4096 tokens (enough for ~1200 reasoning + 2896 content)

### Why GLM-5.1 Uses 8× B200 (and why Kimi K2.6 is better)

GLM-5.1 is stuck at TP=8 due to two constraints:
- **VRAM**: 756GB FP8 → 4× B200 (768GB) leaves only 12GB, not enough for KV cache
- **Heads**: 64 attention heads, TP must divide evenly → TP=4 (768GB too tight), TP=8 (1536GB, 780GB free)

Meanwhile Kimi K2.6 (610GB) on 4× B200 (768GB) has **158GB headroom** — zero compromise.
That's **1T params on 4 GPUs** vs GLM's **754B on 8 GPUs**. Kimi is 1.3× more params at half the GPU cost.

### Kimi K2.6 (Moonshot AI) — Next Model to Deploy

| Spec | Kimi K2.6 | V4-Flash | GLM-5.1 |
|------|-----------|----------|---------|
| Total params | **1T** | 282B | 754B |
| Expert weights | int4 native | FP4 native | FP8 |
| Context | 256K | 1M | 202K |
| Storage | 610GB | 147GB | 756GB |
| GPUs needed | **4× B200** | 4× B200 (container) | 8× B200 |
| Vision | **Yes** | No | No |
| Thinking | **Hybrid** | No | Per-request |
| EAGLE/MTP | Likely yes | Yes (2.8×) | Yes (2×) |
| License | Open-weight | MIT | MIT |

**No performance compromise**: 610GB model in 768GB VRAM = full precision, no quantization needed.
Native int4 experts (trained in int4, not post-quantized) means lossless at this size.

**Deployment plan**: 4× B200, port 8104, SGLang or vLLM, ~30-45 min download to scratch.
If benchmarks confirm SOTA claims → retire GLM-5.1 (frees full fat node).

### Forward Fleet Strategy

| Role | Model | GPUs | Speed | Purpose |
|------|-------|------|-------|---------|
| **Fast coding** | V4-Flash | 4× B200 | 207 tok/s | Agent swarms, 20-30 concurrent |
| **Heavy reasoning** | Kimi K2.6 | 4× B200 | >40 tok/s | Hard problems, vision, thinking |
| **Retire candidate** | GLM-5.1 | 8× B200 | 88 tok/s | Replaced by K2.6 if benchmarks hold |

Total: **8 B200s** (1 fat node) for both models. Down from 12 (V4-Flash + GLM). Leaves 56 B200s free.

---

## GLM-5.1 Tool Calling & Thinking Fix (2026-05-18)

**Status**: Production — tools + per-request thinking control working  
**Problem**: GLM-5.1 generated raw XML tool calls in content field, and `<think>` tags leaked into every response.

### Root Cause & Fix

| Issue | Cause | Fix |
|-------|-------|-----|
| Tool calls as raw XML | `--tool-call-parser glm45` expects newlines between args; GLM-5.1 uses inline format | Switch to `--tool-call-parser glm47` |
| `<think>` tags in content | No reasoning parser → tags pass through as content | Add `--reasoning-parser glm45` (separates into `reasoning_content` field) |
| No thinking control | Thinking always ON by default | `reasoning_effort: "none"` (top-level param) disables per-request |

### Key Insight: Parser Names Are Confusing

- `glm45` tool parser = GLM-4.5/4.6 format (newlines between `<arg_key>` tags)
- `glm47` tool parser = GLM-4.7/5.x format (inline `<arg_key>` tags — what GLM-5.1 generates)
- `glm45` reasoning parser = Shared `<think>...</think>` tag detection (works for all GLM versions)

**No quality impact**: Parsers are post-processors that restructure output text into API fields.

### Thinking Control Architecture

```
Claude Code (/effort slider):
  thinking.enabled=true  → proxy → reasoning_effort:"high" → GLM thinks + answers
  thinking.enabled=false → proxy → reasoning_effort:"none" → GLM answers directly

Zed (direct API, no proxy):
  Default: thinking ON → reasoning_content field (separate from content)
  Content is always clean — no <think> tag leakage either way
```

### Deployment Issues

Job kept getting REQUEUED on fat006 (Prolog error → node drained). Excluded fat001+fat006, eventually started on fat003 after scheduler recovered from drain event.

---

## GLM-5.1 Deployment — 754B MoE at 167 tok/s (2026-05-17)

**Status**: Production on 8× B200, port 8103, EAGLE speculation  
**Motivation**: Deploy a 40B-active coding model alongside V4-Flash. GLM-5.1 offers 3× deeper reasoning per token (40B vs 12B active) while maintaining high throughput via native MTP EAGLE.

### Architecture & Result

| Spec | Value |
|------|-------|
| Model | `zai-org/GLM-5.1-FP8` (MIT license) |
| Architecture | `GlmMoeDsaForCausalLM` — MoE + DSA attention (MLA-like) |
| Total / Active | 754B / 40B (256 routed + 1 shared experts, 8 active per token) |
| FP8 size | 756 GB |
| Hardware | 8× NVIDIA B200 (192GB each, TP=8) |
| Throughput | **167 tok/s** with EAGLE (native MTP, `num_nextn_predict_layers=1`) |
| Context | 202,752 tokens (~200K) |
| Port | 8103 |
| Storage | `/shared/scratch/user/models/glm-5.1-fp8/` (scratch, 756GB) |

### Deployment Journey (6 attempts)

```
Job ID    │ Issue                              │ Fix
──────────┼────────────────────────────────────┼──────────────────────────────────────
1917332   │ module load cuda/12.9.0 fails      │ Add prerequisite: module load hpc-cluster/2509-fat
1917333   │ libnvrtc.so.13: cannot open        │ LD_LIBRARY_PATH=$VENV/.../nvidia/cu13/lib
1917334   │ Container: TokenizersBackend error  │ Abandoned container → use pip venv
1917335   │ TP=5: assert num_heads % tp == 0   │ 64 heads doesn't divide by 5 → try TP=4
1917336   │ TP=4: OOM (189GB > 178GB usable)   │ 756/4=189GB > B200 usable → increase to TP=8
1917337   │ ✅ SUCCESS (167 tok/s)             │ TP=8 (94.5GB/GPU), ~15min DeepGEMM JIT warmup
──────────┴────────────────────────────────────┴──────────────────────────────────────
```

### Key Learnings

1. **TP must divide `num_attention_heads`**: GLM-5.1 has 64 attention heads. TP=5 is mathematically impossible (64 % 5 ≠ 0). Valid TP values: 1, 2, 4, 8, 16, 32, 64.

2. **B200 usable memory is ~178GB, not 192GB**: CUDA driver/ECC overhead takes ~14GB. So 756GB / 4 GPUs = 189GB per GPU > 178GB → OOM. Need TP=8 (94.5GB/GPU, 84GB headroom).

3. **Container incompatible with GLM-5.1**: The `lmsysorg/sglang:deepseek-v4-blackwell` container ships older `transformers` that doesn't know `TokenizersBackend` — GLM-5.1's non-standard tokenizer class. Must use pip venv SGLang instead.

4. **`libnvrtc.so.13` for SM100 (B200)**: SGLang's `sgl_kernel` requires this CUDA runtime compiler library. On pip installs, it lives at `$VENV/lib/python3.11/site-packages/nvidia/cu13/lib/`. Must be on `LD_LIBRARY_PATH`.

5. **DeepGEMM JIT compilation**: First boot takes ~15 minutes compiling 16,384 kernel variants for B200's SM100 architecture. Cached afterward in `$HOME/.cache/deep_gemm/`. Can pre-compile with `python3 -m sglang.compile_deep_gemm`.

6. **Scratch vs Project storage**: GLM-5.1 at 756GB exceeds the 1TB project quota. Used personal scratch (`/shared/scratch/user/`, 15TB, no backups) — appropriate for re-downloadable model weights.

### Also This Session

- **Qwen3-235B deleted** (0% win rate vs Sonnet in our testing, freed 438GB from project storage)
- **Zed IDE** updated with GLM-5.1 at `localhost:8103/v1`
- **`claude-glm`** launcher verified working (tunnel:8103 → proxy:5007 → Claude Code)

---

## Agentic Coding on HPC — Multi-Model Architecture (2026-05-15)

**Status**: Production — three proxy configurations, zero Anthropic API traffic  
**Motivation**: Run Claude Code through our own self-hosted models with complete network isolation from Anthropic's servers. Free, fast, air-gapped.

### Architecture

```
claude-ds  → ds-settings.json  → proxy-ds (5005) → tunnel:8100 → V4-Flash (4× B200)
claude-gm  → gm-settings.json  → proxy-gm (5006) → tunnel:8200 → Gemma-4 MMM (2× L40S)
claude-glm → glm-settings.json → proxy-glm(5007) → tunnel:8103 → GLM-5.1 (8× B200)
```

Each settings file contains three env vars that together block ALL traffic to api.anthropic.com:
- `ANTHROPIC_BASE_URL=http://localhost:500X` — redirects LLM API calls to local proxy
- `CLAUDE_CODE_SKIP_OAUTH=1` — disables OAuth authentication flow
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` — blocks telemetry, MCP registry, org metrics

### Proxy (claude-code-proxy)

Based on [1rgs/claude-code-proxy](https://github.com/1rgs/claude-code-proxy) (LiteLLM-based). Translates:
- Incoming Anthropic Messages API → OpenAI Chat Completions
- Outgoing OpenAI tool_calls → Anthropic tool_use blocks
- Streaming SSE in both directions

### Model Fleet

| Command | Model | Active | Hardware | Port | Throughput | Status |
|---------|-------|--------|----------|------|-----------|--------|
| `claude-ds` | DeepSeek-V4-Flash | 13B | 4× B200 | 8100 | **190 tok/s** | Offline (ready to restart) |
| `claude-gm` | Gemma-4-31B-MMM-SFT | 31B | 2× L40S | 8200 | 23 tok/s | ACTIVE — MMM domain |
| `claude-glm` | GLM-5.1 | 40B | 8× B200 FP8 | 8103 | **167 tok/s** | **ACTIVE** — Alt coding |
| — | DeepSeek-V4-Pro | 49B | 8× B200 | 8100 | ~50-80 (est.) | Not yet downloaded |

**IDE Integration**: Zed connects directly to all models via `localhost:PORT/v1` (OpenAI-compat, no proxy needed). Requires `proxy-ds` running for SSH tunnel.

### V4-Pro: The Bigger Brother (Ready to Deploy)

| Spec | V4-Flash | V4-Pro |
|------|----------|--------|
| Total params | 284B | **1.6T** |
| Active params | 13B | **49B** |
| Experts | 256 routed | 384 routed + 1 shared |
| Context | 1M | 1M |
| KV | MLA (1 head) | CSA/HCA (evolved MLA, 1 head) |
| Storage | 149GB | ~805GB |
| GPUs | 4× B200 | **8× B200 (full fat node)** |
| LiveCodeBench | 88.8% | **93.5%** |
| Codeforces | — | **3206** |
| SWE-bench | 80.8% | 80.6% |

**Why ~50-80 tok/s (not 190)?** Pure math: 49B active / 13B active = 3.8× more compute per token. Memory bandwidth is the bottleneck in autoregressive generation — 49B × 2 bytes = 98GB must stream from VRAM per token vs 26GB for Flash. EAGLE helps both equally (~2.8× multiplier).

Deploy: `sbatch serving/download-deepseek-v4-pro.sh` (805GB, ~30-60min), then `./serving/switch-model.sh v4pro`.

### Tool Calling Fix: `--tool-call-parser deepseekv4`

V4-Flash uses custom "DSML" format for tool calls. Without a parser, SGLang returns raw text. `--tool-call-parser deepseekv4` converts DSML → standard OpenAI `tool_calls` → proxy translates to Anthropic `tool_use`.

### Critical Proxy Fix: Tool Calls Output as Text (2026-05-16)

**Symptom**: Claude Code's agent would "describe" tool calls in text instead of executing them. Output looked like: `[Tool: Agent (ID: call_xxx)] Input: {...}` — proper formatting but zero execution.

**Root Cause**: The proxy's OpenAI message conversion destroyed tool call structure in conversation history:

```python
# THE BUG — server.py line 1267 (old code)
elif block.get("type") == "tool_use":
    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
```

When Claude Code sent conversation history with previous `tool_use` blocks, the proxy flattened them to text before sending to V4-Flash. V4-Flash then **learned from its own history** that "this is how tools are called in this conversation" and mimicked the text pattern instead of emitting structured DSML tokens.

**Why it's insidious**: 
- SGLang's `--tool-call-parser deepseekv4` works perfectly (tested directly)
- The proxy's streaming tool_call handler works perfectly (no `is_claude_model` gate there)
- First few turns work fine (no tool history yet)
- Fails only after 2+ tool-calling turns accumulate in history
- The exact output format matches what V4-Flash sees in its own conversation context

**The Fix** (3 changes to `claude-code-proxy/server.py`):

1. **Proper OpenAI `tool_calls` in message history** — Assistant messages with `tool_use` blocks now convert to `{"role": "assistant", "tool_calls": [{"type": "function", ...}]}` instead of flattened text.

2. **Proper `role: "tool"` messages** — User messages with `tool_result` blocks now become `{"role": "tool", "tool_call_id": "...", "content": "..."}` messages instead of text.

3. **Removed `is_claude_model` gate** — Non-streaming responses now return proper `tool_use` blocks regardless of backend model (was silently dropping tools for non-Claude models).

**Lesson for anyone proxying Claude Code to non-Anthropic models**: The model's conversation history IS its few-shot context. If you corrupt tool_use/tool_result structure into text, the model will output tools as text. This isn't documented anywhere in claude-code-proxy or LiteLLM — it's a protocol-level behavioral failure that only manifests after multi-turn tool usage.

### Alternative Client: Crush (Charmbracelet)

[Crush](https://github.com/charmbracelet/crush) — terminal-based coding agent by Charmbracelet. Connects directly to V4-Flash at `localhost:8100/v1` (no proxy needed, native OpenAI-compat).

**Key config** (`~/.config/crush/crush.json`):
- `"disable_default_providers": true` — blocks built-in Bedrock provider (mandatory for air-gap)
- Model roles `coder`/`summarizer`/`default` — all mapped to V4-Flash
- `"base_url": "http://localhost:8100/v1"` — direct to SGLang

Wrapper: `crush-ds.cmd` clears all AWS env vars before launch (belt + suspenders).

### Zed Editor Direct Integration (2026-05-17)

Zed connects directly to HPC models via OpenAI-compatible endpoint — no proxy needed for the editor (unlike Claude Code which needs Anthropic→OpenAI translation).

**Setup**: Add `openai_compatible` providers in Zed `settings.json`:
```json
"HPC DeepSeek": { "api_url": "http://localhost:8100/v1", "available_models": [...] }
"HPC GLM":      { "api_url": "http://localhost:8103/v1", "available_models": [...] }
"HPC Gemma":    { "api_url": "http://localhost:8200/v1", "available_models": [...] }
```

**Key discovery**: Zed stores API keys in its own credential store, NOT in `settings.json`. The `api_key` field in config is ignored. You must enter a dummy key via Zed's Configure UI (or set `HPC_DEEP_SEEK_API_KEY` env var). SGLang doesn't validate keys, so any value works.

**Prerequisite**: `proxy-ds` must be running — it manages the SSH tunnel that makes `localhost:8100` reachable.

### V4-Flash Stability Fix: OOM with EAGLE at 1M Context (2026-05-17)

**Problem**: V4-Flash crashed with `torch.OutOfMemoryError` after serving the first request. EAGLE speculation's `forward_c4_indexer` needed 2GB for temporary attention tensors, but only 2GB was free per GPU.

**Root cause**: SGLang's auto memory allocation gave ~98% of free GPU memory to the KV cache pool, leaving almost nothing for EAGLE's temporary computation buffers.

**Fix**: `--mem-fraction-static 0.85` — reserves 15% of GPU memory (~28GB per B200) for temporary tensors. KV cache capacity is still massive (MLA = 50× smaller cache than standard MHA).

**Also fixed**: Job QoS bumped from `1d`/10h to `3d`/72h — prevents overnight expiration.

| Config | Before | After |
|--------|--------|-------|
| `--mem-fraction-static` | auto (~0.95) | **0.85** |
| `--qos` | 1d | **3d** |
| `--time` | 10:00:00 | **3-00:00:00** |
| Free GPU memory | ~2GB | **~28GB** |

### GLM-5.2: Deployed (2026-06-20) — 220-250 tok/s, 1M context

| Spec | Value |
|------|-------|
| Model | `zai-org/GLM-5.2-FP8` (MIT license) |
| Architecture | `GlmMoeDsaForCausalLM` — MoE + DSA + IndexShare sparse attention |
| Total params | 754B (78 layers, vs 5.1's 61) |
| Active params | ~40B (8 of 256 experts) |
| Context | **1,048,576 tokens (1M)** |
| FP8 size | 704 GB → 8× B200 (TP=8, ~88GB/GPU weights + ~68GB KV headroom) |
| Throughput | **220-250 tok/s** (EAGLE 5-1-6 speculative) |
| Port | 8103 |
| Engine | `lmsysorg/sglang:latest` Apptainer container |
| Serve script | `serving/serve-glm52-sglang-latest.sh` |
| Job | 2183826, fat001 |

**The deployment saga (2026-06-20):**

Getting GLM-5.2 running took a full day of debugging across four failed approaches before finding the root cause:

1. **SGLang pip 0.5.13** (`serve-glm52.sh`): Crashed during inference with CUDA kernel error in IndexShare DSA attention. Not a config issue — SM100 DSA kernels are simply not compiled into pip 0.5.13.

2. **vLLM container (Apptainer)** (`serve-glm52-vllm.sh`): NCCL TP=8 deadlock after weights loaded. Silent hang, no error.

3. **vLLM native pip 0.23.0** (`serve-glm52-vllm-native.sh`): Installed Python 3.12 venv, loaded all 141 shards in ~17 min — then deadlocked at `MoEPrepareAndFinalizeNoDPEPMonolithic`. Adding `--enforce-eager` made no difference. 0% GPU utilization with workers alive = kernel deadlock, not graph capture issue.

4. **Root cause found via SGLang GitHub research**: The [official SGLang cookbook for B200](https://github.com/sgl-project/sglang/blob/main/docs_new/src/snippets/configs/zai-org/glm-5.2.jsx) explicitly notes: *"DSA prefill Context Parallel is verified on Hopper (H200); the Blackwell sm100 DSA-CP FP8 rope kernel is not yet adapted."* The SM100 DSA kernels are only compiled into `lmsysorg/sglang:latest` (Docker image) — not in pip. Everyone else running GLM-5.2 on B200 uses the container.

5. **SGLang latest container — first crash**: `OSError: [Errno 12] Cannot allocate memory` — FlashInfer's SM100 trtllm cubin JIT needs to write symlinks to `/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins` inside the container, which is read-only under `--cleanenv --writable-tmpfs`.

6. **Fix attempt — wrong bind**: Bound entire `flashinfer_cubin` package → broke `flashinfer_cubin.__version__` → `AttributeError: module 'flashinfer_cubin' has no attribute '__version__'`.

7. **Correct fix**: Bind writable `$LOCAL_TMP/flashinfer-cubins` → `/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins` (the `cubins` subdirectory only). This lets JIT write symlinks without clobbering the package's `__init__.py` and metadata.

8. **Success**: Server came up, 5/5 stress test queries passed, 156GB VRAM per GPU, thinking mode active by default. GLM-5.1 scancel'd.

**Key lesson**: When "everyone else runs it" but you can't — check whether they're using the container or pip. Container images ship compiled CUDA kernels for new GPU architectures weeks before pip packages catch up.

---

### GLM-5.1: Deployed (2026-05-17) — 167 tok/s

| Spec | Value |
|------|-------|
| Model | `zai-org/GLM-5.1-FP8` (MIT license) |
| Architecture | `GlmMoeDsaForCausalLM` — MoE + DSA (DeepSeek-like MLA) |
| Total params | 754B |
| Active params | 40B (8 of 256 experts + 1 shared) |
| Context | 202,752 tokens (~200K) |
| KV cache | DSA compressed (kv_lora_rank=512) — small like MLA |
| EAGLE | Native MTP (`num_nextn_predict_layers: 1`) |
| FP8 size | 756 GB → 8× B200 (TP=8, 94.5GB/GPU) |
| Throughput | **167 tok/s** |
| Port | 8103 |

**Why GLM-5.1**: Same tier as V4-Flash on benchmarks but with 40B active (vs 12B) — deeper reasoning per token for complex coding tasks. Runs on 8× B200 (full fat node). See "GLM-5.1 Deployment" section at top for the full deployment journey.

### Binary Analysis: What Claude Code Sends to Anthropic

| Category | Endpoint | Blocked By |
|----------|----------|-----------|
| LLM API calls | `ANTHROPIC_BASE_URL` | ✓ Redirected to local proxy |
| OAuth | `api.anthropic.com/oauth` | ✓ `CLAUDE_CODE_SKIP_OAUTH=1` |
| Telemetry | `api.anthropic.com/api/event_logging/batch` | ✓ `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` |
| MCP registry | `api.anthropic.com/mcp-registry/v0/servers` | ✓ Same env var |
| Org metrics | `api.anthropic.com/api/claude_code/organizations/metrics_enabled` | ✓ Same env var |

## Agentic Coding on HPC — First Experiment (2026-05-11)

**Status**: Superseded by multi-model architecture above. Original `hpc-claude` repo deleted.  
**Outcome**: Proved V4-Flash can power Claude Code at 190 tok/s. Single-proxy architecture (port 8082) replaced by multi-model proxies (5005/5006/5007).

### V4-Flash vs Gemma-4 MMM Benchmark

Ran head-to-head comparison (6 tasks: 3 MMM domain + 3 general coding, judged by Opus 4.6):

| Metric | V4-Flash (4× B200) | Gemma-4-MMM (2× L40S) |
|--------|--------------------|-----------------------|
| Speed | **207 tok/s** | 10 tok/s |
| Speedup | **21.4×** | 1× |
| Quality wins | 1/6 | **4/6** |
| MMM domain | 1/2 | 1/2 |
| Coding | 0/3 | **3/3** |

**Key insight**: Gemma-4 MMM wins on quality (it's fine-tuned on pharma domain data), but V4-Flash is 21× faster. For agent swarms where throughput matters, V4-Flash is the right choice. For single-turn MMM consulting, Gemma-4 produces better domain answers.

**Caveat**: Both models hit 1024 token limit (truncated responses). Gemma's slower generation meant its truncation point sometimes had more complete code. A fairer test would use higher max_tokens.

### HPC Repo Cleanup (same session)

Removed ~200GB of obsolete files:
- Deprecated Qwen3.6 models (15% win rate, 67GB base + 67GB FP8 + 67GB merged)
- Old Gemma-4 SFT+DPO merged model (replaced by SFT-only production)
- 10+ obsolete serving scripts (old Qwen, pre-container V4-Flash attempts)
- Source repos (`flash-mla-src/`, `sglang-dsv4/`) — 5GB
- Empty directories (`tmp/`, `benchmarks/`, `adapters/`, `jobs/`)

Final HPC structure:
```
/shared/project/<account>/llm/
├── .serve-state.json          # Active coding model (V4-Flash)
├── .serve-state-gemma.json    # Active MMM model (Gemma)
├── containers/                # sglang-dsv4-blackwell.sif (20GB)
├── logs/                      # Slurm output
├── models/
│   ├── deepseek-v4-flash/     # 149GB — primary coding
│   ├── gemma-4-31b-it/        # 59GB — base for fine-tuning
│   ├── gemma-4-31b-mmm-sft/   # 58GB — production MMM
│   ├── qwen3-235b-a22b/       # 438GB — backup coding
│   ├── qwen3-asr-1.7b/        # 4.4GB — voice ASR
│   └── qwen3-tts-1.7b/        # 4.3GB — voice TTS
├── scripts/                   # Training, merge, data extraction
├── serving/                   # Production: v4flash-container, gemma4-l40s, qwen3-235b
├── training/                  # data/ and output/ (checkpoints)
├── training-venv/             # Python env for training
└── venv/                      # SGLang serving env
```

## DeepSeek-V4-Flash Deployment (2026-05-11)

**Status**: Production serving, 190 tok/s, 4× B200 via Apptainer container  
**Motivation**: Need a fast, high-concurrency coding model for agent swarms. V4-Flash has 282B total params (MoE) but only 12B active per token — extreme throughput with massive knowledge base.

### The Problem: MLA Kernel Deadlock on B200/SM100

DeepSeek-V4-Flash uses Multi-Latent Attention (MLA) — a compressed attention mechanism with 1 KV head that enables 50× smaller KV cache than standard MHA. The `flash_mla` kernel (pip-installed via SGLang 0.5.11) deadlocks on B200 GPUs (SM100/Blackwell architecture), despite having SM100 cubins in the package.

**Root cause**: The pip `flash_mla` package was compiled against an older CUDA toolkit. On SM100, the MLA forward pass hangs indefinitely during CUDA graph capture. Everything else works:
- Weights load correctly (147GB FP4, TP=4 → 37GB/GPU)
- MoE expert backends work (marlin, flashinfer_mxfp4)
- Only the MLA attention kernel deadlocks

### The Solution: Official Docker Image via Apptainer

The `lmsysorg/sglang:deepseek-v4-blackwell` Docker image contains a verified build of SGLang 0.5.10rc0 with Python 3.12 and a working `flash_mla` for Blackwell GPUs.

**Pipeline**: Docker → Apptainer SIF → Slurm job

| Step | Script | Time | Notes |
|------|--------|------|-------|
| Pull container | `pull-container.sh` | ~45 min | 20GB SIF, needs 128GB RAM for conversion |
| Serve model | `serve-v4flash-container.sh` | ~3 min load | 4× B200, TP=4, port 8100 |

### Key Apptainer Flags (Each Learned the Hard Way)

| Flag | Why Needed | Without It |
|------|-----------|------------|
| `--nv` | Expose NVIDIA GPUs to container | No CUDA devices visible |
| `--cleanenv` | Prevent host PATH leaking in | Triton tries host GCC (nonexistent Spack path) |
| `--writable-tmpfs` | Allow runtime writes to container FS | FlashInfer can't write cubin symlinks |
| `--env PATH=...` | Set container-internal PATH | Picks up broken host paths |
| `--env HOME=/tmp` | DeepGEMM JIT writes to HOME | Permission denied on read-only / |
| `--bind $MODEL:/models/...` | Mount weights into container | Model not found |
| `--bind $LLM_DIR:/data` | NOT /workspace (overlaps SGLang) | Overwrites container's sglang package |

### EAGLE Speculative Decoding

V4-Flash uses EAGLE (Extrapolation Algorithm for Greater Language model Efficiency) — speculative decoding using the model's built-in Multi-Token Prediction (MTP) heads:

```
Standard decoding:     [token] → [token] → [token] → ...  (1 forward pass per token)
EAGLE speculation:     [token] → [draft 4 tokens] → [verify] → [accept 2-3] → ...
```

- **3 speculative steps**, top-k=1, 4 draft tokens per step
- ~70% acceptance rate → effective 2.8× speedup
- Zero quality loss (verification step rejects bad drafts)
- Combined with CUDA graphs (306 graphs across 51 batch sizes): 8.9 → 190 tok/s

### Throughput Comparison

| Model | Active Params | Hardware | Throughput | Latency (2K tokens) |
|-------|--------------|----------|-----------|---------------------|
| **DeepSeek-V4-Flash** | **12B** | **4× B200** | **207 tok/s** | **7.8s** |
| Claude Sonnet 4.6 | — | AWS Bedrock | 72 tok/s | 26.4s |
| Qwen3-235B-A22B | 36B | 4× B200 | ~65 tok/s | ~30s |
| Gemma-4-31B-MMM | 31B | 2× L40S | 23 tok/s | ~90s |

### V4-Flash vs Sonnet 4.6 Coding Benchmark

8 coding tasks (algorithms, debugging, architecture, refactoring, Rust, ML, parsing, concurrency). Judged by Opus 4.6 with position-randomized blind evaluation.

| Metric | V4-Flash | Sonnet 4.6 |
|--------|----------|-----------|
| **Win Rate** | **69%** | 31% |
| Wins | 5/8 | 2/8 |
| Ties | 1/8 | — |
| Avg Throughput | 207 tok/s | 72 tok/s |
| Avg Latency | 7.8s | 26.4s |
| Speed Advantage | **2.9×** | — |

**Where V4-Flash wins**: LRU cache, race condition debugging, SQL refactoring, ML pipelines, cron parsing  
**Where Sonnet wins**: Distributed system design, async web scraper (deeper architectural reasoning)  
**Key insight**: V4-Flash's 282B expert knowledge base + 12B active compute delivers Sonnet-tier quality at 3× the speed. For agent swarms needing many concurrent sessions, V4-Flash is the clear winner.

### Attempt Timeline

```
Job ID    │ Issue                              │ Fix
──────────┼────────────────────────────────────┼───────────────────────────────────
1880802   │ pull-container: 30min wall timeout │ Increased to --time=02:00:00
1881750   │ ModuleNotFoundError: sglang        │ Changed bind from /workspace to /data
1881831   │ FileNotFoundError: gcc             │ Added --cleanenv + explicit --env PATH
1881888   │ OSError: Read-only file system     │ Added --writable-tmpfs
1882010   │ 8.9 tok/s (no optimization)       │ Removed --disable-cuda-graph, added EAGLE
1882695   │ ✅ PRODUCTION (190 tok/s)          │ Full recipe: CUDA graphs + EAGLE + MoE backend
```

### Production Config

```bash
apptainer exec --nv --cleanenv --writable-tmpfs \
    --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "HOME=/tmp" --env "TRITON_CACHE_DIR=/tmp/triton_cache" \
    --bind $MODEL:/models/deepseek-v4-flash --bind $LLM_DIR:/data \
    $SIF python3 -m sglang.launch_server \
        --model-path /models/deepseek-v4-flash \
        --host 0.0.0.0 --port 8100 --tp-size 4 --trust-remote-code \
        --moe-runner-backend flashinfer_mxfp4 \
        --speculative-algo EAGLE --speculative-num-steps 3 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --chunked-prefill-size 4096 --disable-flashinfer-autotune \
        --context-length 65536 --served-model-name deepseek-v4-flash
```

---

## Multi-Model Serving Architecture (2026-05-09)

**Status**: Three models served simultaneously on heterogeneous hardware  
**Motivation**: Different workloads need different models — coding agent swarms need speed and concurrency, MMM domain work needs specialized knowledge.

### Production Deployment

| Model | Purpose | Hardware | Port | Throughput | Status |
|-------|---------|----------|------|-----------|--------|
| **GLM-5.1** | Alt coding | 8× B200 (SGLang pip + EAGLE) | 8103 | **167 tok/s** | ACTIVE |
| DeepSeek-V4-Flash | Coding swarms | 4× B200 (container) | 8100 | 190 tok/s | Ready |
| Gemma-4-31B-MMM-SFT | Pharma MMM | 2× L40S (native) | 8200 | 23 tok/s | ACTIVE |
| ~~Qwen3-235B-A22B~~ | ~~Coding backup~~ | — | — | — | DELETED |

### Gemma-4 on L40S (Freed B200s for V4-Flash)

Moved Gemma from 1× B200 (overkill for 62GB model) to 2× L40S (48GB each):
- Same BF16 weights, `device_map="auto"` splits ~31GB per GPU
- Zero quality/accuracy loss — mathematically identical inference
- ~5-10ms inter-GPU latency, negligible at 20 tok/s generation
- SGLang incompatible with Gemma-4 → native Transformers (same as before)

### Resource Efficiency

```
Total cluster:  64 B200 (fat) + 176 L40S (gpu)
Used:           8 B200 (GLM-5.1) + 2 L40S (Gemma-4)
Available:      56 B200 + 174 L40S (87% idle)
```

---

## Gemma-4-31B-IT → Pharma MMM Agent (2026-05-07)

**Status**: SFT+GRPO complete, eval complete, re-merge in progress  
**Motivation**: Qwen3.6-35B-A3B (MoE, 3B active) loses 80% of evaluations vs Sonnet 4.6. Root causes: insufficient active compute (3B vs 31B needed), truncated responses from thinking mode, and only 12 SFT examples.

**Model choice**: Gemma-4-31B-IT (dense, 32.7B all active, Arena #39 score 1451). Selected over:
- Qwen3.5-397B-A17B (too large for single GPU, Arena #49 — lower than Gemma)
- DeepSeek-V4-Flash (unranked, 158B MoE)
- GLM-5.1 (754B, doesn't fit any consumer hardware)

**Key improvements over Qwen pipeline**:
1. 10x more active compute per token (31B vs 3B)
2. Training data: 135 Opus-4.6-authored examples (vs 12 extracted from logs)
3. No linear attention issues — gradient checkpointing works cleanly
4. MTP drafter support for 3x inference speedup
5. Higher max_completion_length in GRPO (1024 vs 512)

### Training Results

| Stage | Job ID | Node | Runtime | Loss | VRAM |
|-------|--------|------|---------|------|------|
| SFT | 1856810 | fat-node-006 | ~5 min | N/A | ~79 GB |
| GRPO | 1856987 | fat-node-006 | 11,820s (3h17m) | 0.0009 | 68.5/149 GB |
| Merge | 1861257 | CPU | ~2 min | — | — |

**GRPO reward metrics (final step)**:
- criteria_awareness: 0.56 (understands gate thresholds)
- evidence_reasoning: 0.36 (quantitative domain arguments)
- domain_correctness: 0.08 (avoids version bugs)
- length: 0.30 (sweet spot output length)
- structure: 0.00 (model prefers natural prose over ## headers)
- config_format: 0.00 (doesn't use backtick-format configs often)

### Evaluation Results (SFT-only, 2026-05-07)

**Win rate: 59% against Sonnet 4.6** (target was 40-50%)

| Metric | Sonnet 4.6 | Gemma-4 MMM | Improvement vs Qwen |
|--------|-----------|-------------|---------------------|
| Wins | 7 (26%) | **16 (59%)** | 15% → 59% (+44pp) |
| Ties | 4 (15%) | 4 (15%) | — |
| Accuracy | 7.0/10 | **7.9/10** | 6.8 → 7.9 |
| Completeness | 8.0/10 | **8.3/10** | 7.2 → 8.3 |
| Evidence | 6.2/10 | **7.0/10** | 5.3 → 7.0 |
| Domain Expertise | 7.3/10 | **8.2/10** | 6.6 → 8.2 |

**Strongest categories** (2-0 or 3-0 sweeps):
- channel_attribution (2-0): Correctly identifies cross-sectional confounding, Simpson's paradox
- debugging (2-0): Knows ROAS ceiling is post-estimation, not Bayesian layer
- general_mmm (3-0): Deep knowledge of adstock/saturation interaction, identifiability
- technical_constraints (2-0): Knows pymc-marketing 0.18.0 vs 0.19.x bugs, locked params

**Weakest categories** (where Sonnet still wins):
- config_generation (2-1): Sonnet provides more executable code examples
- scenario_planning (1-0): Sonnet better at uncertainty quantification

### DPO Training & Evaluation (2026-05-08)

**Stage 3: Rejection Sampling** (local, scripts/generate_dpo_data.py)
- Generated 4 responses per prompt from SFT model (temp 0.7-0.85)
- Opus 4.6 scored each on 4 dimensions (accuracy, completeness, evidence, expertise /10)
- Best vs worst paired → 135 DPO pairs, avg score gap 4.8/40
- 50% of pairs had gap ≥ 5 (strong preference signal)

**Stage 4: DPO Training** (Job 1867017, fat-node-006)

| Metric | Value |
|--------|-------|
| Runtime | 469s (7m49s) |
| Steps | 68 |
| Final loss | 0.6864 (down from 0.693 = log(2)) |
| Reference model | gemma-4-31b-mmm-sft (merged SFT) |
| LoRA | r=64, alpha=128, same targets |
| Config | beta=0.1, lr=5e-7, epochs=2, sigmoid loss |
| VRAM peak | ~91 GB on B200 |

**Merge pipeline:**
1. SFT-only merge (job 1866761): base + SFT adapter → `gemma-4-31b-mmm-sft` (2m57s)
2. DPO merge (job 1867021): SFT model + DPO adapter → `gemma-4-31b-mmm` (1m45s)

**Evaluation: SFT+DPO model (2026-05-08)**

| Metric | Sonnet 4.6 | Gemma-4 SFT+DPO |
|--------|-----------|------------------|
| Wins | 10 (38%) | **13 (50%)** |
| Ties | 3 (12%) | 3 (12%) |
| Accuracy | 7.5/10 | **7.9/10** |
| Completeness | 8.0/10 | **8.3/10** |
| Evidence | 6.5/10 | **6.9/10** |
| Domain Expertise | 7.5/10 | **8.0/10** |

### Full Model Progression

| Stage | Win Rate | Wins | Losses | Ties | Notes |
|-------|----------|------|--------|------|-------|
| Qwen3.6-35B-A3B | 15% | 4 | 16 | — | 3B active, 12 SFT examples |
| Gemma-4 base (no fine-tuning) | 18% | 5 | 20 | 2 | Raw model, no domain knowledge |
| Gemma SFT+DPO | 50% | 13 | 10 | 3 | DPO didn't improve on SFT |
| Gemma SFT+GRPO | 52% | 14 | 13 | 0 | GRPO regressed — reward functions too brittle |
| **Gemma SFT-only** | **59%** | 16 | 7 | 4 | Best overall, Opus-authored data |

**Key insight**: Base Gemma-4-31B at 18% proves the SFT lift is real (+41pp). The raw model — despite 31B dense params — performs only marginally better than 3B-active Qwen. Domain knowledge (gate thresholds, pymc-marketing bugs, channel confounding) is entirely learned from SFT, not inherent model capability.

**Conclusion**: SFT-only remains the strongest configuration. DPO training with 135 pairs didn't surpass the SFT model that was already trained on Opus 4.6 responses directly. The preference signal from rejection sampling was too weak — the SFT model already produces near-Opus-quality outputs, so the gap between "best of 4" and "worst of 4" samples is small (avg 4.8/40). GRPO similarly regressed, likely due to brittle reward functions (structure=0, config_format=0) penalizing valid natural-language responses.

**Best model for production**: `gemma-4-31b-mmm-sft` (SFT-only merged, 59% win rate)

### Serving Challenge: SGLang Incompatibility

SGLang 0.5.10 is **fundamentally incompatible** with Gemma-4 due to:
1. Mixed attention heads (16 KV heads in full-attention layers, 8 in sliding-window layers)
2. `v_norm` with `with_scale=False` (no learnable weight) breaks fused kernel replacement
3. Multimodal detection heuristic (`hf_config is not hf_text_config`) catches text-only config

**Resolution**: Native Transformers serving via FastAPI/uvicorn at ~20 tok/s. Sufficient for evaluation but not production. Future: wait for SGLang/vLLM Gemma-4 support.

### Training pipeline
- Data generation: `scripts/generate_sft_data.py` (Opus 4.6 via Uptimize Bedrock)
- SFT: `train_gemma4_sft.py` + `train_gemma4_sft.sh`
- GRPO: `train_gemma4_grpo.py` + `train_gemma4_grpo.sh`
- Merge: `scripts/merge_gemma4_cpu.sh` (on HPC)
- Serve: `serving/serve-gemma4-native.sh` (native Transformers, B200)
- Eval: `eval/run_eval_gemma4.py`

---

## Qwen3.6-35B-A3B → Pharma MMM Agent (2026-04-30)

**Session duration**: ~2h 45m  
**Outcome**: Both SFT and GRPO training completed successfully  
**Final hardware**: 1× NVIDIA B200 (192GB VRAM) on `fat` partition  

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE (COMPLETED)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐        │
│  │ Meta-Harness │     │  extract_training │     │  /llm/training/data/ │        │
│  │ Iteration    │────▶│  _data.py         │────▶│  sft_train.jsonl     │        │
│  │ History      │     │                  │     │  dpo_train.jsonl     │        │
│  │ (14 iters)   │     │  12 SFT pairs    │     │  stats.json          │        │
│  └──────────────┘     │   4 DPO pairs    │     └──────────┬───────────┘        │
│                        └──────────────────┘                │                    │
│                                                            │                    │
│  ┌─────────────────────────────────────────────────────────┼────────────────┐   │
│  │                         STAGE 1: SFT (LoRA)             │                │   │
│  │                                                         ▼                │   │
│  │  ┌───────────────────┐     ┌────────────────────────────────────┐       │   │
│  │  │ Qwen3.6-35B-A3B   │     │  train_qlora.py                    │       │   │
│  │  │ (BF16, 67GB)      │────▶│  • LoRA r=64, alpha=128           │       │   │
│  │  │ device_map={"":0}  │     │  • target: q/k/v/o + gate/up/down │       │   │
│  │  │ on B200 (192GB)   │     │  • 33.4M trainable / 34.7B total  │       │   │
│  │  └───────────────────┘     │  • 3 epochs, lr=2e-4, seq=4096    │       │   │
│  │                            │  • gradient_checkpointing=True     │       │   │
│  │                            └─────────────────┬──────────────────┘       │   │
│  │                                              │                          │   │
│  │                                              ▼                          │   │
│  │                            ┌────────────────────────────────────┐       │   │
│  │                            │  OUTPUT: qlora-20260430-1213/      │       │   │
│  │                            │  • adapter/ (LoRA weights)         │       │   │
│  │                            │  • Loss: 2.57 → 1.29              │       │   │
│  │                            │  • Accuracy: 50% → 68%            │       │   │
│  │                            │  • Runtime: 71 seconds             │       │   │
│  │                            │  • VRAM: 69.5GB / 183GB           │       │   │
│  │                            └─────────────────┬──────────────────┘       │   │
│  └──────────────────────────────────────────────┼──────────────────────────┘   │
│                                                 │                              │
│  ┌──────────────────────────────────────────────┼──────────────────────────┐   │
│  │                         STAGE 2: GRPO                    │              │   │
│  │                                                          ▼              │   │
│  │  ┌───────────────────┐     ┌────────────────────────────────────┐      │   │
│  │  │ Qwen3.6-35B-A3B   │     │  train_grpo.py                     │      │   │
│  │  │ (BF16, 67GB)      │────▶│  • Fresh LoRA (same config)        │      │   │
│  │  │ + model_init_kwargs│     │  • 4 generations per prompt        │      │   │
│  │  │ on B200 (192GB)   │     │  • 512 max completion tokens       │      │   │
│  │  └───────────────────┘     │  • 6 rule-based reward functions   │      │   │
│  │                            │  • loss_type="grpo", beta=0.01     │      │   │
│  │                            │  • 1 epoch, lr=5e-6               │      │   │
│  │                            └─────────────────┬──────────────────┘      │   │
│  │                                              │                         │   │
│  │  ┌────────────────────────────────────┐      │                         │   │
│  │  │  REWARD FUNCTIONS (rule-based)     │      │                         │   │
│  │  │                                    │      │                         │   │
│  │  │  reward_structure       (0.15)  ───┤      │                         │   │
│  │  │  reward_config_format   (0.20)  ───┤      │                         │   │
│  │  │  reward_gate_awareness  (0.20)  ───┼──────┤                         │   │
│  │  │  reward_evidence        (0.20)  ───┤      │                         │   │
│  │  │  reward_domain_correct  (0.15)  ───┤      │                         │   │
│  │  │  reward_length          (0.10)  ───┘      │                         │   │
│  │  └────────────────────────────────────┘      │                         │   │
│  │                                              ▼                         │   │
│  │                            ┌────────────────────────────────────┐      │   │
│  │                            │  OUTPUT: grpo-20260430-1234/       │      │   │
│  │                            │  • adapter/ (GRPO LoRA weights)    │      │   │
│  │                            │  • completions/ (12 parquet files) │      │   │
│  │                            │  • Loss: ~0 (expected for GRPO)    │      │   │
│  │                            │  • Runtime: 494s (8.2 min)         │      │   │
│  │                            │  • VRAM: 140.6GB / 183GB          │      │   │
│  │                            └────────────────────────────────────┘      │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                     CURRENT STATUS: NOT INFERENCING                     │   │
│  │                                                                        │   │
│  │  Adapters are saved on disk. No serving job, no inference endpoint.    │   │
│  │  To use: load base model + adapter with PeftModel.from_pretrained()    │   │
│  │  Next: merge adapters → deploy as vLLM endpoint or integrate into      │   │
│  │  meta-harness as the proposer agent.                                   │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## The Journey: Why Each Step Changed

### Attempt Timeline (Chronological)

```
Job ID    │ Partition │ GPU       │ Approach                    │ Outcome
──────────┼───────────┼───────────┼─────────────────────────────┼──────────────────────
1803902   │ gpu       │ 2× L40S  │ QLoRA 4-bit, max_seq_length │ ❌ TRL API error
1803961   │ gpu       │ 2× L40S  │ FP8 quantization            │ ❌ FP8 rejected by validator
1803994   │ gpu       │ 2× L40S  │ 4-bit, mem=64G              │ ❌ CPU OOM (67GB model)
1804014   │ gpu       │ 2× L40S  │ 4-bit, mem=128G             │ ❌ GPU OOM in kbit prep
1804038   │ gpu       │ 2× L40S  │ 4-bit, no max_memory        │ ❌ CPU module dispatch
1804052   │ gpu       │ 2× L40S  │ 4-bit, device_map="auto"    │ ❌ NLL loss assertion
1804140   │ gpu       │ 2× L40S  │ + grad_ckpt=False           │ ❌ NLL loss assertion
1804176   │ gpu       │ 2× L40S  │ + label validation added    │ ❌ Labels valid, still NLL
1804209   │ gpu       │ 2× L40S  │ FP8 (kernels installed)     │ ❌ ImportError: kernels
1804237   │ gpu       │ 2× L40S  │ FP8 monkey-patch validator  │ ❌ Patch didn't stick
1804272   │ gpu       │ 2× L40S  │ FP8 double-patch            │ ❌ No autograd for FP8
1804317   │ gpu       │ 1× L40S  │ 4-bit single GPU            │ ❌ OOM during quant load
1804319   │ gpu       │ 2× L40S  │ 4-bit, grad_ckpt=False      │ ❌ OOM in backward pass
1804321   │ gpu       │ 2× L40S  │ 4-bit, grad_ckpt=True       │ ❌ NLL assertion returns
1804324   │ gpu       │ 2× L40S  │ Confirmed: ckpt + multi = 💀│ ❌ Same NLL assertion
──────────┼───────────┼───────────┼─────────────────────────────┼──────────────────────
1804326   │ fat       │ 1× B200  │ BF16, no quant, single GPU  │ ✅ SFT COMPLETE (71s)
1804330   │ fat       │ 1× B200  │ GRPO (reward_func format)   │ ❌ 'list' has no 'lower'
1804336   │ fat       │ 1× B200  │ GRPO (reward_weights)       │ ❌ list.to() error
1804337   │ fat       │ 1× B200  │ GRPO (all fixes)            │ ✅ GRPO COMPLETE (494s)
──────────┴───────────┴───────────┴─────────────────────────────┴──────────────────────

Total jobs submitted: 19
Success rate: 2/19 (10.5%) — but these 2 are the only ones that matter.
```

---

## Decision Points & Reasoning

### 1. Initial Approach: QLoRA on 2× L40S (Jobs 1803902–1804324)

**Why this seemed right:**
- Qwen3.6-35B-A3B is a MoE model (35B total, 3B active per token)
- At 4-bit quantization: ~9GB. Should fit on a single L40S (48GB) easily.
- Standard recipe: bitsandbytes 4-bit NF4 + LoRA. Works for most models.

**What went wrong (root cause discovery):**

```
┌─────────────────────────────────────────────────────────────────┐
│  ROOT CAUSE: Linear Attention + Gradient Checkpointing          │
│  + Multi-GPU device_map = Corrupted Forward Pass                │
│                                                                 │
│  Qwen3.6-35B-A3B uses "torch_chunk_gated_delta_rule" linear    │
│  attention (NOT standard self-attention). This architecture     │
│  maintains internal state that gradient checkpointing's         │
│  recomputation phase corrupts when tensors span multiple GPUs.  │
│                                                                 │
│  The corruption manifests as logits containing values that      │
│  produce labels >= vocab_size (248320), triggering:             │
│    CUDA error: device-side assert triggered                     │
│    nll_loss_forward: assertion `t >= 0 && t < n_classes`        │
│                                                                 │
│  This is NOT a data issue (labels validated as correct).        │
│  This is NOT a standard OOM (forward pass completes).           │
│  This is a fundamental architecture incompatibility.            │
└─────────────────────────────────────────────────────────────────┘
```

**Why each variant failed:**

| Approach | Why it failed | What I learned |
|----------|---------------|----------------|
| `max_seq_length` | TRL 1.3.0 renamed to `max_length` | Always check API version |
| FP8 | No backward pass implementation | FP8 is inference-only |
| 4-bit + `mem=64G` | BF16 model passes through CPU during quantization (67GB) | Quant needs peak CPU RAM |
| `prepare_model_for_kbit_training` | Converts layernorms to FP32, exceeds GPU 0 | Remove this for multi-GPU |
| `device_map="auto"` + `grad_ckpt=True` | NLL assertion (root cause above) | Architecture incompatibility |
| `device_map="auto"` + `grad_ckpt=False` | OOM in backward on GPU 1 | Linear attention uses >44GB for backward |
| Single L40S | Can't hold 4-bit model during quantization conversion | 44GB too tight for 35B even at 4-bit |

### 2. The Pivot: B200 on Fat Partition (Job 1804326)

**Why this was the right answer:**

```
                    L40S (48GB)                    B200 (192GB)
                   ┌───────────┐                  ┌───────────────────────────┐
                   │ ████████░░│ 44.4GB usable    │ ████████░░░░░░░░░░░░░░░░░│ 183GB usable
                   │ ████████░░│                  │                           │
                   │ ████████░░│ Model (4-bit):   │ Model (BF16): 67GB        │
                   │ █████░░░░░│ 9GB              │ ████████████████░░░░░░░░░│
                   │ ░░░░░░░░░░│ + Overhead:      │ LoRA: 1GB                │
                   │ ░░░░░░░░░░│ OOM during load  │ █░░░░░░░░░░░░░░░░░░░░░░░│
                   └───────────┘                  │ Optimizer: 4GB            │
                   ❌ CAN'T FIT                   │ ██░░░░░░░░░░░░░░░░░░░░░░░│
                                                  │ Activations: 8GB          │
                                                  │ ███░░░░░░░░░░░░░░░░░░░░░░│
                                                  │                           │
                                                  │ FREE: ~103GB headroom     │
                                                  │ ░░░░░░░░░░░░░░░░░░░░░░░░░│
                                                  └───────────────────────────┘
                                                  ✅ SINGLE DEVICE, NO MULTI-GPU BUGS
```

**Key insight**: By eliminating quantization AND multi-GPU, we eliminate:
- All quantization loading issues
- All device_map split bugs
- All gradient_checkpointing recomputation corruption
- The need for `prepare_model_for_kbit_training`

Trade-off: We use a "fat" node (more expensive in terms of cluster resources), but training completes in **71 seconds** for SFT and **8 minutes** for GRPO. The node is free for other users immediately after.

### 3. GRPO Reward Function Interface (Jobs 1804330–1804337)

**What I discovered about TRL 1.3.0 GRPO:**

```python
# What I assumed:
def reward_fn(completions: list[str], **kwargs) -> list[float]:
    for text in completions:  # text is a string
        text.lower()  # ← works

# What TRL 1.3.0 actually passes (chat model path):
def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
    for completion in completions:  # completion is [{"role": "assistant", "content": "..."}]
        completion.lower()  # ← AttributeError: 'list' has no 'lower'
```

TRL 1.3.0 has three codepaths in `_generate()`:
1. Tool-calling models → `parse_response()` → `list[list[dict]]`
2. Chat models (our case) → `[{"role": "assistant", "content": text}]` → `list[list[dict]]`
3. Plain text models → `batch_decode()` → `list[str]`

Since we're passing `prompt` as a list of message dicts (chat format), TRL takes path 2.

**Fix**: Added `_extract_text()` helper that handles both formats.

**Second issue**: `reward_weights` must be passed via `GRPOConfig(reward_weights=[...])`, not set as `trainer.reward_weights = [...]`. The config converts it to a tensor internally; setting it manually leaves it as a Python list that can't be `.to(device)`.

---

## HPC Cluster Observations

### What I Learned About oneHPC

| Observation | Impact |
|-------------|--------|
| **Fat partition has immediate availability** | All 3 fat jobs started in <15 seconds. No queue wait. The fat nodes are underutilized — 8 nodes × 8 B200 GPUs = 64 B200s total, we used 1. |
| **B200 reports 183,359 MiB (not 192GB)** | ~9GB reserved for ECC + system. Still 179GB usable for CUDA allocations. |
| **Model loads in ~15 seconds** | 1026 weight shards at ~73 shards/sec. NVMe local storage is fast. |
| **No flash-linear-attention installed** | Logs show: "fast path not available". Model falls back to torch implementation. Could install `fla` for 2-3× speedup. |
| **CUDA 12.9.0 available** | B200 (Blackwell) is fully supported. No driver issues. |
| **Python 3.11 in our venv** | Good — PyTorch + Transformers + TRL all compatible. |
| **Node fat-node-006 allocated for all 3 jobs** | Same node each time (likely because it was already warm/allocated). |
| **Generation speed on B200** | ~41 seconds to generate 4 × 512-token completions (2048 tokens total). That's ~50 tok/s for a 35B BF16 model — reasonable without FlashAttention. |
| **No other users on fat partition** | All 3 jobs started instantly. The fat partition appears lightly used (B200 is new hardware, most users haven't migrated from L40S). |

### Cluster Capacity Context

```
Fat Partition (our target):
  8 nodes × 8× B200 (192GB each) = 64 GPUs, 12.3 TB total VRAM
  We used: 1 GPU for <10 minutes total
  Utilization: 0.026% of available GPU-hours

GPU Partition (where we started):
  44 nodes × 4× L40S (48GB each) = 176 GPUs, 8.4 TB total VRAM
  Problem: Individual L40S too small for this model
  Even 2× L40S (96GB) hit architecture bugs
```

---

## Why "v11_fast" in the GRPO Completion

The model generated a response referencing "v11_fast" because **that's what's in the training data**.

Our SFT training pairs come from the meta-harness iteration log — 14 iterations of Bayesian MMM optimization. The prompt the model was responding to included diagnostic context from iteration history:

```
Previous iteration (v11_fast):
  F2F plausibility: 70.3%
  Email attribution: 18.6%
  Gate failures: trust < 50, ess_bulk < 100
  ...
```

The model correctly:
1. Identified which iteration it was looking at (v11_fast)
2. Noted the specific gate failures (trust, ESS)
3. Started reasoning about prior shapes (Gamma, alpha parameterization)
4. Referenced the constraint (pymc-marketing 0.18.0)

This is exactly the behavior we want — the model learned the iteration history format from SFT and is now generating novel proposals in GRPO. The "v11_fast" wasn't cherry-picked; it was simply the 12th (last) prompt in the dataset, so it appeared in the final completion parquet file.

---

## Current Status

### What Exists Now

```
/shared/project/<account>/llm/
├── models/
│   └── qwen3.6-35b-a3b/              # Base model (67GB BF16)
├── training/
│   ├── data/
│   │   ├── sft_train.jsonl            # 12 training pairs
│   │   ├── dpo_train.jsonl            # 4 preference pairs (unused)
│   │   └── stats.json
│   └── output/
│       ├── qlora-20260430-1213/       # SFT adapter (951MB)
│       │   ├── adapter/
│       │   ├── checkpoint-4/
│       │   ├── checkpoint-6/
│       │   └── training_meta.json
│       └── grpo-20260430-1234/        # GRPO adapter (549MB)
│           ├── adapter/
│           ├── checkpoint-12/
│           ├── completions/           # 12 parquet files of generated reasoning
│           └── grpo_meta.json
├── scripts/
│   ├── train_qlora.py
│   ├── train_qlora.sh
│   ├── train_grpo.py
│   └── train_grpo.sh
├── training-venv/                     # Python 3.11 + PyTorch + TRL 1.3.0
└── logs/
    ├── qlora_1804326.out              # Successful SFT log
    └── grpo_1804337.out               # Successful GRPO log
```

### What is NOT Done (Next Steps)

| Step | Description | Effort |
|------|-------------|--------|
| **Merge adapters** | Merge SFT → base, then merge GRPO on top (or use separately) | 10 min script |
| **Inference endpoint** | Deploy merged model via vLLM or TGI on fat partition | 30 min |
| **Integration** | Connect to meta-harness as the proposer agent | 1-2 hours |
| **Evaluation** | Run model against held-out prompts, compare to base | 30 min |
| **DPO training** | We have 4 DPO pairs — could do DPO after GRPO | 15 min (same pattern) |
| **Data augmentation** | 12 SFT pairs is minimal — generate synthetic data | Variable |

### The Adapters Are Just Sitting There

Right now, both adapters are files on disk. They are **not** being served, not connected to any inference pipeline, and not integrated into the meta-harness. To use them, you'd need to:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base + adapter
model = AutoModelForCausalLM.from_pretrained("qwen3.6-35b-a3b", ...)
model = PeftModel.from_pretrained(model, "grpo-20260430-1234/adapter/")

# Or merge permanently
model = model.merge_and_unload()
model.save_pretrained("merged-mmm-agent/")
```

---

## Key Metrics Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING METRICS                              │
├─────────────────────────┬───────────────────────────────────────┤
│ SFT Loss Curve          │ GRPO Reward Scores (final step)       │
│                         │                                       │
│ 2.73 ─┐                │ gate_awareness:  ████████████░ 0.95   │
│ 2.57 ─┤ ╲              │ evidence:        ██████░░░░░░░ 0.61   │
│       │  ╲             │ domain:          █████░░░░░░░░ 0.45   │
│ 1.95 ─┤   ╲            │ length:          ███░░░░░░░░░░ 0.30   │
│       │    ╲           │ config_format:   ░░░░░░░░░░░░░ 0.00   │
│ 1.59 ─┤     ╲ ╱        │ structure:       ░░░░░░░░░░░░░ 0.00   │
│ 1.44 ─┤      ╳         │                                       │
│ 1.29 ─┤     ╱ ╲─final  │ Note: structure/config_format = 0     │
│       └─────────────    │ because model uses thinking-mode      │
│       e1  e2  e3        │ (not ## headers) — reward functions   │
│                         │ need tuning for this generation style. │
├─────────────────────────┴───────────────────────────────────────┤
│ VRAM Utilization                                                │
│                                                                 │
│ SFT:  ████████████████████████░░░░░░░░░░░░░░░ 69.5 / 183 GB    │
│ GRPO: █████████████████████████████████████░░░ 140.6 / 183 GB   │
│                                     ▲                           │
│                                     │                           │
│                    GRPO needs ~70GB extra for 4× generation     │
│                    buffers (4 completions × 512 tokens each)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Lessons Learned

1. **Model architecture matters more than parameter count.** Qwen3.6-35B-A3B's linear attention (`torch_chunk_gated_delta_rule`) is fundamentally incompatible with gradient checkpointing on split devices. No amount of configuration tuning fixes an architecture-level bug.

2. **When the standard recipe fails, go bigger not cleverer.** We spent 14 failed jobs trying to squeeze a 35B model onto 48GB GPUs with quantization tricks. The answer was: use a 192GB GPU and skip quantization entirely.

3. **TRL moves fast — always check the installed version's API.** Between TRL versions, `max_seq_length` → `max_length`, `TrainingArguments` → `SFTConfig`, and GRPO's reward function signature changed to chat-format messages.

4. **Rule-based rewards are sufficient for domain-specific GRPO.** We achieved 0.95 gate awareness without needing a trained reward model. The key insight: domain knowledge can be encoded as regex patterns and keyword matching.

5. **The HPC fat partition is a hidden gem.** B200 GPUs with 192GB VRAM, instant scheduling, and no queue wait. Most users are still on the L40S partition, leaving fat nodes idle.

---

## Deployment: Merge & Serve (2026-05-06)

**Session outcome**: Fine-tuned model deployed end-to-end. The trained model is now serving on port 8100.

### Adapter Merge (Job 1822232)

```
Node: fat-node-005 (B200, 183GB VRAM)
Runtime: 1m33s total
  - Base model load: 9.7s (693 weight shards → 69.3 GB BF16)
  - SFT adapter merge: 1.7s
  - GRPO adapter merge: 0.4s
  - Save merged model: 74.0s (21 shards × 4GB each)
Output: /shared/project/<account>/llm/models/qwen3.6-35b-a3b-mmm/ (69.3 GB)
```

### Serving (Job 1822552)

```
Node: fat-node-005 (B200)
Engine: SGLang 0.5.9
Model: qwen3.6-35b-a3b-mmm (BF16, 65.49 GB VRAM)
Port: 8100
Throughput: 33 tok/s (no CUDA graph)
Context: 65536 tokens
KV Cache: 24.83 GB K + 24.83 GB V
Mamba Cache: 43.65 GB (hybrid linear attention state)
Available VRAM after load: 17.65 GB
```

### Compatibility Issues Discovered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `--gres=gpu:1` rejected | oneHPC requires explicit GPU type | `--gres=gpu:b200:1` |
| `--qos=short` invalid | QoS names are `3h`, `1d`, `3d`, `7d`, `14d` | `--qos=3h` |
| DOS line endings | Windows `scp` doesn't convert | `sed -i 's/\r$//'` on HPC |
| PEFT 0.19.1 `WeightConverter` error | `distributed_operation` kwarg incompatible with transformers 5.7.0 | Downgrade to PEFT 0.15.2 |
| `Qwen3_5MoeForCausalLM` not recognized by SGLang | Newer transformers saves with `_text` suffix architecture class | Copy base model's `config.json` (uses `Qwen3_5MoeForConditionalGeneration`) |
| `transformers 5.8.0` breaks SGLang config parsing | `get_hf_text_config` assertion fails on new config format | Downgrade serving venv to transformers 4.57.1 |
| FlashInfer autotune hangs on B200 | First-time kernel JIT compilation for new GPU architecture | Wait ~8min + `--disable-cuda-graph` + `--attention-backend triton` |

### Key Insight: Config.json Matters for Serving

When `transformers 5.7.0` saves a merged model, it writes:
```json
{"model_type": "qwen3_5_moe_text", "architectures": ["Qwen3_5MoeForCausalLM"]}
```

But SGLang 0.5.9's `EntryClass` only registers `Qwen3_5MoeForConditionalGeneration` (the multimodal wrapper). The fix: copy the base model's original `config.json` into the merged model directory. The weights are identical architecture — same layers, same shapes — just with LoRA deltas baked in. The config just tells SGLang *how to load them*.

### Production Path (TODO)

The BF16 model on B200 works but wastes an expensive training GPU. The proper path:
1. Quantize merged model to FP8 → `models/qwen3.6-35b-a3b-mmm-fp8/` (~18 GB)
2. Serve on L40S (gpu partition, 48 GB) — frees B200 for training
3. Pre-build FlashInfer kernels for L40S (already cached from previous FP8 serving)

### Validation Response

The fine-tuned model immediately demonstrated MMM domain knowledge:
```
Prompt: "F2F plausibility at 70% and email attribution at 18%. Trust score is 33. What should we change?"

Response: Here's a thinking process:
1. Analyze User Input:
   - F2F plausibility: 70% (Face-to-Face interaction plausibility score)
   - Email attribution: 18% (Email channel attribution share)
   - Trust score: 33 (composite metric indicating model reliability/validity)
   ...
```

This is exactly the kind of structured, domain-aware reasoning we trained for.

---

## Evaluation: Qwen MMM vs Sonnet 4.6 (2026-05-06)

### Setup

- **Golden test dataset**: 27 prompts across 10 categories (diagnostics, config generation, optimization, gate failures, channel attribution, technical constraints, scenario planning, general MMM, Bayesian priors, debugging, methodology, production)
- **Models**: Sonnet 4.6 (via Uptimize Bedrock) vs Qwen3.6-35B-A3B-MMM (fine-tuned, via SGLang on B200)
- **Judge**: Opus 4.6 (blind evaluation, position-randomized to avoid order bias)
- **Dimensions**: Accuracy, Completeness, Evidence Quality, Domain Expertise (each /10, total /40)

### Results Summary

| Metric | Sonnet 4.6 | Qwen MMM |
|--------|-----------|----------|
| Wins | 16 (80%) | 4 (20%) |
| Avg Score | 30.0/40 | 25.9/40 |
| Accuracy | 7.6/10 | 6.8/10 |
| Completeness | 8.1/10 | 7.2/10 |
| Evidence | 6.5/10 | 5.3/10 |
| Expertise | 7.8/10 | 6.6/10 |

*Note: 7/27 tests lost to Qwen server disconnect (SSH tunnel dropped after test 21). Results based on 20 completed head-to-head comparisons.*

### Where Qwen Wins

Qwen beat Sonnet on **project-specific knowledge** — exactly what fine-tuning should teach:

| Test | Category | Qwen Score | Sonnet Score | Why Qwen Won |
|------|----------|-----------|-------------|--------------|
| version-01 | technical_constraints | 21 | 17 | Knew about pymc-marketing 0.18.0 vs 0.19.x (PR #2293) |
| version-02 | technical_constraints | 28 | 26 | Understood locked parameters and their specific rationale |
| channel-02 | channel_attribution | 28 | 21 | Correctly identified cross-sectional confounding in F2F correlation |
| gate-02 | gate_failures | 27 | 25 | Better interpretation of F2F yellow zone with project context |

### Where Sonnet Dominates

Sonnet won on reasoning quality, structure, and actionability:

- **Config generation** (3-0): Sonnet provides complete, executable code examples with staged implementation plans
- **General MMM knowledge** (3-0): More polished educational explanations
- **Optimization** (3-0): Better causal reasoning and diagnostic workflows
- **Scenario planning** (2-0): Stronger uncertainty quantification and practical recommendations

### Root Cause Analysis

The judge repeatedly noted the same pattern:

> *"Model [Qwen] is presented as raw thinking/planning that gets cut off... an unfinished outline rather than a complete response"*

> *"Model [Qwen] shows truncated thinking process, while Model [Sonnet] delivers polished, actionable response"*

**Diagnosis**: The GRPO training (with format-agnostic rewards) taught the model to *think well* but not to *present answers well*. The model generates in think-mode — using `<think>` blocks and planning structures — which embeds correct project-specific knowledge but wraps it in unpolished, stream-of-consciousness output.

### Improvement Roadmap

1. **More SFT data**: Current training set is only 12 examples (+ 55 augmented). Need 100+ diverse examples with polished assistant responses
2. **Response format training**: Add SFT examples that model well-structured output (## Approach / ## Reasoning / ## Config Changes) without think-mode rambling
3. **DPO on think-vs-answer pairs**: Use the evaluation results as DPO training data — Sonnet's polished answers as "chosen", Qwen's thinking dumps as "rejected"
4. **System prompt engineering**: The model may benefit from explicit "deliver a complete, actionable answer" instructions during inference
5. **Longer context**: Many Qwen responses hit the max_tokens (1500) limit — the think-mode overhead consumes tokens that should go to the answer

### Key Takeaway

Fine-tuning 35B params on 12 domain-specific examples produced a model that **knows project-specific facts** (pymc-marketing version bugs, locked parameters, confounding patterns) but **can't compete with Sonnet 4.6's reasoning and presentation quality**. The value of the fine-tuned model is as a **domain knowledge complement**, not a replacement. The optimal architecture may be RAG (retrieval-augmented Sonnet with project docs) rather than full fine-tuning.

## MiniMax M3 MXFP8 on 4× B200 — the symmetric_memory.rendezvous deadlock (2026-07-12)

### What happened

MiniMax M3 MXFP8 — the multimodal (vision+video) 427B MoE / 26B active coding+agent model — would not boot on 4× B200. Seven jobs deadlocked at NCCL init with an identical signature: workers reached `world_size=4 ... backend=nccl`, then froze — GPU mem pinned at ~1.1GB, utilization 0%, log mtime frozen, no `cuda_communicator.py:245` all-reduce-setup line. A prior boot (2026-06-22, job 2190313, same container, same args) had succeeded in 33s.

### The diagnostic that cracked it

`strace -c -p <worker>` showed ~350,000 `ioctl` + `sched_yield` calls/sec — a **pure userspace spin loop** (R-state, `wchan=0`, zero reads/writes). Not I/O, not a lock wait. Installing `py-spy` in the container and dumping the worker stack revealed the exact deadlocked call:

```
rendezvous  (torch/distributed/_symmetric_memory/__init__.py:1991)
_alloc_symm_buffer_bytes  (flashinfer/comm/torch_symmetric_memory.py:71)
trtllm_create_ipc_workspace_for_all_reduce_fusion  (flashinfer/comm/trtllm_ar.py:658)
__init__  (flashinfer/comm/allreduce.py:126)
create_allreduce_fusion_workspace  (flashinfer/comm/allreduce.py:413)
_create_workspace  (vllm/.../flashinfer_all_reduce.py:56)
get_fi_ar_workspace  (vllm/.../flashinfer_all_reduce.py:147)
_can_use_flashinfer  (vllm/.../fused_allreduce_gemma_rms_norm.py:90)
fused_allreduce_gemma_rms_norm  (...:120)
forward  (vllm/models/minimax_m3/nvidia/model.py:760)
```

**Root cause**: MiniMax M3's model code (`model.py:760`) calls `fused_allreduce_gemma_rms_norm` *directly in forward()* — a model-layer fused op, NOT a torch.compile pass. That op calls `_can_use_flashinfer` → `get_fi_ar_workspace` → `_create_workspace` → `trtllm_create_ipc_workspace_for_all_reduce_fusion` → `torch._symmetric_memory.rendezvous`, which hangs on B200 (SM100) non-deterministically. vLLM issue [#45800](https://github.com/vllm-project/vllm/issues/45800) reports the same hang; the reporter tried the wrong env-var name (`VLLM_FLASHINFER_ALLREDUCE`, "Unknown") and `fuse_allreduce_rms=False` (doesn't help — the model calls the op directly, bypassing the compilation-pass gate). Issue is open/unresolved upstream.

### What did NOT work (the 7-job elimination matrix)

| Attempt | Change | Result |
|---------|--------|--------|
| 1 | baseline TP=4+EP | deadlock |
| 2 | `--no-enable-flashinfer-autotune` (env var) | env var "Unknown" — deadlock |
| 3 | `--no-enable-flashinfer-autotune` (CLI flag, confirmed False) | deadlock |
| 4 | + `--enforce-eager` (cudagraph off) | deadlock |
| 5 | `VLLM_ALLREDUCE_USE_SYMM_MEM=0` (skips SymmMemCommunicator init) | progressed past first deadlock, hit second one in FlashInfer workspace |
| 6 | + `VLLM_ALLREDUCE_USE_FLASHINFER=0` + `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` | deadlock (trtllm still calls rendezvous) |
| 7 | `--optimization-level 1` (fuse_allreduce_rms=False at profile level) | deadlock (model calls op directly, not via pass) |

**Lesson**: env-var and compilation-config flags gate the *communicator-level* and *pass-level* paths. The MiniMax model calls the fused op at the *model layer*, which none of those gates touch. Source-reading + py-spy was the only way to find the actual call site.

### The fix (bind-mounted source patch)

Two vLLM source files extracted from the container and edited, then bind-mounted over the container's copies (so every process — API server + multiprocessing-spawned workers — loads the patched code):

1. `vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py` — `_can_use_flashinfer()` → `return (False, 0)`. The model takes the documented fallback: `tensor_model_parallel_all_reduce` + `GemmaRMSNorm`, "numerically identical to the unfused model path" per the vLLM docstring.
2. `vllm/distributed/device_communicators/flashinfer_all_reduce.py` — `get_fi_ar_workspace()` → `return None` after the docstring. Belt-and-suspenders: any other caller also skips `_create_workspace → rendezvous`.

Patched files live at `/shared/project/<account>/llm/tmp/vllm-patch/vllm/{...}` and are bind-mounted in `serving/serve-minimax-m3-container.sh`. With the patch, boot reaches `cuda_communicator.py:245 Using ['CUSTOM', 'PYNCCL']` (SYMM_MEM dropped), loads weights in ~30s, then compiles CUTLASS MSA sparse-attention kernels for SM100 (cold-cache AOT, ~3min), then serves. `/health` green. Vision (image_url) confirmed working.

### Result

- **4× B200, TP=4+EP, MXFP8, port 8105, 811K context, multimodal (vision+video) enabled.**
- Text + image inference verified; GPU util 73-99% under load.
- claude-minimax end-to-end verified: `claude → :5032 (logger) → :5014 (proxy) → :8105 (tunnel) → MiniMax`. Tool calls work (Bash tool executed `echo MINIMAX_E2E_OK`), per-request thinking works (`<mm:think>` blocks).
- Cost: the fused all-reduce+RMSNorm op is bypassed (falls back to separate all-reduce + RMSNorm). Throughput impact unmeasured but expected small on TP=4; the 06-22 success used the fused path — a future fix (upstream `symmetric_memory.rendezvous` B200 patch or a real env-var gate) can re-enable it.

### Why 2× B200 was impossible (also 2026-07-12)

Before pivoting to 4× B200, the 2× B200 goal was exhaustively ruled out:

- **NVFP4 (233GB, fits 2× B200 at 116GB/GPU)**: TP=2 deadlocks in the same MSA `rendezvous` (unmerged NVFP4 indexer support, vLLM roadmap #45668). Official vLLM recipe (`recipes.vllm.ai/MiniMaxAI/MiniMax-M3`) documents **TP=8** as the only tested B200 baseline for NVFP4. No TP=2 or TP=4 recipe exists.
- **MXFP8/FP8 (414GB)**: doesn't fit 2× B200 (207GB/GPU > 192GB).
- **Pipeline parallel (PP=2, TP=1)**: `NotImplementedError: Pipeline parallelism is not supported for this model` — MiniMax M3 doesn't implement the `SupportsPP` interface.

So 2× B200 is physically + engine-level impossible for MiniMax M3 in the current vLLM build. 4× B200 MXFP8 (the documented working config) is the minimum.

### Lessons (the real value)

1. **Research upstream before the second failing boot.** I burned 7 jobs re-deriving what the vLLM recipe + issue #45800 said in one fetch. The official recipe documents TP=8 as the tested B200 baseline; TP=2/4 are not supported configs. (See `feedback_research_online_when_stuck.md`.)
2. **`strace -c` + `py-spy dump` are the decisive diagnostics for "stuck process" bugs.** `ps` shows R-state but can't distinguish "real work" from "spin loop". strace's syscall-count breakdown (350K ioctl+sched_yield, zero reads) = pure spin. py-spy gives the exact Python line. Without these I was guessing at flags; with them I found the call site in minutes. (See `feedback_flashinfer_autotune_wedge.md` — same lesson as GLM-5.2: check process state before concluding I/O vs CPU-bound.)
3. **The Blackwell cold-boot wedge is a family, not a one-off.** GLM-5.2 FP8 (SGLang) hung in FlashInfer autotune; MiniMax M3 (vLLM) hangs in `symmetric_memory.rendezvous`. Both are B200/SM100 cold-cache kernel/comm-init deadlocks. The pre-flight rule: check cache warmth, disable autotune, don't cancel cold JIT < 15min. (Generalized in `feedback_flashinfer_autotune_wedge.md`.)
4. **Read the source to find the real env var name.** vLLM's naming is inconsistent: `VLLM_FLASHINFER_ALLREDUCE_BACKEND` (envs.py:196) vs `VLLM_ALLREDUCE_USE_FLASHINFER` (cuda_communicator.py:55) vs `VLLM_ALLREDUCE_USE_SYMM_MEM` (envs.py:235). Issue #45800's reporter tried `VLLM_FLASHINFER_ALLREDUCE` (wrong, "Unknown") and gave up. Source-reading `cuda_communicator.py:55` and `envs.py` found the real names.
5. **When flags don't gate the buggy path, patch the source.** Every config flag (`-O 1`, `--enforce-eager`, env vars) gates paths the MiniMax model doesn't use — it calls the fused op directly in `forward()`. A bind-mounted source patch is the surgical fix at the actual call site.

---

## V4-Flash-0731 + DSpark — 256 tok/s on 4× B200 (2026-08-05)

**The win:** DeepSeek V4-Flash-0731 hits **256 tok/s on 4× B200** with DSpark speculation — matching/beating GLM-5.2 FP8 on 8× B200 (~220-250 tok/s), on half the hardware. 3.1× the 83 tok/s baseline.

### The 11-attempt journey

| # | Config | tok/s | Outcome |
|---|---|---|---|
| 1 | EAGLE 3-token, sglang-dsv4-blackwell.sif 0.5.10rc0 | crash | wrong container (no DFLASH/DSpark) |
| 2 | DFLASH, sglang-latest.sif 0.5.13.post1 | crash | DFLASH not allowed for DeepSeekV4 |
| 3 | DSPARK, sglang-latest.sif 0.5.13.post1 | crash | DSPARK not in allowed list (needs 0.5.16+) |
| 4 | EAGLE 3-token + mem 0.85 | 83 | baseline |
| 5 | EAGLE 5-token + FP4 indexer | 82 | no improvement |
| 6 | EAGLE 7-token + cuda-graph-max-bs 4 | 76 | worse (acceptance drops with more tokens) |
| 7 | EAGLE 2-token + cuda-graph-max-bs 1 | 94 | slightly better |
| 8 | No speculation + cuda-graph-max-bs 1 + mem 0.92 | 119 | best non-DSpark config |
| 9 | `deep_gemm` MoE backend (no spec) | crash | TVM kernel bug `SiluAndMulMaskedPostQuantKernel` |
| 10 | `--enable-multi-layer-eagle` | crash | `draft_model_idx` TypeError |
| **11** | **DSpark gamma=5, sglang-v0516.sif (SGLang 0.5.16)** | **256** | **✅ goal met** |

### The breakthrough

DSpark is DeepSeek's native speculation algorithm (arXiv 2606.19348, PR #30261 in SGLang, released July 2026 in v0.5.16). It uses layers 40-42 of the same checkpoint as the drafter — no separate draft model, no extra VRAM. The hard requirement: **SGLang 0.5.16+** (`sglang-v0516.sif`).

Our `sglang-latest.sif` was SGLang 0.5.13.post1, which hard-asserts EAGLE only for DeepSeekV4 via the `apply_deepseek_v4_defaults` hook. DSpark/DFLASH are rejected.

**The mistake I made:** When DSpark failed on 0.5.13.post1, I gave up on it and spent 6 boot cycles tuning EAGLE — none exceeded 119 tok/s. The user said "Why didn't we do DSpark yet?" — and the fix was simply to pull a newer SGLang container. DSpark worked immediately, hit 256 tok/s. Lesson saved to `feedback_dspark_not_eagle.md`.

### Final config

- Container: `sglang-v0516.sif` (SGLang 0.5.16-cu130 from `lmsysorg/sglang:v0.5.16-cu130`)
- TP=4, `flashinfer_mxfp4` MoE backend, `dsv4` attention
- `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` (gamma=5)
- `--enable-deepseek-v4-fp4-indexer`
- `--cuda-graph-max-bs 192` (lmsys blog value)
- `--mem-fraction-static 0.90`, `--kv-cache-dtype fp8_e4m3`
- `--chunked-prefill-size 4096`, `--disable-flashinfer-autotune`
- FlashInfer + Triton caches on disk (not tmpfs — host RAM OOMs during JIT)

### Container pull gotchas

- 8GB RAM Slurm job OOMs during mksquashfs — needs `--mem=64G --cpus-per-task=8`
- `/tmp` ran out of space during layer unpacking — set `APPTAINER_TMPDIR=<project>/.cache/apptainer-tmp`
- Pull takes ~15 min on cpu partition
- Recipe: `serving/pull-sglang-v0516.sh`

### Known issues to watch

- **PR #33614 (OPEN)**: DSpark TP rank divergence — sampling decisions made independently per rank, may deadlock at TP>1 (we're TP=4). Watch for deadlocks at long context.
- **Issue #33549**: decode hang at ~245K context with DSpark + TP=8. TP=4 less affected, but watch past 200K.
- **Issue #33493**: DSpark looks up `acc_linear_penalities` (typo) instead of `acc_additive_penalties`. Breaks `min_new_tokens` and additive penalties.

### Full deployment doc

`docs/ops/v4flash-0731-dspark-deployment.md` — complete reference with container pull recipe, all flags explained, iteration history, known issues, and source links.
