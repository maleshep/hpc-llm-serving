# HPC Inventory — What Lives on the Cluster

> **A manifest of the files, containers, venvs, and model weights that live on the Slurm cluster, and what should be there.** Small text files are tracked in git; binaries (containers, venvs, model weights) are recorded as pointers with size and regeneration instructions — never copied into the repo.

This is the **inventory chapter** of the playbook. It answers two questions:
1. **What should be on the cluster?** (source of truth — this file + the `serving/` scripts)
2. **What's actually there?** (drift detection — verified by inspection)

Like the rest of this repo, it is **sanitized**: real account names, node names,
and user IDs are replaced with `<account>`, `<fat-node>`, `<muid>`. The
structure is real; the values are generic.

---

## Path Conventions

| Placeholder | Meaning |
|-------------|---------|
| `$LLM_BASE` | `/shared/project/<account>/llm` — project storage root for the fleet |
| `$SCRATCH` | `/shared/scratch/<muid>` — scratch storage (large models, no snapshots) |
| `<fat-node>` | A fat partition node with 8× B200 (192 GB each) |
| `<gpu-node>` | A gpu partition node with 4× L40S (48 GB each) |

**Two storage tiers and why they matter:**

| Tier | Path | Quota | Snapshots | Use for |
|------|------|-------|-----------|---------|
| **Project** | `/shared/project/<account>/` | ~1 TB | Daily (2 weeks) | Code, scripts, configs, small files, venvs |
| **Scratch** | `/shared/scratch/<muid>/` | ~4 TB | **None** | Model weights, large intermediates |

Model weights (hundreds of GB) always live on scratch — they'd blow the project
quota. Small configs and serve scripts live in project storage and are
**tracked in git** (this repo). The inventory below records which is which.

---

## Directory Layout (canonical)

```
$LLM_BASE/
├── README.md
├── containers/              # Apptainer SIF files (binaries — pointers only)
├── logs/                     # Slurm .out/.err per job
├── models/                   # Symlinks or small models (project-quota-safe)
├── scripts/                  # Training, merge, data extraction
├── serving/                  # serve-*.sh + download-*.sh + pull-*.sh
├── .fleet/                   # Fleet monitor state (state.json + events.log)
├── .serve-state-*.json       # Per-model runtime state (node, port, status)
├── tmp/                      # Triton/torch-ext caches (persistent, shared)
├── training/                 # data/ + output/ (adapters)
├── training-venv/            # PyTorch + Transformers + TRL (training)
├── venv/                     # SGLang pip (GLM / Gemma / Qwen3)
├── vllm-venv/                # vLLM 0.21 (Kimi)
└── vllm-native-venv/         # vLLM native (experiments)
```

---

## 1. Serve Scripts (`serving/`)

**Source of truth:** this repo's `serving/` directory. **Deploy:** rsync to
`$LLM_BASE/serving/`. These are small bash files and ARE tracked in git.

### Production serve scripts

| Script | Model | Hardware | Port | QoS | Notes |
|--------|-------|----------|------|-----|-------|
| `serve-glm52-sglang-latest.sh` | GLM-5.2 FP8 | 8× B200 | 8103 | 7d | Primary coding, 1M ctx, EAGLE, TP=8 |
| `serve-glm52-nvfp4.sh` | GLM-5.2 NVFP4 | 4× B200 | 8106 | 3d | Alt, 512K ctx, half-hw |
| `serve-glm52-nvfp4-warm.sh` | GLM-5.2 NVFP4 | 4× B200 | 8107 | 3d | Keep-warm fallback (hot-swap tier) |
| `serve-minimax-m3-container.sh` | MiniMax M3 MXFP8 | 4× B200 | 8105 | 3d | Multimodal, 811K ctx, vLLM container |
| `serve-gemma4-b200.sh` | Gemma-4-31B-IT | 2× B200 | 8105 | 1d | Fast coding, FROZEN_KV_MTP |
| `serve-gemma4-l40s.sh` | Gemma-4-31B-MMM-SFT | 2× L40S | 8200 | 1d | MMM domain agent |
| `serve-kimi-k27.sh` | Kimi K2.7-Code | 4× B200 | 8104 | 1d | TOKENSPEED_MLA, vLLM |
| `serve-v4pro-container.sh` | DeepSeek-V4-Pro | 8× B200 | 8101 | 1d | Heavy coding, 1M ctx |
| `serve-qwen3-235b.sh` | Qwen3-235B-A22B | 4× B200 | 8100 | 1d | Backup coding |

### Download / pull scripts (regenerate weights + containers)

| Script | What it fetches |
|--------|-----------------|
| `download-glm51.sh`, `pull-glm52-vllm.sh` | GLM model weights from HuggingFace |
| `download-kimi-k2.sh`, `download-qwen3-235b.sh` | Other model weights |
| `download-gemma4-mtp.sh` | Gemma MTP drafter (0.94 GB) |
| `download-nemotron-ultra.sh` | Nemotron weights |
| `pull-container.sh`, `pull-sglang-latest.sh` | Apptainer SIF containers |
| `pull-minimax-m3-container.sh`, `pull-sglang-minimax-m3.sh` | Model-specific containers |

> **Pruning note:** experiment scripts accumulate (`serve-glm52-reap-*.sh`,
> `*.bak.*`, `serve-glm52-*-test.sh`). These are not production. Periodically
> archive or delete to keep `serving/` navigable. The `.bak.*` files are
> checkpoints from edits — safe to remove once the canonical script is stable.

---

## 2. Runtime State Files (`.serve-state-*.json`)

**Not in git** — these are runtime artifacts written by serve scripts and read
by `proxy-ai check` for tunnel healing. Each records the current job's node,
port, status, and tunnel command. If a state file is stale (points at a dead
node), `proxy-ai check` skips healing it.

| State file | Model | Owner script |
|------------|-------|--------------|
| `.serve-state-glm.json` | GLM-5.2 FP8 (primary) | `serve-glm52-sglang-latest.sh` |
| `.serve-state-glm-nvfp4.json` | GLM-5.2 NVFP4 (alt) | `serve-glm52-nvfp4.sh` |
| `.serve-state-glm-nvfp4-warm.json` | GLM-5.2 NVFP4 (keep-warm) | `serve-glm52-nvfp4-warm.sh` |
| `.serve-state-minimax.json` | MiniMax M3 | `serve-minimax-m3-container.sh` |
| `.serve-state-gemma.json` | Gemma-4 MMM (L40S) | `serve-gemma4-l40s.sh` |
| `.serve-state-gemma-b200.json` | Gemma-4 base (B200) | `serve-gemma4-b200.sh` |
| `.serve-state-kimi.json` | Kimi K2.7 | `serve-kimi-k27.sh` |
| `.serve-state-pro.json` | DeepSeek V4-Pro | `serve-v4pro-container.sh` |
| `.serve-state.json` | DeepSeek V4-Flash | (legacy) |

---

## 3. Fleet Monitor State (`.fleet/`)

**Not in git** — written by `fleet-monitor.py` (the observer daemon running in
the mmm-serve cpu job). Read by `proxy-ai watch` to render the FLEET panel.

| File | Purpose |
|------|---------|
| `.fleet/state.json` | Current snapshot: daemon state, current GLM job, free nodes, 7d budget, cadence |
| `.fleet/events.log` | Append-only ALERT log: opportunities, expiry warnings, gaps |

**Source of truth for the monitor:** `model_training` repo → `serving/fleet-monitor.py`.
**Deploy:** rsync to `$LLM_BASE/serving/fleet-monitor.py`. Launched by
`marketing-mix` repo's `hpc/serve.sh` as a background loop inside the mmm-serve
cpu job (laptop-independent, zero B200 budget).

---

## 4. Containers (`containers/`)

**Binaries — never copied to git.** Recorded here with size + how to rebuild.

| SIF | Size | How to (re)generate | Used by |
|-----|------|----------------------|---------| 
| `sglang-latest.sif` | 11 GB | `pull-sglang-latest.sh` (apptainer pull from DockerHub) | GLM-5.2 FP8, NVFP4, NVFP4-warm |
| `sglang-dsv4-blackwell.sif` | 20 GB | `pull-container.sh` (~42 min pull) | DeepSeek V4-Pro, V4-Flash |
| `vllm-glm52.sif` | 11 GB | `pull-glm52-vllm.sh` | GLM vLLM experiments |
| `vllm-minimax-m3.sif` | 7.4 GB | `pull-sglang-minimax-m3.sh` | MiniMax M3 |
| `sglang-v0512.sif` | 11 GB | (legacy) | Older GLM-5.1 — superseded by `sglang-latest.sif` |

> Total container storage: ~58 GB. Persistent caches (Triton, torch-extensions)
> live in `$LLM_BASE/tmp/` — shared across jobs to avoid recompilation.

---

## 5. Model Weights (on `$SCRATCH/models/`)

**Never copied to git** — too large. Recorded as a manifest: path, size, source,
quantization. All live on scratch (no snapshots — back up the HF source, not
the local copy).

| Model dir | Size | Quantization | Source | Use |
|-----------|------|--------------|--------|-----|
| `glm-5.2-fp8/` | 704 GB | FP8 | `zai-org/GLM-5.2-FP8` | Primary coding (8× B200) |
| `glm-5.2-nvfp4/` | 433 GB | NVFP4 | `nvidia/GLM-5.2-NVFP4` | Alt + keep-warm (4× B200) |
| `glm-5.2-w4afp8/` | 332 GB | W4AFP8 | `PhalaCloud/GLM-5.2-W4AFP8` | Backup (buggy on B200 — skip) |
| `glm-5.2-reap-504b/` | 300 GB | NVFP4 pruned | `0xSero/GLM-5.2-504B` | 1M-ctx experiment (wedged on B200) |
| `glm-5.2-fp8-dflash/` | 7.0 GB | — | experiment | DeepFlash test |
| `glm-5.1-fp8/` | 705 GB | FP8 | `zai-org/GLM-5.1-FP8` | Legacy (superseded by 5.2) |
| `kimi-k2.7-code/` | 555 GB | compressed-tensors | `moonshotai/Kimi-K2.7-Code` | Alt coding |
| `kimi-k2.6-nvfp4/` | 555 GB | NVFP4 | scratch | Legacy Kimi |
| `minimax-m3-mxfp8/` | 414 GB | MXFP8 | `MiniMaxAI/MiniMax-M3-MXFP8` | Multimodal |
| `deepseek-v4-pro/` | 806 GB | — | DeepSeek | Heavy coding |
| `nemotron-3-ultra-nvfp4/` | 329 GB | NVFP4 | Nemotron | Backup |
| `cosyvoice3-0.5b/` | 4.6 GB | — | TTS | Speech |
| `qwen2-audio-7b/` | 16 GB | — | Qwen | Audio |

> Small models (Gemma, Qwen3, ASR/TTS drafters) live in `$LLM_BASE/models/`
> (project storage) — they fit the quota. See the project `models/` dir for:
> `gemma-4-31b-it` (59 GB), `gemma-4-31b-mmm-sft` (58 GB),
> `gemma-4-31b-it-assistant` (0.94 GB MTP drafter), `qwen3-asr-1.7b`,
> `qwen3-tts-1.7b`.

---

## 6. Python Environments (venvs)

**Binaries — never copied to git.** Recorded with size + how to rebuild.

| Venv | Size | Stack | Used for |
|------|------|-------|---------|
| `venv/` | 16 GB | SGLang 0.5.12 pip | GLM, Gemma, Qwen3 serving |
| `vllm-venv/` | 8.0 GB | vLLM 0.21 | Kimi K2.7 serving |
| `vllm-native-venv/` | 8.9 GB | vLLM native | Experiments |
| `training-venv/` | 7.2 GB | PyTorch + Transformers + TRL 1.3.0 | SFT/GRPO/DPO training |
| `venv-asr/` | 7.7 GB | — | ASR (Qwen) |
| `venv-tts/` | 7.6 GB | — | TTS (CosyVoice/Kokoro) |
| `venv-home/` | 4.9 GB | — | Home/general |

> **Rebuild pattern:** each venv is recreated via `uv pip install` from a
> requirements file (or inline in the serve script). The persistent
> `tmp/torch-extensions` and `tmp/triton-cache` dirs are shared across venvs
> to avoid recompiling custom CUDA kernels on every job.

---

## 7. Operational Tooling (not in this repo)

These tools are the *clients* of the fleet — they run on the laptop or as
separate HPC jobs, not inside `hpc-llm-serving`. Recorded here so the
inventory is complete.

| Tool | Where | Purpose |
|------|-------|---------|
| `proxy-ai` | Laptop (`~/.local/bin/proxy-ai.cmd`) | Tunnel + proxy orchestrator, `watch` shows fleet panel |
| `claude-code-proxy` | Laptop (`~/repo/claude-code-proxy/`) | Anthropic↔OpenAI API translation |
| `/hot-swap` skill | Laptop (`~/.claude/skills/hot-swap/`) | Intelligent hot-swap trigger (propose → go → execute) |
| `fleet-monitor.py` | HPC (in mmm-serve cpu job) | Observer daemon, writes `.fleet/state.json` |
| `mmm-serve` job | HPC (cpu partition, 7d) | Web-app serving + launches fleet monitor |
| `switch-model.sh` | HPC (`$LLM_BASE/serving/`) | Model switcher (run on login node) |
| `llm-cost-dashboard` | HPC | 3-route token/cost dashboard (port 4400) |
| GLM resubmit cron | HPC | Every-45-min auto-resubmit if GLM dead (never cancel) |

---

## Drift Detection — How to Verify

To check what's actually on HPC vs this manifest:

```bash
# Serve scripts present?
ssh <login> "ls $LLM_BASE/serving/serve-*.sh"

# Containers + sizes?
ssh <login> "ls -lh $LLM_BASE/containers/"

# Model weights + sizes (scratch)?
ssh <login> "du -sh $SCRATCH/models/*/"

# Venvs?
ssh <login> "du -sh $LLM_BASE/*venv*"

# Fleet monitor deployed?
ssh <login> "ls $LLM_BASE/.fleet/ $LLM_BASE/serving/fleet-monitor.py"

# State files (which models have live state)?
ssh <login> "ls $LLM_BASE/.serve-state-*.json"
```

A healthy cluster matches this manifest. Missing serve scripts or stale
state files (pointing at dead nodes) are the most common drift.

---

## Last Verified

| Section | Date | Status |
|---------|------|--------|
| Serve scripts | 2026-07-12 | Verified — production scripts present, experiment `.bak` files accumulating (prune candidate) |
| Containers | 2026-07-12 | Verified — 5 SIFs, ~58 GB total |
| Model weights | 2026-07-12 | Verified — 13 dirs on scratch, ~4.7 TB total |
| Venvs | 2026-07-12 | Verified — 7 venvs, ~60 GB total |
| Fleet monitor | 2026-07-12 | **Deployed** — running in mmm-serve job 2474026 (cpu partition). `fleet-monitor.py` + `fleet_panel.py` at `$LLM_BASE/serving/`. Panel verified live in `proxy-ai check`. |
| State files | 2026-07-12 | Verified — 14 state files (some stale from experiments) |
