# Multimodal Stack Architecture

**Status**: Design doc, 2026-08-04. Living document — update as pieces land.
**Scope**: How speech (TTS + ASR), video generation, LLM serving, and Claude Code tooling connect on HPC cluster + Windows client.

---

## 1. Goals

1. **Pluggable speech stack** — swap TTS/ASR backends via config, no client code changes.
2. **Closed-loop eval** — every model is benchmarked against peers before being flipped to default. No "ship the new shiny" without numbers.
3. **Multi-modal MMM pipeline** — MMM agent outputs a marketing recommendation; downstream renders it as a narrated video pitch (north star, not yet committed to GPU-hours).
4. **Zero Anthropic traffic** — all inference routes through HPC models or the company enterprise-LLM-gateway/Bedrock gateway (`claude-opus`). Telemetry blocked.
5. **Survive reimage** — repo-tracked skills (`/hpc`, `/hpc-code`), recipes, and architecture; untracked but rebuildable `~/.local/bin/` launchers.

---

## 2. Component map

```
Windows client (laptop, post-reimage)
-------------------------------------
  Claude Code (claude-glm, claude-kimi, claude-opus)
       |
       | SSH tunnels (proxy-ai managed for claude-*, manual for open_wispr)
       v
HPC cluster (the company Munich cluster, account hpc-llm)
--------------------------------------------------
  Fat partition (8x B200 192GB per node):
    - GLM-5.2 FP8 coding primary       (port 8103, sglang container)
    - V4-Flash-0731 fast coding        (port 8100, sglang container, DSpark+MXFP4)
    - MiniMax M3 long ctx + vision     (port 8105, vllm container)
    - Kimi K3 / K2.7 alt coding        (port 8102/8104, vllm)
    - Gemma-4-MMM-SFT domain agent     (port 8200, native transformers)
    - [PROPOSED] MiniMax H3 video gen  (port 8315, sglang omni, 4x B200, ulysses=4)

  GPU partition (1x L40S 48GB per node):
    - VibeVoice-ASR-9B transcription   (port 8300, vllm 0.14.1 container)
    - [NEW] Audio8-TTS-Preview-0.6b     (port 8310, sglang omni container)
    - open_wispr unified TTS+ASR        (port 8200, faster-whisper + f5-tts active)
```

The HPC has **two partitions** that matter for this stack: `fat` (8x B200, for big LLMs and the future video gen) and `gpu` (1x L40S, for the speech stack — small models, cheap boot, interactive use).

---

## 3. The three serving tiers and how they differ

| Tier | Purpose | Boot cost | Pattern | Hot-swap? | Fleet dashboard tag |
|---|---|---|---|---|---|
| **Claude Code fleet** (fat partition) | Coding/reasoning backends behind `claude-code-proxy` | 5-15 min | `proxy-ai` + per-model settings JSON | Yes (per-port) | `GLM`, `V4F`, `M3`, `K3`, etc. |
| **Standalone eval engines** (gpu partition) | Single-model TTS/ASR endpoints, OpenAI-compatible | <2 min | Direct `sbatch`, no proxy | No (one per port) | `VVASR` (VibeVoice), `A8TTS` (Audio8, proposed) |
| **open_wispr unified stack** (gpu partition) | Pluggable ASR+TTS with hot-swap config | <1 min | FastAPI server, `model_registry.yaml` | Yes (runtime endpoint) | `OW` (proposed) |

The three tiers serve different needs:
- **Claude Code fleet** — interactive chat/reasoning behind the Anthropic Messages API. Latency-critical, always-on, expensive to boot, managed by `proxy-ai`.
- **Standalone eval engines** — single-task models (transcribe, synthesize) that don't need the Anthropic translation layer. They speak OpenAI directly. Used for evaluation AND for production single-task jobs (e.g., transcribing a 60-min meeting recording).
- **open_wispr unified** — the interactive speech layer for the Windows hotkey client. Pluggable backends so we can A/B test new TTS/ASR without disrupting the client.

---

## 4. Data flow — six concrete paths

### Path A: Claude Code on GLM-5.2 (existing, primary)

```
User types in Claude Code
  -> claude-glm launcher
    -> proxy-ai glm (opens SSH tunnel, starts claude-code-proxy on :5007)
      -> Claude Code sends Anthropic Messages API to localhost:5007
        -> claude-code-proxy translates to OpenAI chat/completions
          -> SSH tunnel to HPC node:8103 (GLM-5.2 SGLang container)
            -> GLM-5.2 generates response
          <- OpenAI response back through tunnel
        <- proxy translates back to Anthropic shape
      <- Claude Code renders response
```

**No speech, no video. Pure text coding.**

### Path B: Windows hotkey voice-to-text (existing, open_wispr)

```
User holds hotkey, speaks
  -> open_wispr.pyw Windows client captures mic audio
    -> POST to localhost:8200/transcribe (SSH-tunneled to HPC open_wispr server)
      -> open_wispr server: faster-whisper ASR backend transcribes
    <- text returned to client
  -> Client pastes text into active window (keyboard simulation)
```

**TTS side unused on this path. ASR is faster-whisper (active default).**

### Path C: Voice cloning demo (existing, open_wispr + CosyVoice fallback)

```
User uploads reference clip + text via open_wispr web UI
  -> POST localhost:8200/clone (form: reference, text)
    -> open_wispr server: TTSBackend.clone() dispatches to active backend (f5-tts)
      -> f5-tts generates WAV
  <- WAV returned, played in browser
```

### Path D: Long-form meeting transcription (existing, VibeVoice-ASR standalone)

```
User has a 60-min recording on HPC
  -> sbatch serving/serve-vibevoice-asr.sh (or already running)
  -> curl POST localhost:8300/v1/chat/completions with audio data URI
    -> VibeVoice-ASR-9B transcribes (single-pass, no chunking)
  <- Full transcript with word timestamps + diarization
```

**Why not use open_wispr's faster-whisper for this?** Faster-whisper chunks at 30s; VibeVoice handles 60+ min single-pass with better diarization. Standalone endpoint = no open_wispr server contention.

### Path E [NEW]: Audio8-TTS eval (proposed, standalone)

```
bench_tts.py runs locally
  -> For each prompt:
    -> Hit Audio8 SGLang Omni :8310/v1/audio/speech (and open_wispr :8200/synthesize for f5-tts/cosyvoice)
      -> WAV bytes returned
    -> Send WAV to VibeVoice-ASR :8300 for transcription
    -> Compute WER(transcript, original_prompt)
  -> Aggregate: WER mean/median, RTF (real-time factor), synth latency
  -> Decision gate: if Audio8 wins on WER AND RTF < 1.0, flip open_wispr registry to active: audio8
```

**This is the closed loop**: Audio8's output is judged by VibeVoice-ASR (which we trust because it's already the long-form transcription reference). No human-in-the-loop subjective scoring needed for the gate — though spot-listen checks remain good practice.

**Peer-verification status (2026-08-04)**: ✅ **Audio8 confirmed as the right pick.** The agent surveyed 16 TTS peers (F5-TTS, Kokoro-82M, Parler-TTS Large v1, Sesame CSM-1b, XTTS-v2, Fish S2 Pro, Higgs Audio v2, CosyVoice2/3, MOSS-TTS, VoxCPM2, IndexTTS2, Spark-TTS, MaskGCT, MegaTTS3, Supertone supertonic-3, SwanTale). Verdict: Audio8 is the only model that clears all six hard constraints (Apache 2.0, 1× L40S fit, 11+ languages incl. EN/DE/FR/IT/ES/JP/KR/ZH, zero-shot voice cloning, SGLang or vLLM serving, no custom runtime).

**🚨 F5-TTS LICENSE BLOCKER**: F5-TTS — the current active TTS in `open_wispr/model_registry.yaml` — is **CC-BY-NC-4.0 (non-commercial only)**. For the company corporate use, that's a compliance issue. Audio8 (Apache 2.0) strictly dominates F5-TTS on WER (1.506 vs 2.24), language coverage, AND license. **Action**: prioritize the Audio8 eval and flip `active: audio8` as soon as bench passes; demote F5-TTS to legacy/backup only (do not use as default).

**Monitoring**: Qwen3-TTS demo Space appeared 2026-06-09. Weights not yet released but Qwen org has an Apache-2.0 track record + Qwen3 backbone. Could be a serious challenger once released — worth a recheck in 3-6 months.

**Note on VibeVoice-Realtime-0.5B**: Microsoft publishes a separate `VibeVoice-Realtime-0.5B` TTS model (~300ms TTFA, true streaming, English-only) that is complementary to Audio8 — not competitive (real-time conversational English vs multilingual cloning TTS). We have NOT deployed it (only `VibeVoice-ASR-9B` for transcription is in the repo at `serve-vibevoice-asr.sh`). If real-time English TTS becomes a need (e.g., for the open_wispr hotkey client to feel instantaneous), that's a future addition on a separate port.

### Path F [NORTH STAR, not yet built]: MMM agent -> video pitch

```
User asks Gemma-4-MMM-SFT for a budget recommendation
  -> Gemma returns: "Shift 30% to digital channels because..."
    -> Text recommendation captured
  -> Post to MiniMax H3 endpoint (proposed :8315)
    -> H3 generates 4-15s video with native stereo audio narration
  <- Video pitch returned
```

**Why H3 over a TTS+video pipeline?** Native audio means the voice is synchronized to lip movements / scene cuts automatically — no post-process stitching. The 33B dense + Qwen3-VL-32B encoder fits one fat node.

**Peer-verification status (2026-08-04)**: H3 is **NOT** the only open-weight model with native audio in 2026. Five serious competitors exist:
- **MOVA** (`OpenMOSS-Team/MOVA-720p`) — Apache 2.0, 32B MoE / 18B active, 720p, SGLang-tagged
- **NAVA** (`baidu/NAVA`) — Apache 2.0, 6.3B, smaller/faster
- **daVinci-MagiHuman** (`SandAI-org/daVinci-MagiHuman`) — Apache 2.0, 15B, single H100, portrait-focused
- **LTX-2.3** (`Lightricks/LTX-2.3`) — 22B, $10M revenue cap license (NOT Apache)
- **JavisDiT++** — joint audio-video DiT (paper-level, availability varies)

**🚨 LICENSE BLOCKER for H3**: The literal MiniMax H3 community-license **text excludes the United States** as a permitted territory. MiniMax's Q&A document contradicts this and says US is allowed. This conflict must be resolved in writing by the company legal (or directly with MiniMax) before any GPU-hours are committed. **Do not deploy H3 without legal sign-off.**

**Drop-in fallback if H3 license is blocked**: MOVA (Apache 2.0, legally unambiguous, fits 4× B200, SGLang-native). Loses Ref2VA multi-reference conditioning (the killer feature for marketing pitches conditioned on brand assets) and runs at 720p instead of 2K. Worth a parallel eval if H3 stalls.

---

## 5. Port allocation (cluster-wide)

| Port | Model | Partition | Tier | Status |
|---|---|---|---|---|
| 8100 | V4-Flash-0731 | fat | Claude Code fleet | Live |
| 8101 | V4-Pro | fat | Claude Code fleet | Backup |
| 8102 | Kimi K3 | fat | Claude Code fleet | Live |
| 8103 | GLM-5.2 FP8 | fat | Claude Code fleet | Primary |
| 8104 | Kimi K2.7 | fat | Claude Code fleet | Live |
| 8105 | MiniMax M3 MXFP8 | fat | Claude Code fleet | Live |
| 8106 | GLM-5.2 NVFP4 | fat | Claude Code fleet | Backup |
| 8109 | GLM-5.2-REAP-504B | fat | Claude Code fleet | Experimental |
| 8110 | Inkling NVFP4 | fat | Claude Code fleet | Experimental |
| 8113 | GLM-old alias | fat | Claude Code fleet | Hot-swap alias |
| 8116 | GLM-5.2-Vision | fat | Claude Code fleet | Live |
| 8200 | open_wispr unified | gpu | open_wispr | Live |
| 8300 | VibeVoice-ASR-9B | gpu | Standalone eval | Live |
| 8310 | Audio8-TTS-Preview-0.6b | gpu | Standalone eval | **Proposed** |
| 8315 | MiniMax H3 | fat | Standalone eval | **Proposed (north star)** |

Proxy ports on the Windows client (5007-5037) map 1:1 to backend ports via SSH tunnels — see `serving/fleet_watch_local.py:LOCAL_PORTS`.

---

## 6. Decision gates and hot-swap points

### Speech stack (open_wispr)

**Active**: `f5-tts` (TTS), `faster-whisper` (ASR).
**Eval candidates**: `audio8` (TTS, standalone on :8310).

**Gate criteria** (must hit ALL to flip `model_registry.yaml: active`):
- WER mean within 10% of current active backend's WER (or better)
- RTF (real-time factor) < 1.0 on L40S (else too slow for interactive hotkey)
- Voice cloning SIM (speaker embedding cosine) within 5% of active
- License is Apache 2.0 or equivalent permissive (the company corporate)

**Swap mechanism**: `POST /admin/swap-model` with `service: tts, backend_name: audio8` (already implemented in `open_wispr/server.py`). No restart needed.

### Claude Code fleet

**Active**: GLM-5.2 FP8 (primary, port 8103).
**Hot-swap**: `proxy-ai` restart on a different port, or `switch-model.sh` for shared-port swaps. See `/hpc` skill Step 2.

**Gate criteria for adding a new coding model**:
- Terminal-Bench or DeepSWE score within 5% of GLM-5.2 (or better)
- Throughput > 100 tok/s on 8x B200 (or 4x B200 if it's a half-hw variant)
- Context > 128K (GLM-5.2 has 1M)
- Boot cost < 20 min (else needs `--qos=3d`)

### Video gen (north star)

**Active**: none.
**Primary candidate**: MiniMax H3 (peer-verified 2026-08-04 as technically best, license-blocked pending legal).
**Drop-in fallback**: MOVA (`OpenMOSS-Team/MOVA-720p`, Apache 2.0, 32B MoE / 18B active, 720p, SGLang-native, fits 4× B200). Loses Ref2VA multi-reference conditioning and 2K resolution.
**Other Apache-2.0 alternatives verified**: NAVA (`baidu/NAVA`, 6.3B, smaller/faster), daVinci-MagiHuman (`SandAI-org/daVinci-MagiHuman`, 15B, single H100, portrait-focused).

**Gate criteria for committing GPU-hours** (all must hold):
- Concrete pharma marketing use case (not "we might want this someday")
- 🚨 **License unambiguous in writing** — H3's literal LICENSE text excludes the US; Q&A doc contradicts. the company legal must confirm in writing before deployment. MOVA/NAVA/daVinci bypass this issue entirely (Apache 2.0).
- Fits on 8x B200 fat node without OOMing the coding fleet
- Native audio (the whole point — otherwise use a separate TTS + video pipeline)
- For H3 specifically: Ref2VA multi-reference conditioning matters for brand-asset-conditioned pitches. If we can't use H3, MOVA loses this — re-evaluate whether the use case still works without it.

---

## 7. What is NOT in scope (and why)

- **Real-time voice-to-voice conversational agent** (Sesame CSM-style). Latency budget on HPC + SSH tunnel + Windows client is too high for true duplex conversation. The open_wispr client is push-to-talk, not duplex. If duplex becomes a need, that's a separate architecture.
- **Live video streaming** (e.g., real-time avatar rendering). H3 generates 4-15s clips offline; live streaming would need a different model (probably API-only).
- **Open-sourcing open_wispr**. The `open_wispr` name was chosen to suggest open-source-ability, but the immediate goal is internal use. Open-sourcing later would require legal review of the bundled model dependencies (CosyVoice, F5-TTS licenses).
- **Replacing claude-code-proxy with a different translation layer**. The proxy at `~/repo/claude-code-proxy/server.py` is load-bearing for the entire Claude Code fleet. Any replacement would need to handle the reasoning_effort per-model quirks, max_tokens capping, context safety net, and Gemma-4 schema simplification. Out of scope for this architecture.

---

## 8. Implementation order (concrete, sequenced)

| # | Task | Owner | Blocks | Status |
|---|---|---|---|---|
| 1 | Rename `wisprflow/` -> `open_wispr/` | This repo | #2-#6 | Done 2026-08-04 |
| 2 | Verify Audio8 is the right 2026 TTS pick vs peers | Subagent | #4, #5 | ✅ Done 2026-08-04 — Audio8 confirmed |
| 3 | Verify MiniMax H3 is the right video-gen pick vs peers | Subagent | #8 | ✅ Done 2026-08-04 — H3 best technically, license blocked |
| 4 | Write `serving/download-audio8-tts.sh` + `serve-audio8-tts.sh` | This repo | #5 | Done 2026-08-04 (draft) |
| 5 | Run `bench_tts.py` head-to-head (F5 vs CosyVoice3 vs Audio8) | Operator (you) | Decision gate | Recipe written, needs HPC run |
| 6 | Decision gate: flip open_wispr `active: audio8` or not | Operator | — | Pending #5 — **URGENT due to F5-TTS license** |
| 7 | Wire Audio8 as open_wispr backend if it wins | This repo | — | Pending #6 |
| 8 | Write `docs/ops/h3-video-pipeline.md` (design only) | This repo | — | Pending — **blocked on H3 license legal review OR MOVA pivot** |
| 9 | Write `serve-minimax-h3.sh` OR `serve-mova-720p.sh` | This repo | — | Pending #8 legal decision |
| 10 | Add fleet dashboard rows for `A8TTS` and video-gen tag | This repo | #4, #9 | Pending |
| 11 | 🚨 Surface F5-TTS CC-BY-NC-4.0 license issue to legal | Operator | — | Pending — flag for awareness, decide on demote-to-legacy timing |
| 12 | 🚨 Surface H3 US-territory license conflict to legal | Operator | — | Pending — written confirmation from MiniMax needed before any HPC deployment |

---

## 9. Open questions for the user

1. **F5-TTS license issue (URGENT)**: F5-TTS is CC-BY-NC-4.0 (non-commercial only). For the company corporate use, this is a compliance issue. Do you want to (a) flip `active: audio8` immediately without waiting for the full bench (accept Audio8 as default on the strength of the peer-verification alone), (b) run the bench first then flip, or (c) demote F5-TTS to legacy-only now and use CosyVoice (Apache-2.0) as a temporary default until the bench completes?
2. **H3 legal review (URGENT)**: H3's literal LICENSE text excludes the US; Q&A doc says opposite. Should I draft a one-pager for legal/procurement to get written confirmation from MiniMax BEFORE we invest design time in the H3 video pipeline? Or pivot to MOVA (Apache 2.0, no legal risk, lower quality) for the design doc?
3. **Fleet dashboard ownership**: The other agent is rebuilding `fleet_watch_local.py` / `fleet_panel.py`. Should Audio8 and video-gen entries be added by them (they own the table format) or by us (we own the serve scripts)?
4. **Bench prompt set**: The 8 default prompts in `bench_tts.py` are pharma-MMM-flavored. Want to expand to a richer set (e.g., the Seed-TTS test set, or your own internal eval set)?
5. **Voice cloning reference clips**: For the SIM (speaker similarity) pass in `bench_tts.py`, we need reference clips. Are there internal the company voice samples we can use, or should I use a public reference set (LibriSpeech)?

---

## 10. References

- `CLAUDE.md` (repo root) — source of truth for sbatch table, QoS, fleet list
- `docs/ops/client-tooling.md` — Windows client tooling reference
- `docs/ops/proxy.md` — claude-code-proxy internals
- `serving/fleet_watch_local.py` — fleet watchdog (owned by other agent)
- `open_wispr/README.md` — speech stack reference
- `serving/serve-vibevoice-asr.sh` — VibeVoice-ASR standalone recipe (template for Audio8)
- `serving/bench_tts.py` — closed-loop TTS eval (this architecture, Path E)
- `/hpc` and `/hpc-code` skills (`/.claude/skills/`) — operator-facing Slurm + launcher troubleshooting
