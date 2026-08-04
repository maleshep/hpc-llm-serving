# Video Pipeline Design — MMM Agent to Narrated Video Pitch

**Status**: Design doc, 2026-08-04. **No GPU-hours committed.**
**Blocking**: H3 license legal review (see section 3).
**Scope**: Architecture for rendering Gemma-4-MMM-SFT's marketing-mix recommendations as 4-15s narrated video pitches.

---

## 1. The pipeline (north star)

```
User asks Gemma-4-MMM-SFT for a budget recommendation
  -> Gemma returns: "Shift 30% to digital channels because..."
    -> Recommendation captured as structured text (decision, rationale, headline number)
  -> Post to video-gen endpoint (proposed :8315)
    -> Model generates 4-15s video with native stereo audio
      (voice-over narrates the recommendation; visuals are abstract or brand-conditioned)
  <- Video returned, played in browser or saved to disk
```

**Why native audio matters**: A separate TTS + video pipeline would need:
1. TTS to generate voice-over from the recommendation text
2. Video gen to generate visuals (no audio)
3. ffmpeg to mux them with manual alignment

Native audio eliminates step 3 — the model handles lip/scene/voice synchronization in one pass. This is the only reason we're considering H3 over a TTS+video pipeline.

---

## 2. Two candidate models

### 2a. Primary: MiniMax H3 (if license clears)

- **Model**: `MiniMaxAI/MiniMax-H3` (33B dense + Qwen3-VL-32B encoder, ~130 GB BF16 total)
- **Hardware**: 8× B200 fat node, SGLang with `--num-gpus 4 --ulysses-degree 4`
- **Resolution**: up to 2K
- **Length**: 4-15s
- **Audio**: native stereo, 32 kHz
- **Ref2VA**: multi-reference conditioning — can feed brand assets (logo, color palette, prior ad clips) to condition the generation. **This is the killer feature for pharma marketing pitches.**
- **License**: MiniMax Community License — 🚨 **literal text excludes the US** (see section 3)
- **Serving**: SGLang Omni adapter ships with the model

### 2b. Fallback: MOVA (Apache 2.0, no license risk)

- **Model**: `OpenMOSS-Team/MOVA-720p` (32B MoE, 18B active)
- **Hardware**: 4× B200, SGLang-native
- **Resolution**: 720p (lower than H3's 2K)
- **Length**: TBD (per model card)
- **Audio**: native (per agent verification)
- **Ref2VA**: NOT supported — loses brand-asset conditioning
- **License**: Apache 2.0 — legally unambiguous, no US exclusion
- **Serving**: SGLang-tagged in model card

**Trade-off**: MOVA trades resolution (720p vs 2K) and Ref2VA (absent) for legal safety. For internal-only video pitches that don't need brand-asset conditioning, MOVA is sufficient. For external-facing creative with brand assets, H3 is materially better — if legal clears.

---

## 3. 🚨 H3 license blocker

**The conflict**: MiniMax H3's LICENSE text (the literal legal document on Hugging Face) **excludes the United States** from the list of permitted territories. MiniMax's Q&A document contradicts this, saying US use is allowed.

**Implication**: the company legal cannot rely on the Q&A document alone. The literal LICENSE text governs. Until this is resolved in writing, H3 deployment is blocked for the company (US-headquartered, Munich HPC also serves US-based employees).

**Resolution paths**:
1. **MiniMax written confirmation**: Procurement contacts MiniMax for a signed amendment explicitly permitting US use. Highest-confidence path, slowest (weeks).
2. **the company legal interpretation**: Legal reviews the LICENSE vs Q&A conflict and issues an internal memo interpreting which controls. Faster but risk-bearing.
3. **Pivot to MOVA**: Skip H3 entirely, eat the quality loss, ship on Apache-2.0. Zero legal risk, immediate.

**Recommendation**: Path 3 (pivot to MOVA) for the design doc and the first end-to-end pipeline build. If the pipeline proves valuable enough to justify H3's quality gain, initiate Path 1 in parallel.

---

## 4. Hardware plan (H3 path)

```bash
# sbatch serving/serve-minimax-h3.sh (DRAFT — do not submit until license clears)
#SBATCH --partition=fat
#SBATCH --gres=gpu:b200:8
#SBATCH --qos=3d                       # >10min boot, DeepGEMM-like JIT likely
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/project/hpc-llm/llm/logs/h3_%j.out

PORT=8315

apptainer exec --nv --cleanenv \
    $SIF \
    python3 -m sglang.launch_server \
        --model-path /models/minimax-h3 \
        --served-model-name minimax-h3 \
        --trust-remote-code \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --ulysses-degree 4 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --host 0.0.0.0 --port $PORT
```

**Why `--qos=3d`**: H3 is a 33B dense transformer + Qwen3-VL-32B encoder loaded together (~130 GB BF16). First boot likely involves SGLang Omni adapter JIT (5-15 min). Same QoS tier as GLM-5.2 and MiniMax M3 per CLAUDE.md.

**Why `--tensor-parallel-size 4 --ulysses-degree 4`**: per H3 model card SGLang recipe. 8× B200 fat node has 8 GPUs; Ulysses partitions attention heads across 4, TP shards weights across 4 — combined gives the cross-node attention parallelism the omni-modal architecture needs.

---

## 5. Hardware plan (MOVA path)

```bash
# sbatch serving/serve-mova-720p.sh
#SBATCH --partition=fat
#SBATCH --gres=gpu:b200:4
#SBATCH --qos=3d
#SBATCH --time=3-00:00:00

PORT=8316

apptainer exec --nv --cleanenv \
    $SIF \
    python3 -m sglang.launch_server \
        --model-path /models/mova-720p \
        --served-model-name mova \
        --trust-remote-code \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --host 0.0.0.0 --port $PORT
```

**Half the hardware of H3** (4× B200 vs 8× B200), no Ulysses needed since MOVA is a standard MoE without omni-modal cross-attention. **This is a tangible cost saving** if the quality trade-off is acceptable.

---

## 6. API contract (both paths)

```python
# POST /v1/videos/generations  (proposed OpenAI-compatible shape, modeled after /v1/images/generations)
{
    "model": "minimax-h3" | "mova",
    "prompt": "<Gemma-4-MMM-SFT recommendation text>",
    "duration_seconds": 8,        # 4-15 range
    "resolution": "720p" | "2k",  # model-dependent
    "reference_assets": [          # H3 only — Ref2VA conditioning
        {"type": "image", "url": "data:image/png;base64,..."},
        {"type": "video", "url": "..."},
        {"type": "audio", "url": "..."}
    ],
    "response_format": "mp4"
}

# Response: video bytes (or signed URL if we stage to S3-equivalent)
```

**Open question**: SGLang's Omni adapter for H3 may not expose this exact shape — the model card mentions SGLang support but the API contract isn't documented. First task once license clears: probe the actual endpoint shape with a curl and adapt.

---

## 7. End-to-end MVP build (MOVA path)

1. `serving/download-mova-720p.sh` — hf download to `/shared/project/hpc-llm/llm/models/mova-720p/`
2. `serving/serve-mova-720p.sh` — Slurm recipe (section 5 above)
3. `scripts/mmm_to_video.py` — Python client that:
   - Reads a Gemma-4-MMM-SFT recommendation (JSON: `{decision, rationale, headline_number}`)
   - Formats it as a video prompt ("Generate a 8-second video pitch: [decision]. Voice-over: [rationale]. Highlight: [headline_number].")
   - POSTs to the video-gen endpoint
   - Saves the returned MP4 to `/shared/project/hpc-llm/llm/output/video-pitches/<timestamp>.mp4`
4. Wire into Gemma-4-MMM-SFT serving: after the agent returns a recommendation, optionally auto-trigger the video render.

**Estimated effort**: 1-2 days for MVP, assuming MOVA SGLang Omni adapter works out of the box.

---

## 8. Use cases (concrete)

1. **Internal stakeholder pitches**: MMM agent runs on Q3 ad-spend data → renders a 10s video summarizing the recommendation for the marketing VP. More engaging than a slide; less work than a slide deck.
2. **Sales team enablement**: Pharma brand launch → MMM agent recommends channel mix → video pitch the rep can play on an iPad in the field.
3. **Training data**: Generate synthetic video pitches with known-good recommendations → fine-tune a future video-aware MMM agent.
4. **A/B creative testing**: Same recommendation, multiple video renderings (vary the prompt) → test which visual framing resonates with a sample audience.

**None of these are urgent enough to bypass the H3 license review**. All four work with MOVA at 720p.

---

## 9. Open questions

1. **License path**: Do you want to pursue H3 legal review in parallel with building the MOVA MVP, or commit to MOVA and revisit H3 only if MOVA's quality is insufficient?
2. **Use case priority**: Which of the four use cases in section 8 is the actual driver? If "internal stakeholder pitches," MOVA is clearly sufficient. If "A/B creative testing" at scale, H3's 2K + Ref2VA may matter.
3. **SGLang Omni adapter probe**: Want me to draft a one-page eval plan for probing the actual H3 API shape (separate from submitting a Slurm job), so when legal clears, we can deploy in hours not days?
4. **Output storage**: MP4s are 50-200 MB each for 15s at 2K. Where do they live? Project storage quota is finite. Do we want an auto-cleanup policy (e.g., delete after 30 days unless pinned)?
5. **Fleet dashboard**: Add `H3` (or `MOVA`) tag — us or the other agent?
