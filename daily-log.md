# Daily Activity Log

2026-06-06: Repo initialized.
2026-06-06 12:37 UTC: Daily sync.
2026-06-07 12:23 UTC: Daily sync.
2026-06-08 14:50 UTC: Daily sync.
2026-06-09 13:45 UTC: Daily sync.
2026-06-10 14:26 UTC: Daily sync.
2026-06-11 14:41 UTC: Daily sync.
2026-06-12 14:11 UTC: Daily sync.
2026-06-13 12:52 UTC: Daily sync.
2026-06-14 13:01 UTC: Daily sync.
2026-06-15 16:20 UTC: Daily sync.
2026-06-16 15:59 UTC: Daily sync.
2026-06-17 14:27 UTC: Daily sync.
2026-06-18 14:13 UTC: Daily sync.
2026-06-19 14:14 UTC: Daily sync.
2026-06-20 12:27 UTC: Daily sync.
2026-06-21 13:10 UTC: Daily sync.
2026-06-22 15:57 UTC: Daily sync.
2026-06-23 13:44 UTC: Daily sync.
2026-06-24 13:24 UTC: Daily sync.
2026-06-25 13:22 UTC: Daily sync.
2026-06-26 13:14 UTC: Daily sync.
2026-06-27 12:10 UTC: Daily sync.
2026-06-28 12:18 UTC: Daily sync.
2026-06-29 14:48 UTC: Daily sync.
2026-06-30 13:10 UTC: Daily sync.
2026-07-01 13:35 UTC: Daily sync.
2026-07-02 12:00 UTC: Added `docs/PROXY.md` — API translation table (reasoning_effort per engine, max_tokens caps) and the context-management pattern for smaller-window backends. Documents Tier-2 summarization (proxy self-calls the backend to compress old context into a preserving summary rather than dropping messages), content-hash caching, in-conversation visibility banner, and fallback semantics. Applies to any backend whose serve-time context is smaller than the client's declared model window (e.g. quantized MoE at 512K vs Sonnet-class 1M assumption).
2026-07-02 13:00 UTC: Daily sync.
2026-07-03 12:58 UTC: Daily sync.
2026-07-04 12:09 UTC: Daily sync.
2026-07-05 12:19 UTC: Daily sync.
2026-07-06 14:29 UTC: Daily sync.
2026-07-07 13:22 UTC: Daily sync.
2026-07-08 12:22 UTC: Daily sync.
2026-07-09 13:51 UTC: Daily sync.
2026-07-10 13:14 UTC: Daily sync.
2026-07-11 11:55 UTC: Daily sync.
2026-07-12 12:02 UTC: Daily sync.
2026-07-13 13:26 UTC: Daily sync.
2026-07-14 12:17 UTC: Daily sync.
2026-07-15 12:18 UTC: Daily sync.
2026-07-16 12:22 UTC: Daily sync.
2026-07-17 12:12 UTC: Daily sync.
2026-07-18 11:54 UTC: Daily sync.
2026-07-19 12:01 UTC: Daily sync.
2026-07-20 13:09 UTC: Daily sync.
2026-07-21 12:24 UTC: Daily sync.
2026-07-22 12:27 UTC: Daily sync.
2026-07-23 12:25 UTC: Daily sync.
2026-07-24 12:22 UTC: Daily sync.
2026-07-25 12:04 UTC: Daily sync.
2026-07-26 12:06 UTC: Daily sync.
2026-07-27 13:40 UTC: Daily sync.
2026-07-28 12:56 UTC: Daily sync.
2026-07-29 13:02 UTC: Daily sync.
2026-07-30 12:47 UTC: Daily sync.
2026-07-31 12:58 UTC: Daily sync.
2026-08-01 12:04 UTC: Daily sync.
2026-08-02 12:05 UTC: Daily sync.
2026-08-03 13:42 UTC: Daily sync.
2026-08-04 13:04 UTC: Daily sync.
2026-08-04 21:10 UTC: Renamed `wisprflow/` -> `open_wispr/` (SaaS name collision with Wispr Flow consumer product). Added `serving/serve-audio8-tts.sh` + `download-audio8-tts.sh` — SGLang Omni recipe for Audio8-TTS-Preview-0.6b on 1x L40S, port 8310, QoS 1d. Added `serving/bench_tts.py` — closed-loop TTS eval (synthesize -> VibeVoice-ASR transcribe -> WER), no external deps. Added `docs/ops/multimodal-stack-architecture.md` — 3-tier serving design (Claude Code fleet / standalone eval engines / open_wispr unified), 6 data-flow paths, port allocation, decision gates per tier. Added `docs/ops/h3-video-pipeline.md` — dual-path video-gen design (H3 if license clears, MOVA Apache-2.0 fallback). Peer-verification: Audio8 confirmed best of 16 TTS peers (Apache 2.0, only model clearing all 6 hard constraints); MiniMax H3 technically best of 6 video-gen peers but LICENSE text excludes US territory — blocked on legal review, MOVA is drop-in fallback.
2026-08-04 22:00 UTC: DeepSeek V4-Flash-0731 replaced old V4-Flash in place (304B MoE-A8B vs 284B-A13B; beats V4-Pro on every coding benchmark per DeepSeek model card: Terminal-Bench 82.7 vs 72.1, DeepSWE 54.4 vs 12.8, Toolathlon 70.3 vs 55.9). 4x B200, 1M native ctx, EAGLE 3-token speculation (DSpark/DFLASH not supported for DeepSeekV4 arch in our SGLang container), MXFP4 runtime MoE quant, fp8_e4m3 KV cache, mem-fraction-static 0.85. Measured 79 tok/s on 4x B200. Fixed systemic proxy env-var bug: restart_proxy() was setting OPENAI_*/BIG_MODEL/SMALL_MODEL vars that server.py never reads; every model was silently defaulting to PROFILE=glm52 + BACKEND_URL=localhost:8103 regardless of requested model. Now sets PROFILE/BACKEND_URL/BACKEND_MODEL/ROUTE_NAME correctly + _kill_port() prevents stale-process squatting. Fleet dashboard integration: added V4-Flash slot to fleet_panel.py TUNNELS, fleet_watch_local.py PROXY_CONFIG/LOGGER_CONFIG/LOCAL_PORTS, fleet-monitor.py V4FLASH_JOBNAME + state.json v4flash field, FLEET panel render row. Bumped fetch_hpc_state() SSH timeout 20s -> 30s. Verified end-to-end: claude-ds.cmd -> proxy :5010 -> tunnel :8100 -> 200 OK response.

2026-08-05 12:30 UTC: V4-Flash-0731 + DSpark speculation hits 256 tok/s on 4x B200 (3.1x baseline of 83 tok/s with EAGLE). DSpark requires SGLang 0.5.16+ (sglang-v0516.sif container); older 0.5.13.post1 hard-asserts EAGLE only for DeepSeekV4. Winning config: --speculative-algorithm DSPARK --speculative-dspark-block-size 5, flashinfer_mxfp4 MoE runtime, fp8_e4m3 KV cache, cuda-graph-max-bs 192, mem-fraction 0.90, enable-deepseek-v4-fp4-indexer. Beats V4-Pro on every coding benchmark (Terminal-Bench 82.7 vs 72.1, DeepSWE 54.4 vs 12.8, Toolathlon 70.3 vs 55.9). New ops doc: docs/ops/v4flash-0731-dspark-deployment.md. New /meta-harness-loop skill. Removed /hpc-code skill (folded into hpc + docs/ops/client-tooling.md).
2026-08-05 12:56 UTC: Daily sync.
2026-08-06 12:58 UTC: Daily sync.
2026-08-07 11:49 UTC: Daily sync.
2026-08-08 11:32 UTC: Daily sync.
2026-08-09 11:32 UTC: Daily sync.
2026-08-10 11:50 UTC: Daily sync.
