# Laguna DFlash to NGRAM Pivot

Date: 2026-07-26
Trace: `.optimization/trace_laguna-dflash_2026-07-26/`

## Goal

DFlash block-diffusion on Poolside Laguna S 2.1 (2x B200, NVFP4) targeting ~270 tok/s (2.4x over the 113 tok/s plain baseline). DFlash reuses the base model's own hidden states as a draft, avoiding a separate drafter network.

## Timeline (9 checkpoints)

**CP1-6 — Integration surgery.** Backported `set_dflash_layers_to_capture` into the SGLang Laguna path (absent in the pinned revision). Added a `prefix_lens=None` guard to the capture hook so absent prefix tensors do not index-bomb during the first forward. Made the drafter forward's return tuple conditional: returns `(logits, hidden)` when DFlash capture is active, bare `logits` otherwise. Overrode the drafter config so KV-head and layer-count mismatches between the SGLang runtime config and the Laguna checkpoint metadata do not abort dispatch. After CP6 the model booted with DFlash enabled, but acceptance was zero.

**CP7 — Root cause localized.** DFlash capture during PREFILL produced NaN hidden states at layer 10 (the first full-attention layer). Decode-time captures were clean. The NaN was reproducible across `--moe-runner-backend` settings and across `--mem-fraction-static` values, isolating it to the prefill capture path.

**CP8 — Confirmation via source dive + Opus expert review.** The masked-MoE prefill path routes through `flashinfer_cutedsl_moe_masked` (tree-mask cu_seqlens). On B200/SM100 with NVFP4 expert weights, this kernel NaNs. Plain Laguna NVFP4 prefill does NOT NaN because normal causal attention never invokes the masked cutedsl MoE path. Opus 4.6 expert review of the FlashInfer source confirmed the kernel is unpatched for SM100 NVFP4 weight tensors. `--moe-runner-backend triton` only changes topk output formatting; the expert GEMM is still dispatched via the quantization_config-bound method. `--flashinfer-mxfp4-moe-precision bf16` is consumed by the standard trtllm path, not the masked cutedsl path. Not config-fixable from our side.

**CP9 — NGRAM pivot shipped.** Switched speculation to NGRAM (n-gram draft from KV cache, no target-hidden capture). Accept rate 0.05 to 0.38, peak 152 tok/s, 1.16-1.35x lossless speedup.

## Root cause

DFlash target-hidden capture during prefill routes through `flashinfer_cutedsl_moe_masked` (tree-mask cu_seqlens). On B200/SM100 with NVFP4 expert weights this NaNs at the first full-attention layer (layer 10). Plain Laguna NVFP4 prefill does not NaN — normal causal attention never touches the masked-MoE path. The NVFP4 expert method is bound to quantized weights via `quantization_config`, independent of `--moe-runner-backend`; setting `triton` changes topk output format, not the expert GEMM. `--flashinfer-mxfp4-moe-precision bf16` is consumed by the standard trtllm path, not the masked cutedsl path.

## Outcome

DFlash proven upstream-blocked. The correct remediation is a FlashInfer/SGLang fix to `flashinfer_cutedsl_moe_masked` for SM100 NVFP4; a repro should be filed upstream. NGRAM shipped as the working alternative: accept rate 0.05 to 0.38, peak 152 tok/s, lossless. `claude-laguna` wired to NGRAM (port 8115); plain Laguna (port 8114) killed.

## Lessons

1. Not every stalled sub-objective is fixable from our side. Upstream kernel bugs masquerade as configuration issues; exhaust the config space, then escalate to source.
2. The prefill-vs-decode NaN split was the diagnostic key. Isolating the failure to the capture path (prefill only) versus the decode path (clean) pinpointed the masked-MoE kernel without a debugger session.
3. NGRAM sidesteps the masked-MoE path entirely by not capturing target hidden states. When the fast path is kernel-blocked, the alternative path that avoids the kernel is the ship.

## Artifacts

- Trace: `.optimization/trace_laguna-dflash_2026-07-26/` (insights.md authored)
- Serve script: `serve-laguna-s21-nvfp4.sh` supports both `DFLASH_ON=1` and `NGRAM_ON=1`
- Memory: `project_laguna_dflash_blocked.md`
