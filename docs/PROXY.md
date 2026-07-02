# Proxy Layer

> **Anthropic Messages API ↔ OpenAI Chat Completions translation, plus context management for smaller-window backends.**

The proxy is what makes self-hosted open-source LLMs drop-in for a Claude Code / crush client without any Anthropic API traffic. It sits between the client (which speaks Anthropic's Messages API) and the serving engine (SGLang, vLLM, native Transformers — all OpenAI-compatible).

This document covers three things:
1. **API translation** — Anthropic → OpenAI mapping (reasoning_effort, max_tokens, tool schemas).
2. **Context management** — how to handle backends with less context than the client assumes.
3. **Summarization for smaller-window backends** — the pattern that prevents silent context loss when the backend can't hold a full Sonnet-class window.

---

## API Translation

The proxy translates:

| Anthropic concept | Backend concept (OpenAI-compatible) | Notes |
|-------------------|-------------------------------------|-------|
| `messages` array of `content` blocks | `messages` array of role+text | `tool_use` / `tool_result` blocks flatten into function-call / function-result messages |
| `thinking` block toggle | `reasoning_effort` field | Per-engine mapping — see below |
| `max_tokens` | `max_completion_tokens` or `max_tokens` | Cap adjusted per-engine to avoid EAGLE scheduling degradation |
| `stop_sequences` | `stop` | 1:1 |
| Tool schemas (verbose JSON Schema) | Simplified schemas for smaller models | Optional; reduces tool-def tokens by ~50% |

### `reasoning_effort` per Engine

Not every backend accepts every `reasoning_effort` value. Handle this in the translation layer:

| Engine | `"none"` | `"low"` | `"medium"` | `"high"` | `"max"` |
|--------|----------|---------|-----------|---------|---------|
| SGLang pip (GLM-5.x) | ✅ suppresses thinking | ✅ | ✅ (maps to `high`) | ✅ | ✅ |
| SGLang container (DeepSeek-V4 family) | ❌ rejects | ✅ | ✅ | ✅ | ✅ |
| vLLM (Kimi K2.x family) | ❌ ignored (always thinks) | ❌ ignored | ❌ ignored | ❌ ignored | ❌ ignored |
| Native Transformers | N/A (usually non-reasoning) | omit field | omit field | omit field | omit field |

**Rule:** if the client says "thinking off" and the engine can't accept `"none"`, **omit the field entirely** — don't pass `"none"` to an engine that rejects it, and don't pass a truthy value to one that can't turn thinking off.

### `max_tokens` Caps per Engine

EAGLE-based speculative decoding pre-allocates CUDA graph memory to `max_tokens`. Uncapped, a client requesting 32K output can crater EAGLE's steady-state throughput. Cap in the proxy:

| Engine | Cap | Reason |
|--------|-----|--------|
| SGLang container with EAGLE | 4096 (no thinking) / 8192 (thinking) | EAGLE CUDA graphs pre-alloc |
| SGLang pip with MTP | 16384 | MTP scheduling less punished by high caps |
| vLLM (PagedAttention) | 16384 | No EAGLE penalty, but leaves context margin |

---

## Context Management for Smaller-Window Backends

A common mismatch: **Claude Code assumes the backend has Sonnet-class context (~1M tokens)** because the client-side settings declare `ANTHROPIC_MODEL=claude-sonnet-4-6`. Its own auto-compact fires at ~80% of that — around **800K**.

If the actual backend has less context (e.g., a quantized model capped at 512K), the client's auto-compact **never fires**, and requests can arrive over the backend's limit. The proxy has three options:

1. **Reject** — return `400 context_length_exceeded`. Terrible UX, user has to `/clear` manually.
2. **Trim silently** — drop oldest messages until the payload fits. Session degrades invisibly.
3. **Summarize** — call the backend model itself to compress the oldest portion into a preserving summary, leave the recent turns verbatim.

Options 1 and 2 are unacceptable on any tool a developer actually uses. Option 3 is the pattern documented in the rest of this file.

### Symptom Detection

Set a hard `BACKEND_CONTEXT_LIMIT` in the proxy environment matching the serve script's `--context-length`. On every request, estimate `input_tokens = total_chars / 1.5` (undercounts less than `/2` on tool-heavy conversations) and compare to `BACKEND_CONTEXT_LIMIT - max_completion - safety_margin`.

If over: enter the compression cascade.

### The Compression Cascade

Order matters — cheapest, most-preserving strategies first:

| Tier | Strategy | Preservation | When to use |
|------|----------|--------------|-------------|
| 1 | Truncate `tool_result` bodies to first N KB | High (keeps intent) | Every request over budget |
| 2 | **Summarize old messages via the backend model itself** | High (semantic) | When Tier 1 wasn't enough |
| 2-fallback | Compress assistant text messages | Medium | Only if summarization fails |
| 3-fallback | Truncate user messages | Low | Only if the above fails |
| 4-fallback | Drop oldest messages | **Zero — lossy, avoid** | Last resort |

The key move is **Tier 2 = summarize, not compress-then-drop.** Summarize preserves semantics; drop deletes turns.

### Summarization Pattern (Tier 2)

The proxy makes a self-call to its own backend with a preservation-focused prompt:

```
System: You are compressing an earlier portion of a coding conversation for context
management. Produce a DENSE, chronological summary in under N tokens. PRESERVE EXACTLY:
- Every file path mentioned, read, or edited
- Every command run and its outcome (success/failure/key output)
- Every technical decision made and its stated reason
- Every error encountered and how it was resolved
- User's goals, constraints, and preferences
- Names of functions/classes/variables created or modified
Format as terse bullet points grouped by topic. No commentary, no filler.
Do NOT paraphrase quoted user requirements — keep them verbatim in quotes.
This summary REPLACES the raw messages in a follow-up conversation, so
completeness matters more than brevity.

User: Summarize this conversation transcript:
<serialized old messages>
```

The result replaces the compressible portion as a single assistant message wrapped in `[compressed-earlier-session]...[/compressed-earlier-session]` markers. The recent tail (last ~8 messages) stays verbatim so active tool-call chains never get summarized mid-stream.

### Cache

Content-hash the compressible portion (SHA-256 of JSON-serialized messages) and cache the resulting summary in-process. LRU cap ~20 entries. Same compressible prefix on a subsequent turn → instant reuse.

Real numbers on a mid-sized quantized MoE (~40B active):
- **Cold call**: ~2 seconds to summarize ~50 messages into ~1000 tokens.
- **Cache hit**: ~10 ms (200× speedup).
- **Preservation**: file paths, tool-call outcomes, benchmark numbers, error resolutions all retained.

### Visibility

The user has to know when compression fired. Two channels:

1. **Log line** — one info-level log per compression event, greppable.
2. **In-conversation banner** — inject a synthetic system-role message *before* forwarding the compressed context: `"[proxy: session at context limit. Compressed N older messages into a ~T-token summary. IMPORTANT: In your next reply, briefly mention that older context was summarized so the user knows compression happened.]"`

Because it's a system message, the model naturally surfaces it in its next reply. The user sees compression happen live in the conversation, not just in a log file.

### Fallback

If the self-call fails (timeout, backend unreachable, quota), fall through to the legacy Tier 2-fallback / Tier 3-fallback / Tier 4-fallback lossy cascade rather than returning an error. Safety net over correctness — a degraded session is better than a broken one.

### When to Enable

Enable on backends where the client-declared context exceeds the actual serve-time context:

- Backend with quantized weights at reduced context (e.g., 512K NVFP4 on 4× GPUs when client assumes 1M Sonnet)
- Any backend whose `--context-length` is < 80% of the client's declared model window

**Do NOT enable on backends that match the client's declared context.** The extra self-call adds ~2s of latency on the compression trigger, and if the client's own auto-compact will fire first, there's no reason to intervene.

### Tuning Knobs (env vars)

Recommended env-var interface for the proxy:

| Var | Default | Purpose |
|-----|---------|---------|
| `BACKEND_CONTEXT_LIMIT` | 0 (disabled) | Hard ceiling; below this the safety net doesn't run |
| `SUMMARIZE_ENABLED` | 0 | Master switch for Tier 2 summarization |
| `SUMMARIZE_TARGET_TOKENS` | 2000 | Summary length ceiling |
| `SUMMARIZE_MAX_LATENCY_S` | 30 | httpx timeout on the self-call |
| `SUMMARIZE_MODEL` | `$BIG_MODEL` | Override if you want a different model doing the summarization |
| `COMPRESS_PROTECTED_TAIL` | 8 | Last N messages always kept verbatim |

### Testing

Ship a probe script that forces summarization by setting `BACKEND_CONTEXT_LIMIT` low and constructing a conversation over the limit. Verify:
1. Summary path executes without crashing.
2. Summary preserves key facts (file paths, tool outcomes, decisions).
3. Recent tail stays verbatim.
4. Cache hit on identical retry.

Ballpark: on a mid-sized MoE, a 50-message synthetic conversation compresses in 2 seconds and preserves ≥80% of the content-critical facts.

---

## Cross-References

- Loading OSS LLMs onto HPC: [`../README.md`](../README.md#loading-open-source-llms-onto-hpc)
- Serving tuning (EAGLE, MTP, chunked prefill): [`SERVING_TUNING.md`](SERVING_TUNING.md) *(planned)*
- QoS + hot-swap: [`OPERATIONS.md`](OPERATIONS.md)
- Container deployment issues: [`DEPLOYMENT_LESSONS.md`](DEPLOYMENT_LESSONS.md) *(planned)*
