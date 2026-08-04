"""
Head-to-head TTS eval: F5-TTS vs CosyVoice3 vs Audio8-TTS.

Closed loop:
    text --> TTS backend --> WAV --> VibeVoice-ASR (port 8300) --> transcription
    WER = edit_distance(transcription, original_text) / len(original_text)

Optional (if resemblyzer/pyannote available in the venv):
    speaker embedding cosine similarity between reference clone clip and
    synthesized clip.

Each backend is hit via its own OpenAI-compatible /v1/audio/speech endpoint
(or the open_wispr unified /synthesize on port 8200 for the active backend).

Usage:
    # All three backends, default prompts
    python serving/bench_tts.py

    # Custom prompts file (JSON list of {"id","text"} dicts)
    python serving/bench_tts.py --prompts my_prompts.json

    # Skip the speaker-similarity pass (no resemblyzer installed)
    python serving/bench_tts.py --no-sim

Outputs:
    serving/bench_tts_results.json   (full per-prompt scores)
    serving/bench_tts_summary.csv    (one row per backend, aggregate)

Decision gate: if Audio8 wins on WER AND RTF (real-time factor) is acceptable,
flip open_wispr/model_registry.yaml: tts.active: f5-tts -> audio8.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
import wave
import io
import tempfile
from pathlib import Path

# ============================================================================
# Backend configs — each speaks OpenAI /v1/audio/speech (or open_wispr shape).
# Ports match the .serve-state*.json files / serve scripts.
# ============================================================================
BACKENDS = {
    "f5": {
        "url": "http://localhost:8200/synthesize",   # open_wispr unified (active=f5-tts)
        "shape": "openwispr",                        # form-fields, not OpenAI JSON
        "voice_field": "voice",
    },
    "cosyvoice": {
        "url": "http://localhost:8200/synthesize",   # same server, swap-model to cosyvoice
        "shape": "openwispr",
        "voice_field": "voice",
        "pre_swap": ("http://localhost:8200/admin/swap-model",
                     {"service": "tts", "backend_name": "cosyvoice"}),
        "post_swap": ("http://localhost:8200/admin/swap-model",
                      {"service": "tts", "backend_name": "f5-tts"}),
    },
    "audio8": {
        "url": "http://localhost:8310/v1/audio/speech",  # SGLang Omni standalone
        "shape": "openai",
        "voice_field": "voice",
    },
}

# VibeVoice-ASR (serve-vibevoice-asr.sh, port 8300) — OpenAI chat-style.
ASR_URL = "http://localhost:8300/v1/chat/completions"
ASR_MODEL = "vibevoice"

# ============================================================================
# Default eval prompts — short, multilingual, mix of plain text + SSML-ish.
# ============================================================================
DEFAULT_PROMPTS = [
    {"id": "en-short",  "text": "The recommended budget allocation shifts thirty percent toward digital channels."},
    {"id": "en-med",    "text": "Our meta-harness analysis across four thousand iterations shows Gemma four outperforms Sonnet four point six on pharma marketing mix decisions with a fifty nine percent win rate."},
    {"id": "en-long",   "text": "The fleet dashboard shows GLM five point two FP8 sustaining two hundred twenty tokens per second on eight B200 GPUs with one megabyte of context, while MiniMax M3 on four B200s delivers one hundred seventeen tokens per second with vision and video enabled."},
    {"id": "de",        "text": "Die empfohlene Budgetverteilung verschiebt dreißig Prozent in Richtung digitale Kanäle."},
    {"id": "fr",        "text": "L'allocation budgétaire recommandée déplace trente pour cent vers les canaux numériques."},
    {"id": "es",        "text": "La asignación presupuestaria recomendada desplaza el treinta por ciento hacia los canales digitales."},
    {"id": "zh",        "text": "建议的预算分配将百分之三十转向数字渠道。"},
    {"id": "ja",        "text": "推奨される予算配分は、デジタルチャネルに30%をシフトします。"},
]

# ============================================================================
# Helpers
# ============================================================================

def http_post(url, body_bytes, headers=None, timeout=120):
    """Plain urllib POST, returns raw response bytes."""
    req = urllib.request.Request(url, data=body_bytes, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_post_json(url, obj, timeout=120):
    return http_post(url, json.dumps(obj).encode(),
                     {"Content-Type": "application/json"}, timeout)


def synthesize(backend_cfg, text, voice="default"):
    """Hit a TTS backend, return (wav_bytes, elapsed_seconds)."""
    t0 = time.time()
    if backend_cfg["shape"] == "openai":
        # OpenAI /v1/audio/speech shape
        body = json.dumps({
            "model": "audio8-tts",
            "input": text,
            "voice": voice,
            "response_format": "wav",
        }).encode()
        wav = http_post(backend_cfg["url"], body,
                        {"Content-Type": "application/json"})
    elif backend_cfg["shape"] == "openwispr":
        # open_wispr /synthesize form-fields -> returns audio/wav
        # Build multipart/form-data manually (stdlib, no requests dep)
        boundary = "----bench_tts" + str(int(time.time() * 1000))
        parts = []
        for k, v in [("text", text), ("voice", voice), ("speed", "1.0")]:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            parts.append(v.encode() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        wav = http_post(backend_cfg["url"], body,
                        {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        raise ValueError(f"unknown shape: {backend_cfg['shape']}")
    return wav, time.time() - t0


def transcribe(wav_bytes):
    """Send WAV bytes to VibeVoice-ASR, return transcript string."""
    # VibeVoice uses OpenAI chat-style: audio as a data URI in message content.
    import base64
    b64 = base64.b64encode(wav_bytes).decode()
    data_uri = f"data:audio/wav;base64,{b64}"
    body = {
        "model": ASR_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio verbatim."},
                {"type": "audio_url", "audio_url": {"url": data_uri}},
            ],
        }],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": False,
    }
    r = json.loads(http_post_json(ASR_URL, body, timeout=180))
    return r["choices"][0]["message"]["content"].strip()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (lower=better). Simple Levenshtein on whitespace tokens."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # DP table
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    edits = prev[-1]
    return edits / len(ref)


def wav_duration_seconds(wav_bytes) -> float:
    """Parse WAV header for duration in seconds."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def maybe_swap(url_and_fields):
    """Call open_wispr /admin/swap-model if a pre/post swap is configured."""
    if not url_and_fields:
        return
    url, fields = url_and_fields
    # Multipart form
    boundary = "----swap" + str(int(time.time() * 1000))
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        parts.append(v.encode() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    try:
        http_post(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=30)
    except Exception as e:
        print(f"  WARN: swap-model failed: {e}", file=sys.stderr)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", help="JSON file with list of {id,text}")
    ap.add_argument("--no-sim", action="store_true", help="skip speaker similarity")
    ap.add_argument("--only", help="comma-separated backend names to run (default: all)")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())

    selected = list(BACKENDS.keys())
    if args.only:
        selected = [b.strip() for b in args.only.split(",") if b.strip() in BACKENDS]

    results = {b: [] for b in selected}

    for name in selected:
        cfg = BACKENDS[name]
        print(f"\n=== Backend: {name} ({cfg['url']}) ===")
        maybe_swap(cfg.get("pre_swap"))

        # Warmup
        try:
            _, _ = synthesize(cfg, "Warmup.", voice="default")
        except Exception as e:
            print(f"  WARMUP FAILED: {e}")
            maybe_swap(cfg.get("post_swap"))
            continue

        for p in prompts:
            try:
                wav, synth_s = synthesize(cfg, p["text"])
                dur = wav_duration_seconds(wav)
                rtf = synth_s / dur if dur > 0 else float("inf")
                transcript = transcribe(wav)
                err = wer(p["text"], transcript)
                results[name].append({
                    "id": p["id"],
                    "text": p["text"],
                    "transcript": transcript,
                    "wer": err,
                    "synth_seconds": synth_s,
                    "audio_seconds": dur,
                    "rtf": rtf,
                })
                print(f"  {p['id']:12s} WER={err:.3f}  RTF={rtf:.2f}  synth={synth_s:.2f}s  audio={dur:.2f}s")
            except Exception as e:
                print(f"  {p['id']:12s} ERROR: {e}")
                results[name].append({"id": p["id"], "error": str(e)})

        maybe_swap(cfg.get("post_swap"))

    # Write full results
    Path("serving/bench_tts_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))

    # Write summary CSV
    with open("serving/bench_tts_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "n", "wer_mean", "wer_median", "rtf_mean", "synth_s_mean"])
        for name in selected:
            ok = [r for r in results[name] if "wer" in r]
            if not ok:
                w.writerow([name, 0, "", "", "", ""])
                continue
            wers = sorted(r["wer"] for r in ok)
            rtfs = [r["rtf"] for r in ok]
            synths = [r["synth_seconds"] for r in ok]
            n = len(ok)
            wer_mean = sum(wers) / n
            wer_median = wers[n // 2]
            rtf_mean = sum(rtfs) / n
            synth_mean = sum(synths) / n
            w.writerow([name, n,
                        f"{wer_mean:.4f}", f"{wer_median:.4f}",
                        f"{rtf_mean:.3f}", f"{synth_mean:.3f}"])

    print("\n=== SUMMARY ===")
    print(f"  Results: serving/bench_tts_results.json")
    print(f"  Summary: serving/bench_tts_summary.csv")


if __name__ == "__main__":
    main()
