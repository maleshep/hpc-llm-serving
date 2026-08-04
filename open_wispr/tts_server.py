"""
OpenWispr TTS Server — CosyVoice3-0.5B

FastAPI server for text-to-speech and zero-shot voice cloning.
Runs on HPC alongside ASR + LLM on the same L40S GPU.

Usage:
    uvicorn tts_server:app --host 0.0.0.0 --port 8300

VRAM: ~2 GB
"""

import io
import time
import tempfile
from pathlib import Path

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="OpenWispr TTS", version="1.0.0")

cosyvoice = None
MODEL_PATH = "/shared/project/hpc-llm/models/cosyvoice3-0.5b"


@app.on_event("startup")
async def load_model():
    """Load CosyVoice3 model."""
    global cosyvoice

    # CosyVoice uses its own model loading API
    import sys
    sys.path.insert(0, str(Path(MODEL_PATH).parent / "CosyVoice"))

    from cosyvoice.cli.cosyvoice import CosyVoice2

    print(f"Loading CosyVoice3 from {MODEL_PATH}...")
    cosyvoice = CosyVoice2(MODEL_PATH, load_jit=False, load_trt=False)
    print(f"CosyVoice3 loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Fun-CosyVoice3-0.5B",
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0,
    }


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    voice: str = Form("default"),
    speed: float = Form(1.0),
):
    """
    Synthesize speech from text using a built-in voice.

    - text: The text to speak
    - voice: Voice preset name (default, or one of CosyVoice's built-in speakers)
    - speed: Speech speed multiplier (0.5-2.0)
    """
    t0 = time.perf_counter()

    # Use CosyVoice's built-in zero-shot or SFT inference
    # For built-in voices, use sft_inference
    audio_chunks = []
    for chunk in cosyvoice.inference_sft(text, voice, stream=False):
        audio_chunks.append(chunk["tts_speech"])

    if not audio_chunks:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Concatenate all chunks
    audio = torch.cat(audio_chunks, dim=-1)

    # Apply speed adjustment
    if speed != 1.0:
        audio = torchaudio.functional.speed(audio, orig_freq=22050, factor=speed)[0]

    # Convert to WAV bytes
    buf = io.BytesIO()
    torchaudio.save(buf, audio.unsqueeze(0).cpu(), 22050, format="wav")
    buf.seek(0)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Processing-Ms": str(round(elapsed_ms, 1)),
            "X-Audio-Duration-S": str(round(audio.shape[-1] / 22050, 2)),
        },
    )


@app.post("/clone")
async def clone(
    reference: UploadFile = File(...),
    text: str = Form(...),
):
    """
    Zero-shot voice cloning: speak the given text in the reference voice.

    - reference: 3-10 second WAV/MP3 sample of the target voice
    - text: Text to synthesize in that voice
    """
    t0 = time.perf_counter()

    # Save reference audio
    ref_bytes = await reference.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(ref_bytes)
        ref_path = tmp.name

    try:
        # Load reference for prompt extraction
        ref_audio, ref_sr = torchaudio.load(ref_path)

        # Resample to model's expected rate if needed
        if ref_sr != 16000:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 16000)

        # Use CosyVoice's zero-shot inference
        # The reference audio is used to extract speaker embedding
        audio_chunks = []
        for chunk in cosyvoice.inference_zero_shot(
            text,
            "",  # prompt text (can be empty for pure cloning)
            ref_path,
            stream=False,
        ):
            audio_chunks.append(chunk["tts_speech"])

        if not audio_chunks:
            return JSONResponse({"error": "No audio generated"}, status_code=500)

        audio = torch.cat(audio_chunks, dim=-1)

        buf = io.BytesIO()
        torchaudio.save(buf, audio.unsqueeze(0).cpu(), 22050, format="wav")
        buf.seek(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={
                "X-Processing-Ms": str(round(elapsed_ms, 1)),
                "X-Audio-Duration-S": str(round(audio.shape[-1] / 22050, 2)),
            },
        )
    finally:
        Path(ref_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8300)
