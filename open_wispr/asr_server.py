"""
OpenWispr ASR Server — Qwen2-Audio-7B-Instruct

FastAPI server that accepts audio uploads and returns transcriptions.
Runs on HPC alongside SGLang (Qwen3.6 LLM) on the same L40S GPU.

Usage:
    uvicorn asr_server:app --host 0.0.0.0 --port 8200

VRAM: ~14 GB (BF16)
"""

import io
import time
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="OpenWispr ASR", version="1.0.0")

# Global model references (loaded once at startup)
processor = None
model = None
DEVICE = "cuda"
MODEL_PATH = "/shared/project/hpc-llm/models/qwen2-audio-7b"


@app.on_event("startup")
async def load_model():
    """Load Qwen2-Audio model into GPU memory."""
    global processor, model

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2-Audio from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Qwen2-Audio loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Qwen2-Audio-7B-Instruct",
        "device": DEVICE,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    prompt: str = Form("Transcribe this audio."),
):
    """
    Transcribe uploaded audio file to text.

    - audio: WAV, MP3, FLAC, or OGG file
    - language: target language hint (en, de, zh, etc.)
    - prompt: instruction for the model (default: simple transcription)
    """
    t0 = time.perf_counter()

    # Save uploaded audio to temp file (Qwen2-Audio needs file path or librosa array)
    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        import librosa

        # Load audio at 16kHz (Qwen2-Audio expected sample rate)
        audio_array, sr = librosa.load(tmp_path, sr=16000)

        # Build conversation for the model
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process with the chat template
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text],
            audios=[audio_array],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )

        # Decode — skip input tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        text_out = processor.decode(generated, skip_special_tokens=True).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration_s = len(audio_array) / sr

        return JSONResponse({
            "text": text_out,
            "duration_s": round(duration_s, 2),
            "processing_ms": round(elapsed_ms, 1),
            "language": language,
        })

    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200)
