"""
OpenWispr Unified Server — Model-agnostic ASR + TTS

Single FastAPI application that dynamically loads backends from model_registry.yaml.
Replaces the separate asr_server.py and tts_server.py.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8200

Ports:
    8200 — unified (ASR + TTS on same port)
"""

import io
import time
import importlib
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from backends import ASRBackend, TTSBackend

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path(__file__).parent / "model_registry.yaml"

# Active backend instances
asr_backend: ASRBackend | None = None
tts_backend: TTSBackend | None = None
registry: dict = {}


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def instantiate_backend(class_path: str):
    """Dynamically import and instantiate a backend class from dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


async def load_asr_backend(config: dict, backend_name: str) -> ASRBackend:
    backend_cfg = config["asr"]["backends"][backend_name]
    class_path = backend_cfg["class"]
    kwargs = {k: v for k, v in backend_cfg.items() if k != "class"}

    backend = instantiate_backend(class_path)
    await backend.load(**kwargs)
    return backend


async def load_tts_backend(config: dict, backend_name: str) -> TTSBackend:
    backend_cfg = config["tts"]["backends"][backend_name]
    class_path = backend_cfg["class"]
    kwargs = {k: v for k, v in backend_cfg.items() if k != "class"}

    backend = instantiate_backend(class_path)
    await backend.load(**kwargs)
    return backend


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_backend, tts_backend, registry

    registry = load_registry()

    # Load ASR
    asr_name = registry["asr"]["active"]
    print(f"Loading ASR backend: {asr_name}")
    try:
        asr_backend = await load_asr_backend(registry, asr_name)
    except Exception as e:
        print(f"WARNING: Failed to load ASR backend '{asr_name}': {e}")
        # Try fallback
        fallback_names = [n for n in registry["asr"]["backends"] if n != asr_name]
        for fb in fallback_names:
            try:
                print(f"  Trying fallback ASR: {fb}")
                asr_backend = await load_asr_backend(registry, fb)
                break
            except Exception as e2:
                print(f"  Fallback '{fb}' also failed: {e2}")

    # Load TTS
    tts_name = registry["tts"]["active"]
    print(f"Loading TTS backend: {tts_name}")
    try:
        tts_backend = await load_tts_backend(registry, tts_name)
    except Exception as e:
        print(f"WARNING: Failed to load TTS backend '{tts_name}': {e}")
        fallback_names = [n for n in registry["tts"]["backends"] if n != tts_name]
        for fb in fallback_names:
            try:
                print(f"  Trying fallback TTS: {fb}")
                tts_backend = await load_tts_backend(registry, fb)
                break
            except Exception as e2:
                print(f"  Fallback '{fb}' also failed: {e2}")

    yield  # App runs

    # Cleanup (nothing to do for now)
    print("Shutting down OpenWispr server.")


app = FastAPI(title="OpenWispr", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API Endpoints — backward-compatible with old asr_server + tts_server
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    asr_health = await asr_backend.health() if asr_backend else {"status": "not_loaded"}
    tts_health = await tts_backend.health() if tts_backend else {"status": "not_loaded"}
    return {
        "status": "ok",
        "asr": asr_health,
        "tts": tts_health,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
):
    """
    Transcribe uploaded audio file to text.
    Backward-compatible with old /transcribe endpoint.
    """
    if not asr_backend:
        return JSONResponse({"error": "No ASR backend loaded"}, status_code=503)

    audio_bytes = await audio.read()
    result = await asr_backend.transcribe(audio_bytes, language=language)

    return JSONResponse({
        "text": result.text,
        "language": result.language,
        "processing_ms": result.processing_ms,
    })


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    voice: str = Form("default"),
    speed: float = Form(1.0),
):
    """
    Synthesize speech from text.
    Backward-compatible with old /synthesize endpoint.
    """
    if not tts_backend:
        return JSONResponse({"error": "No TTS backend loaded"}, status_code=503)

    result = await tts_backend.synthesize(text, speaker=voice)

    if not result.audio:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    buf = io.BytesIO(result.audio)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Processing-Ms": str(result.duration_s),  # kept for compat
            "X-Audio-Duration-S": str(result.duration_s),
        },
    )


@app.post("/clone")
async def clone(
    reference: UploadFile = File(...),
    text: str = Form(...),
):
    """
    Zero-shot voice cloning.
    Backward-compatible with old /clone endpoint.
    """
    if not tts_backend:
        return JSONResponse({"error": "No TTS backend loaded"}, status_code=503)

    ref_bytes = await reference.read()
    result = await tts_backend.clone(ref_bytes, text)

    if not result.audio:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    buf = io.BytesIO(result.audio)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration-S": str(result.duration_s),
        },
    )


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.post("/admin/swap-model")
async def swap_model(
    service: str = Form(...),  # "asr" or "tts"
    backend_name: str = Form(...),
):
    """
    Hot-swap a backend without server restart.
    Example: service=asr, backend_name=qwen-audio
    """
    global asr_backend, tts_backend

    if service not in ("asr", "tts"):
        return JSONResponse({"error": "service must be 'asr' or 'tts'"}, status_code=400)

    if backend_name not in registry[service]["backends"]:
        available = list(registry[service]["backends"].keys())
        return JSONResponse(
            {"error": f"Unknown backend '{backend_name}'. Available: {available}"},
            status_code=400,
        )

    try:
        if service == "asr":
            asr_backend = await load_asr_backend(registry, backend_name)
            registry["asr"]["active"] = backend_name
        else:
            tts_backend = await load_tts_backend(registry, backend_name)
            registry["tts"]["active"] = backend_name

        return JSONResponse({
            "status": "ok",
            "service": service,
            "active_backend": backend_name,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket streaming ASR (pseudo-streaming via VAD chunks)
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for streaming ASR.

    Protocol:
        Client sends binary audio frames (PCM 16kHz 16-bit mono).
        Server sends JSON: {"text": "...", "is_partial": true/false}
        Client sends text "END" to signal end of stream.
    """
    await websocket.accept()

    if not asr_backend:
        await websocket.send_json({"error": "No ASR backend loaded"})
        await websocket.close()
        return

    audio_buffer = bytearray()

    try:
        while True:
            data = await websocket.receive()

            if "text" in data and data["text"] == "END":
                # Final transcription of buffered audio
                if audio_buffer:
                    result = await asr_backend.transcribe(bytes(audio_buffer))
                    await websocket.send_json({
                        "text": result.text,
                        "is_partial": False,
                        "language": result.language,
                        "processing_ms": result.processing_ms,
                    })
                await websocket.close()
                break

            if "bytes" in data:
                audio_buffer.extend(data["bytes"])

                # Pseudo-streaming: transcribe every ~2 seconds of audio (32000 samples @ 16kHz 16-bit)
                if len(audio_buffer) >= 64000:
                    result = await asr_backend.transcribe(bytes(audio_buffer))
                    await websocket.send_json({
                        "text": result.text,
                        "is_partial": True,
                        "language": result.language,
                        "processing_ms": result.processing_ms,
                    })

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200)
