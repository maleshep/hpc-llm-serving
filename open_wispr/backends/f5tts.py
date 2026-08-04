"""
F5-TTS Backend — Zero-shot voice cloning with high quality.

VRAM: ~1.5-2 GB | Latency: 1-3s | Zero-shot cloning supported.
"""

import io
import time
import tempfile
import asyncio
from pathlib import Path

import torch

from . import TTSBackend, TTSResult


class F5TTSBackend(TTSBackend):
    def __init__(self):
        self.model = None
        self.model_name: str = "SWivid/F5-TTS"
        self.device: str = "cuda"
        self.sample_rate: int = 24000

    async def load(self, **kwargs) -> None:
        self.model_name = kwargs.get("model", self.model_name)
        self.device = kwargs.get("device", self.device)

        from f5_tts.api import F5TTS

        self.model = F5TTS(model_type="F5-TTS", device=self.device)
        print(f"F5-TTS loaded ({self.model_name}) on {self.device}")

    async def synthesize(
        self, text: str, speaker: str = "default", language: str = "English"
    ) -> TTSResult:
        t0 = time.perf_counter()

        # F5-TTS generates audio from text; for non-clone mode we use a default reference
        # If no reference is available, use basic inference
        audio_array, sr, _ = await asyncio.to_thread(
            self.model.infer,
            ref_file="",  # empty = use built-in default
            ref_text="",
            gen_text=text,
        )

        self.sample_rate = sr

        # Convert to WAV bytes
        import torchaudio
        if not isinstance(audio_array, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_array).float()
        else:
            audio_tensor = audio_array.float()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        buf = io.BytesIO()
        torchaudio.save(buf, audio_tensor.cpu(), sr, format="wav")
        audio_bytes = buf.getvalue()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration_s = audio_tensor.shape[-1] / sr

        return TTSResult(
            audio=audio_bytes,
            sample_rate=sr,
            duration_s=round(duration_s, 2),
        )

    async def clone(
        self, reference_audio: bytes, text: str, language: str = "English"
    ) -> TTSResult:
        t0 = time.perf_counter()

        # Write reference to temp file (F5-TTS needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(reference_audio)
            ref_path = tmp.name

        try:
            audio_array, sr, _ = await asyncio.to_thread(
                self.model.infer,
                ref_file=ref_path,
                ref_text="",  # auto-transcribe reference
                gen_text=text,
            )

            self.sample_rate = sr

            import torchaudio
            if not isinstance(audio_array, torch.Tensor):
                audio_tensor = torch.from_numpy(audio_array).float()
            else:
                audio_tensor = audio_array.float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            buf = io.BytesIO()
            torchaudio.save(buf, audio_tensor.cpu(), sr, format="wav")
            audio_bytes = buf.getvalue()

            elapsed_ms = (time.perf_counter() - t0) * 1000
            duration_s = audio_tensor.shape[-1] / sr

            return TTSResult(
                audio=audio_bytes,
                sample_rate=sr,
                duration_s=round(duration_s, 2),
            )
        finally:
            Path(ref_path).unlink(missing_ok=True)

    async def health(self) -> dict:
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        return {
            "status": "ok" if self.model is not None else "not_loaded",
            "backend": "f5-tts",
            "model": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "vram_gb": vram_gb,
        }
