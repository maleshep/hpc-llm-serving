"""
Legacy CosyVoice3 TTS Backend — wraps existing logic as fallback.

VRAM: ~2 GB | Latency: variable (slow) | Built-in voices + zero-shot cloning.
"""

import io
import time
import tempfile
import asyncio
from pathlib import Path

import torch
import torchaudio

from . import TTSBackend, TTSResult


class CosyVoiceTTS(TTSBackend):
    def __init__(self):
        self.cosyvoice = None
        self.model_path: str = "/shared/project/hpc-llm/models/cosyvoice3-0.5b"
        self.sample_rate: int = 22050

    async def load(self, **kwargs) -> None:
        self.model_path = kwargs.get("model_path", self.model_path)

        import sys
        sys.path.insert(0, str(Path(self.model_path).parent / "CosyVoice"))

        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.cosyvoice = CosyVoice2(self.model_path, load_jit=False, load_trt=False)
        print(f"CosyVoice3 loaded from {self.model_path}. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    async def synthesize(
        self, text: str, speaker: str = "default", language: str = "English"
    ) -> TTSResult:
        t0 = time.perf_counter()

        audio_chunks = []
        chunks = await asyncio.to_thread(
            lambda: list(self.cosyvoice.inference_sft(text, speaker, stream=False))
        )
        for chunk in chunks:
            audio_chunks.append(chunk["tts_speech"])

        if not audio_chunks:
            return TTSResult(audio=b"", sample_rate=self.sample_rate, duration_s=0.0)

        audio = torch.cat(audio_chunks, dim=-1)

        buf = io.BytesIO()
        torchaudio.save(buf, audio.unsqueeze(0).cpu(), self.sample_rate, format="wav")
        audio_bytes = buf.getvalue()

        duration_s = audio.shape[-1] / self.sample_rate

        return TTSResult(
            audio=audio_bytes,
            sample_rate=self.sample_rate,
            duration_s=round(duration_s, 2),
        )

    async def clone(
        self, reference_audio: bytes, text: str, language: str = "English"
    ) -> TTSResult:
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(reference_audio)
            ref_path = tmp.name

        try:
            audio_chunks = []
            chunks = await asyncio.to_thread(
                lambda: list(self.cosyvoice.inference_zero_shot(
                    text, "", ref_path, stream=False
                ))
            )
            for chunk in chunks:
                audio_chunks.append(chunk["tts_speech"])

            if not audio_chunks:
                return TTSResult(audio=b"", sample_rate=self.sample_rate, duration_s=0.0)

            audio = torch.cat(audio_chunks, dim=-1)

            buf = io.BytesIO()
            torchaudio.save(buf, audio.unsqueeze(0).cpu(), self.sample_rate, format="wav")
            audio_bytes = buf.getvalue()

            duration_s = audio.shape[-1] / self.sample_rate

            return TTSResult(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                duration_s=round(duration_s, 2),
            )
        finally:
            Path(ref_path).unlink(missing_ok=True)

    async def health(self) -> dict:
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        return {
            "status": "ok" if self.cosyvoice is not None else "not_loaded",
            "backend": "cosyvoice",
            "model": "Fun-CosyVoice3-0.5B",
            "sample_rate": self.sample_rate,
            "vram_gb": vram_gb,
        }
