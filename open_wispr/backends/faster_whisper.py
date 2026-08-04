"""
FasterWhisper ASR Backend — large-v3-turbo with int8 quantization.

VRAM: ~0.8 GB | Latency: ~200-300ms | WER: 2.3%
Uses Silero VAD for speech segmentation.
"""

import io
import time
import asyncio
from typing import Optional

import torch

from . import ASRBackend, ASRResult


class FasterWhisperASR(ASRBackend):
    def __init__(self):
        self.model = None
        self.vad_model = None
        self.model_size: str = "large-v3-turbo"
        self.compute_type: str = "int8"
        self.device: str = "cuda"

    async def load(self, **kwargs) -> None:
        self.model_size = kwargs.get("model", self.model_size)
        self.compute_type = kwargs.get("compute_type", self.compute_type)
        self.device = kwargs.get("device", self.device)

        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        # Load Silero VAD for speech detection
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        print(f"FasterWhisper ({self.model_size}, {self.compute_type}) loaded on {self.device}")

    async def transcribe(self, audio_bytes: bytes, language: str = "auto") -> ASRResult:
        t0 = time.perf_counter()

        import numpy as np
        import soundfile as sf

        # Decode audio bytes to numpy array at 16kHz
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = audio_array.astype(np.float32)

        # Run transcription
        lang = None if language == "auto" else language
        segments, info = await asyncio.to_thread(
            self.model.transcribe,
            audio_array,
            language=lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Collect segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ASRResult(
            text=text,
            is_partial=False,
            language=info.language if info else language,
            processing_ms=round(elapsed_ms, 1),
        )

    async def health(self) -> dict:
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        return {
            "status": "ok" if self.model is not None else "not_loaded",
            "backend": "faster-whisper",
            "model": self.model_size,
            "compute_type": self.compute_type,
            "device": self.device,
            "vram_gb": vram_gb,
        }
