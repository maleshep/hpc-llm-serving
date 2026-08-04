"""
Legacy Qwen2-Audio ASR Backend — wraps existing logic for A/B testing.

VRAM: ~14 GB (BF16) | Latency: varies | Full multimodal audio understanding.
"""

import io
import time
import tempfile
import asyncio
from pathlib import Path

import torch

from . import ASRBackend, ASRResult


class QwenAudioASR(ASRBackend):
    def __init__(self):
        self.processor = None
        self.model = None
        self.model_path: str = "/shared/project/hpc-llm/models/qwen2-audio-7b"
        self.device: str = "cuda"

    async def load(self, **kwargs) -> None:
        self.model_path = kwargs.get("model_path", self.model_path)
        self.device = kwargs.get("device", self.device)

        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Qwen2-Audio loaded from {self.model_path}. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    async def transcribe(self, audio_bytes: bytes, language: str = "auto") -> ASRResult:
        t0 = time.perf_counter()

        import librosa

        # Write to temp file and load with librosa
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            audio_array, sr = librosa.load(tmp_path, sr=16000)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_array},
                        {"type": "text", "text": "Transcribe this audio."},
                    ],
                }
            ]

            text_input = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=[text_input],
                audios=[audio_array],
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]
            text_out = self.processor.decode(generated, skip_special_tokens=True).strip()

            elapsed_ms = (time.perf_counter() - t0) * 1000

            return ASRResult(
                text=text_out,
                is_partial=False,
                language=language if language != "auto" else "en",
                processing_ms=round(elapsed_ms, 1),
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def health(self) -> dict:
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        return {
            "status": "ok" if self.model is not None else "not_loaded",
            "backend": "qwen-audio",
            "model": "Qwen2-Audio-7B-Instruct",
            "device": self.device,
            "vram_gb": vram_gb,
        }
