"""
OpenWispr Backend Interfaces — Model-agnostic ASR & TTS abstractions.

All backends implement these ABCs so the server can hot-swap models
via configuration without code changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class ASRResult:
    text: str
    is_partial: bool = False
    language: Optional[str] = None
    processing_ms: Optional[float] = None


@dataclass
class TTSResult:
    audio: bytes
    sample_rate: int = 22050
    duration_s: float = 0.0


class ASRBackend(ABC):
    """Abstract base class for speech recognition backends."""

    @abstractmethod
    async def load(self, **kwargs) -> None:
        """Load model into memory (GPU/CPU)."""
        ...

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "auto") -> ASRResult:
        """Transcribe audio bytes to text."""
        ...

    @abstractmethod
    async def health(self) -> dict:
        """Return health/status info including VRAM usage."""
        ...


class TTSBackend(ABC):
    """Abstract base class for text-to-speech backends."""

    @abstractmethod
    async def load(self, **kwargs) -> None:
        """Load model into memory (GPU/CPU)."""
        ...

    @abstractmethod
    async def synthesize(
        self, text: str, speaker: str = "default", language: str = "English"
    ) -> TTSResult:
        """Synthesize text to audio bytes."""
        ...

    @abstractmethod
    async def clone(
        self, reference_audio: bytes, text: str, language: str = "English"
    ) -> TTSResult:
        """Zero-shot voice cloning from reference audio."""
        ...

    @abstractmethod
    async def health(self) -> dict:
        """Return health/status info including VRAM usage."""
        ...
