"""TTS engine abstraction layer.

KokoroEngine is the primary engine (quality-first).
CoquiEngine is the fallback for broader voice selection.

Both engines write a raw .wav, then the caller should pass the output
through core.audio_processor.remove_silence() before playback/export.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TTS_ENGINE_ENV = os.environ.get("TTS_ENGINE", "kokoro")


@dataclass
class VoiceProfile:
    id: str           # Engine-specific voice identifier
    name: str         # Human-readable display name
    engine: str       # "kokoro" or "coqui"
    lang_code: str = "en"
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name} ({self.engine})"


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def list_voices(self) -> list[VoiceProfile]:
        """Return available voice profiles for this engine."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: VoiceProfile | None = None,
        speed: float = 1.0,
    ) -> Path:
        """Synthesize text to a WAV file.

        Args:
            text: Input text to synthesize.
            output_path: Destination path for the raw .wav output.
            voice: Voice profile to use; uses engine default if None.
            speed: Playback speed multiplier (1.0 = normal).

        Returns:
            output_path on success.
        """


# ------------------------------------------------------------------
# Kokoro Engine
# ------------------------------------------------------------------

class KokoroEngine(TTSEngine):
    """High-quality TTS via the Kokoro-82M model."""

    VOICES = [
        VoiceProfile("af_heart", "Heart (EN-F)", "kokoro", "en"),
        VoiceProfile("af_bella", "Bella (EN-F)", "kokoro", "en"),
        VoiceProfile("af_sarah", "Sarah (EN-F)", "kokoro", "en"),
        VoiceProfile("am_adam", "Adam (EN-M)", "kokoro", "en"),
        VoiceProfile("am_michael", "Michael (EN-M)", "kokoro", "en"),
        VoiceProfile("bf_emma", "Emma (EN-GB-F)", "kokoro", "en-gb"),
        VoiceProfile("bm_george", "George (EN-GB-M)", "kokoro", "en-gb"),
    ]

    def __init__(self) -> None:
        self._pipeline = None

    def _get_pipeline(self, lang_code: str = "a"):
        """Lazy-load the Kokoro pipeline."""
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
                self._pipeline = KPipeline(lang_code=lang_code)
                logger.info("Kokoro pipeline loaded (lang=%s)", lang_code)
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro package not installed. Run: pip install kokoro"
                ) from exc
        return self._pipeline

    def list_voices(self) -> list[VoiceProfile]:
        return list(self.VOICES)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: VoiceProfile | None = None,
        speed: float = 1.0,
    ) -> Path:
        import numpy as np
        import soundfile as sf

        voice = voice or self.VOICES[0]
        lang_code = "a" if voice.lang_code.startswith("en") else voice.lang_code
        pipeline = self._get_pipeline(lang_code)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Kokoro synthesizing %d chars with voice=%s speed=%.1f",
            len(text), voice.id, speed,
        )

        audio_chunks = []
        sample_rate = 24000

        try:
            generator = pipeline(
                text,
                voice=voice.id,
                speed=speed,
                split_pattern=r"\n+",
            )
            for _gs, _ps, audio in generator:
                if audio is not None:
                    audio_chunks.append(audio)
        except Exception as exc:
            raise RuntimeError(f"Kokoro synthesis failed: {exc}") from exc

        if not audio_chunks:
            raise RuntimeError("Kokoro produced no audio output.")

        combined = np.concatenate(audio_chunks)
        sf.write(str(output_path), combined, sample_rate)
        logger.info("Kokoro raw output: %s (%.1f s)", output_path, len(combined) / sample_rate)
        return output_path


# ------------------------------------------------------------------
# Coqui Engine (fallback)
# ------------------------------------------------------------------

class CoquiEngine(TTSEngine):
    """Fallback TTS via Coqui TTS library."""

    DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

    VOICES = [
        VoiceProfile("tts_models/en/ljspeech/tacotron2-DDC", "LJSpeech Tacotron2", "coqui"),
        VoiceProfile("tts_models/en/ljspeech/glow-tts", "LJSpeech Glow-TTS", "coqui"),
        VoiceProfile("tts_models/en/vctk/vits", "VCTK Multi-speaker", "coqui"),
    ]

    def __init__(self) -> None:
        self._tts_instances: dict[str, object] = {}

    def _get_tts(self, model_name: str):
        if model_name not in self._tts_instances:
            try:
                from TTS.api import TTS
                self._tts_instances[model_name] = TTS(model_name=model_name)
                logger.info("Coqui TTS loaded model: %s", model_name)
            except ImportError as exc:
                raise RuntimeError(
                    "Coqui TTS not installed. Run: pip install TTS"
                ) from exc
        return self._tts_instances[model_name]

    def list_voices(self) -> list[VoiceProfile]:
        return list(self.VOICES)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: VoiceProfile | None = None,
        speed: float = 1.0,
    ) -> Path:
        voice = voice or self.VOICES[0]
        tts = self._get_tts(voice.id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Coqui synthesizing %d chars with model=%s", len(text), voice.id)
        try:
            tts.tts_to_file(text=text, file_path=str(output_path))
        except Exception as exc:
            raise RuntimeError(f"Coqui synthesis failed: {exc}") from exc

        logger.info("Coqui raw output: %s", output_path)
        return output_path


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def get_engine(engine_name: str | None = None) -> TTSEngine:
    """Return the appropriate TTS engine instance.

    Args:
        engine_name: "kokoro" or "coqui". Defaults to TTS_ENGINE env var.

    Returns:
        An initialized TTSEngine subclass instance.
    """
    name = (engine_name or TTS_ENGINE_ENV).lower()
    if name == "kokoro":
        return KokoroEngine()
    elif name == "coqui":
        return CoquiEngine()
    else:
        logger.warning("Unknown TTS engine '%s', falling back to Kokoro.", name)
        return KokoroEngine()
