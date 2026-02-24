"""TTS engine abstraction layer.

KokoroEngine is the primary engine (quality-first).
CoquiEngine is the fallback for broader voice selection.

Both engines write a raw .wav, then the caller should pass the output
through core.audio_processor.remove_silence() before playback/export.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TTS_ENGINE_ENV = os.environ.get("TTS_ENGINE", "kokoro")

# ---------------------------------------------------------------------------
# Narration pacing defaults — silence (ms) inserted AFTER each segment type.
# These are the fall-through defaults; callers pass their own values.
# ---------------------------------------------------------------------------
DEFAULT_SPEED         = 0.9
DEFAULT_SENTENCE_MS   = 500
DEFAULT_MDASH_MS      = 500
DEFAULT_PARAGRAPH_MS  = 1000


def _modulate_speeds(
    tokens: list[tuple[str, str]],
    base_speed: float,
) -> list[float]:
    """Return a per-segment speed that mimics natural speech cadence.

    Rules (all multiplicative, capped to [0.5, 1.4]):

    Phrase-final lengthening
      • Last segment before paragraph  →  ×0.88  (strong closure)
      • Last segment before sentence   →  ×0.93  (gentle slowdown)
      • Last segment before mdash      →  ×0.94  (slight suspense)

    Paragraph opening
      • First segment of each paragraph → ×0.96  (thoughtful start)

    Post-mdash re-entry
      • First segment after a mdash    → ×0.96  (deliberate return)

    Mid-paragraph cadence oscillation
      • ±3% sine wave across segment index within each paragraph.
        Creates a natural ebb-and-flow rather than robotic uniformity.
    """
    import math

    speeds: list[float] = []
    para_idx = 0       # position within current paragraph
    post_mdash = False

    for chunk, pause_type in tokens:
        s = base_speed

        # Phrase-final lengthening
        if pause_type == "paragraph":
            s *= 0.88
        elif pause_type == "sentence":
            s *= 0.93
        elif pause_type == "mdash":
            s *= 0.94

        # Paragraph-initial — first sentence feels unhurried
        if para_idx == 0:
            s *= 0.96

        # Post-mdash re-entry
        if post_mdash:
            s *= 0.96

        # Gentle cadence oscillation (±3%)
        s *= 1.0 + 0.03 * math.sin(para_idx * 1.2)

        speeds.append(round(max(0.5, min(1.4, s)), 3))

        # Advance state
        post_mdash = (pause_type == "mdash")
        para_idx = 0 if pause_type == "paragraph" else para_idx + 1

    return speeds


def _tokenize_for_pacing(text: str) -> list[tuple[str, str]]:
    """Split *text* into (segment, pause_type) pairs.

    pause_type is one of: 'paragraph', 'sentence', 'mdash', 'none'
    The pause is inserted *after* the segment during assembly.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split on paragraph breaks, em-dashes, or sentence-ending whitespace.
    # Sentence split requires a capital letter / quote after the space so that
    # abbreviations like "Mr. Smith" or "e.g. something" are NOT split.
    # The capturing group keeps separators in the result list so we can
    # inspect them (re.split alternates: [chunk, sep, chunk, sep, …, chunk]).
    pattern = r"""(\n\n+|\s*(?:--|—)\s*|(?<=[.!?])\s+(?=[A-Z"'\u201c\u2018]))"""
    parts = re.split(pattern, text)

    result: list[tuple[str, str]] = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if not chunk:
            continue

        pause = "none"
        if i + 1 < len(parts):
            sep = parts[i + 1]
            if "\n\n" in sep or sep.count("\n") >= 2:
                pause = "paragraph"
            elif "--" in sep or "—" in sep:
                pause = "mdash"
            else:
                pause = "sentence"

        result.append((chunk, pause))

    return result


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
# Kokoro Engine (via kokoro-onnx)
# ------------------------------------------------------------------

class KokoroEngine(TTSEngine):
    """High-quality TTS via the Kokoro-82M ONNX model."""

    VOICES = [
        VoiceProfile("af_heart", "Heart (EN-F)", "kokoro", "en-us"),
        VoiceProfile("af_bella", "Bella (EN-F)", "kokoro", "en-us"),
        VoiceProfile("af_sarah", "Sarah (EN-F)", "kokoro", "en-us"),
        VoiceProfile("am_adam", "Adam (EN-M)", "kokoro", "en-us"),
        VoiceProfile("am_michael", "Michael (EN-M)", "kokoro", "en-us"),
        VoiceProfile("bf_emma", "Emma (EN-GB-F)", "kokoro", "en-gb"),
        VoiceProfile("bm_george", "George (EN-GB-M)", "kokoro", "en-gb"),
    ]

    def __init__(self) -> None:
        self._kokoro = None

    @staticmethod
    def _model_dir() -> Path:
        from platformdirs import user_data_dir
        return Path(user_data_dir("veritas_reader")) / "models"

    def _get_kokoro(self):
        """Lazy-load the Kokoro ONNX instance."""
        if self._kokoro is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro-onnx not installed. Run: pip install kokoro-onnx"
                ) from exc

            model_dir = self._model_dir()
            voices_path = model_dir / "voices-v1.0.bin"

            # Prefer higher-quality models when available
            for candidate in ("kokoro-v1.0.fp16.onnx", "kokoro-v1.0.int8.onnx"):
                model_path = model_dir / candidate
                if model_path.exists():
                    logger.info("Using Kokoro model: %s", candidate)
                    break
            else:
                raise RuntimeError(
                    f"No Kokoro model file found in {model_dir}. "
                    "Download kokoro-v1.0.fp16.onnx (or int8) and voices-v1.0.bin from "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases"
                )

            if not voices_path.exists():
                raise RuntimeError(f"voices-v1.0.bin missing in {model_dir}.")

            import onnxruntime as rt
            opts = rt.SessionOptions()
            opts.intra_op_num_threads = 8
            opts.inter_op_num_threads = 4
            opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            session = rt.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._kokoro = Kokoro.from_session(session, str(voices_path))
            logger.info(
                "Kokoro ONNX loaded from %s (8 intra-op / 4 inter-op threads)",
                model_dir,
            )
        return self._kokoro

    def list_voices(self) -> list[VoiceProfile]:
        return list(self.VOICES)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: VoiceProfile | None = None,
        speed: float = DEFAULT_SPEED,
        pause_sentence_ms: int = DEFAULT_SENTENCE_MS,
        pause_mdash_ms: int = DEFAULT_MDASH_MS,
        pause_paragraph_ms: int = DEFAULT_PARAGRAPH_MS,
    ) -> Path:
        import numpy as np
        import soundfile as sf

        voice = voice or self.VOICES[0]
        kokoro = self._get_kokoro()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from core.text_preprocessor import preprocess
        text = preprocess(text)

        tokens = _tokenize_for_pacing(text)
        local_speeds = _modulate_speeds(tokens, speed)
        logger.info(
            "Kokoro synthesizing %d segment(s) | voice=%s base_speed=%.2f "
            "pauses: sentence=%dms mdash=%dms para=%dms",
            len(tokens), voice.id, speed,
            pause_sentence_ms, pause_mdash_ms, pause_paragraph_ms,
        )

        pause_map = {
            "sentence":  pause_sentence_ms,
            "mdash":     pause_mdash_ms,
            "paragraph": pause_paragraph_ms,
            "none":      0,
        }

        segments: list[np.ndarray] = []
        sample_rate: int | None = None

        for (chunk, pause_type), local_speed in zip(tokens, local_speeds):
            # Skip segments too short to synthesise (punctuation-only fragments, etc.)
            if len(chunk.strip()) < 3:
                continue

            try:
                samples, sr = kokoro.create(
                    text=chunk,
                    voice=voice.id,
                    speed=local_speed,
                    lang=voice.lang_code,
                )
            except Exception as exc:
                raise RuntimeError(f"Kokoro synthesis failed on segment: {exc}") from exc

            if samples is None or len(samples) == 0:
                logger.warning("Kokoro returned empty audio for segment: %r — skipping.", chunk[:60])
                continue

            if sample_rate is None:
                sample_rate = sr

            segments.append(samples)

            pause_ms = pause_map.get(pause_type, 0)
            if pause_ms > 0:
                silence = np.zeros(int(sr * pause_ms / 1000), dtype=samples.dtype)
                segments.append(silence)

        if not segments:
            raise RuntimeError("No audio produced — text may have been empty after tokenisation.")

        combined = np.concatenate(segments)
        sf.write(str(output_path), combined, sample_rate)
        logger.info(
            "Kokoro raw output: %s (%.1f s, %d segment(s))",
            output_path, len(combined) / sample_rate, len(tokens),
        )
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
