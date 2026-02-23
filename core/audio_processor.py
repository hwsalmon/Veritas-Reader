"""Audio post-processing utilities.

Handles silence removal, padding, and export to WAV/MP3.
Requires ffmpeg to be installed and on PATH.
"""

import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence

logger = logging.getLogger(__name__)

# Defaults — also exposed in config/settings.py for user override
DEFAULT_SILENCE_THRESH_DB = -40
DEFAULT_MIN_SILENCE_MS = 300
DEFAULT_KEEP_SILENCE_MS = 50


class AudioProcessorError(Exception):
    """Raised when audio processing fails."""


def remove_silence(
    input_path: Path,
    output_path: Path,
    silence_thresh_db: int = DEFAULT_SILENCE_THRESH_DB,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    keep_silence_ms: int = DEFAULT_KEEP_SILENCE_MS,
) -> Path:
    """Strip long silences from a WAV file and write a cleaned version.

    The raw input file is left untouched. The processed result is
    written to output_path.

    Args:
        input_path: Path to the source .wav file.
        output_path: Destination path for processed .wav file.
        silence_thresh_db: Silence threshold in dBFS (default -40).
        min_silence_ms: Minimum silence length to remove (default 300 ms).
        keep_silence_ms: Short silence padding added between rejoined chunks
                         for naturalness (default 50 ms).

    Returns:
        output_path on success.
    """
    logger.debug(
        "remove_silence: %s → %s (thresh=%d dBFS, min=%d ms, pad=%d ms)",
        input_path,
        output_path,
        silence_thresh_db,
        min_silence_ms,
        keep_silence_ms,
    )

    try:
        audio = AudioSegment.from_wav(str(input_path))
    except Exception as exc:
        raise AudioProcessorError(f"Failed to load audio file: {exc}") from exc

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms,
    )

    if not chunks:
        logger.warning("No audio chunks found after silence removal — copying raw file.")
        audio.export(str(output_path), format="wav")
        return output_path

    logger.debug("Rejoining %d chunk(s).", len(chunks))
    silence_pad = AudioSegment.silent(duration=keep_silence_ms)
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined = combined + silence_pad + chunk

    combined.export(str(output_path), format="wav")
    logger.info("Processed audio written to %s (%.1f s)", output_path, len(combined) / 1000)
    return output_path


def export_wav(audio: AudioSegment, path: Path) -> Path:
    """Export an AudioSegment to a WAV file.

    Args:
        audio: pydub AudioSegment to export.
        path: Destination .wav path.

    Returns:
        path on success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(path), format="wav")
    logger.debug("Exported WAV: %s", path)
    return path


def export_mp3(audio: AudioSegment, path: Path, bitrate: str = "192k") -> Path:
    """Export an AudioSegment to an MP3 file.

    Args:
        audio: pydub AudioSegment to export.
        path: Destination .mp3 path.
        bitrate: MP3 bitrate string (default '192k').

    Returns:
        path on success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(path), format="mp3", bitrate=bitrate)
    logger.debug("Exported MP3 @ %s: %s", bitrate, path)
    return path


def load_audio(path: Path) -> AudioSegment:
    """Load any pydub-supported audio file into an AudioSegment."""
    suffix = path.suffix.lower().lstrip(".")
    fmt = suffix if suffix in ("wav", "mp3", "ogg", "flac") else "wav"
    try:
        return AudioSegment.from_file(str(path), format=fmt)
    except Exception as exc:
        raise AudioProcessorError(f"Cannot load {path}: {exc}") from exc
