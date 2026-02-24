"""
core/renderer.py — GPT-SoVITS-v2 offline batch renderer for Scriptum Veritas.

Provides ROCm-accelerated synthesis using a custom voice profile.  Audio is
written directly to .wav — no playback, no soundcard involvement.

Prerequisites
-------------
1.  ROCm-enabled PyTorch (Framework 16 / Radeon 780M = gfx1103):

        pip install torch torchaudio \\
            --index-url https://download.pytorch.org/whl/rocm6.2

2.  GPT-SoVITS-v2 repository cloned somewhere on disk:

        git clone https://github.com/RVC-Boss/GPT-SoVITS
        export GPTSOVITS_PATH=/path/to/GPT-SoVITS   # or set in profile.json

3.  scipy  (WAV writing / resampling):

        pip install scipy

Voice Profile Layout
--------------------
A profile is a directory with a profile.json and its referenced files:

    my_voice/
    ├── profile.json
    ├── ref_audio.wav          # 3-10 s clean reference clip (.mp3/.flac/.ogg also accepted)
    ├── gpt_model.ckpt         # GPT-SoVITS GPT weights
    └── sovits_model.pth       # GPT-SoVITS SoVITS weights

profile.json keys (all paths relative to the profile directory):

    {
        "gpt_model":       "gpt_model.ckpt",
        "sovits_model":    "sovits_model.pth",
        "ref_audio":       "ref_audio.wav",
        "ref_text":        "Hello, this is my reference recording.",
        "ref_language":    "en",
        "text_language":   "en",
        "sample_rate":     32000,
        "top_k":           5,
        "top_p":           1.0,
        "temperature":     1.0,
        "speed":           1.0,
        "gptsovits_path":  null          // overrides GPTSOVITS_PATH env var
    }

Public API
----------
    export_to_wav(text, output_filename, profile_path)  -> Path
    batch_render(paragraphs, output_prefix, profile_path) -> list[Path]
    unload_models()                                      -> None
"""

import gc
import json
import logging
import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROCm / device bootstrap
# Must happen BEFORE torch is imported so HIP picks up the override.
# Radeon 780M  = gfx1103  →  HSA_OVERRIDE_GFX_VERSION 11.0.3
# Radeon 860M  = gfx1152  →  HSA_OVERRIDE_GFX_VERSION 11.5.2  (set elsewhere)
# Set the env var in your launch script if you need a different value; this
# module only sets a default so it is safe to import without clobbering yours.
# ---------------------------------------------------------------------------
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.3")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not found — renderer will use CPU fallback (slow).")


# ---------------------------------------------------------------------------
# Voice Profile
# ---------------------------------------------------------------------------

@dataclass
class VoiceProfile:
    gpt_model:       Path
    sovits_model:    Path
    ref_audio:       Path
    ref_text:        str
    ref_language:    str  = "en"
    text_language:   str  = "en"
    sample_rate:     int  = 32000
    top_k:           int  = 5
    top_p:           float = 1.0
    temperature:     float = 1.0
    speed:           float = 1.0
    gptsovits_path:  Path | None = None


def load_profile(profile_path: str | Path) -> VoiceProfile:
    """Parse a voice profile directory into a VoiceProfile dataclass."""
    root = Path(profile_path).expanduser().resolve()
    json_path = root / "profile.json"
    if not json_path.exists():
        raise FileNotFoundError(f"profile.json not found in {root}")

    with open(json_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    def _resolve(key: str) -> Path:
        raw = cfg.get(key)
        if not raw:
            raise KeyError(f"profile.json missing required key: '{key}'")
        p = Path(raw)
        return p if p.is_absolute() else (root / p).resolve()

    gpt_path = _resolve("gpt_model")
    sov_path = _resolve("sovits_model")
    ref_path = _resolve("ref_audio")

    for p in (gpt_path, sov_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"Profile file not found: {p}")

    sv_raw = cfg.get("gptsovits_path")
    sv_path: Path | None = None
    if sv_raw:
        sv_path = Path(sv_raw).expanduser().resolve()
    elif os.environ.get("GPTSOVITS_PATH"):
        sv_path = Path(os.environ["GPTSOVITS_PATH"]).expanduser().resolve()

    return VoiceProfile(
        gpt_model      = gpt_path,
        sovits_model   = sov_path,
        ref_audio      = ref_path,
        ref_text       = cfg.get("ref_text", ""),
        ref_language   = cfg.get("ref_language", "en"),
        text_language  = cfg.get("text_language", "en"),
        sample_rate    = int(cfg.get("sample_rate", 32000)),
        top_k          = int(cfg.get("top_k", 5)),
        top_p          = float(cfg.get("top_p", 1.0)),
        temperature    = float(cfg.get("temperature", 1.0)),
        speed          = float(cfg.get("speed", 1.0)),
        gptsovits_path = sv_path,
    )


# ---------------------------------------------------------------------------
# Model cache  (loaded lazily, one slot — swap on profile change)
# ---------------------------------------------------------------------------

@dataclass
class _ModelCache:
    gpt_path:   Path
    sovits_path: Path
    get_tts_wav: object    # callable imported from GPT-SoVITS


_cache: _ModelCache | None = None


def _get_device() -> str:
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    logger.warning("ROCm/CUDA not available — falling back to CPU.")
    return "cpu"


def _ensure_gptsovits_on_path(profile: VoiceProfile) -> None:
    """Add GPT-SoVITS repo to sys.path if not already present."""
    sv_path = profile.gptsovits_path
    if sv_path is None:
        raise EnvironmentError(
            "GPT-SoVITS path is not configured.\n"
            "Set GPTSOVITS_PATH environment variable or add 'gptsovits_path' "
            "to profile.json."
        )
    sv_str = str(sv_path)
    if sv_str not in sys.path:
        sys.path.insert(0, sv_str)
        logger.debug("Added GPT-SoVITS to sys.path: %s", sv_str)


def _load_models(profile: VoiceProfile) -> None:
    """Load (or swap) GPT-SoVITS models into the module-level cache."""
    global _cache

    if (
        _cache is not None
        and _cache.gpt_path == profile.gpt_model
        and _cache.sovits_path == profile.sovits_model
    ):
        return  # already loaded with the right weights

    _ensure_gptsovits_on_path(profile)

    # Import inference helpers from the GPT-SoVITS repo
    try:
        from GPT_SoVITS.inference_webui import (  # type: ignore[import]
            change_gpt_weights,
            change_sovits_weights,
            get_tts_wav,
        )
    except ImportError as exc:
        raise ImportError(
            f"Could not import GPT_SoVITS.inference_webui — "
            f"check GPTSOVITS_PATH ({profile.gptsovits_path}): {exc}"
        ) from exc

    logger.info("Loading GPT weights:    %s", profile.gpt_model)
    logger.info("Loading SoVITS weights: %s", profile.sovits_model)
    change_gpt_weights(str(profile.gpt_model))
    change_sovits_weights(str(profile.sovits_model))

    _cache = _ModelCache(
        gpt_path    = profile.gpt_model,
        sovits_path = profile.sovits_model,
        get_tts_wav = get_tts_wav,
    )
    logger.info("GPT-SoVITS models loaded on device: %s", _get_device())


def unload_models() -> None:
    """Release cached models and flush GPU memory.  Call between sessions."""
    global _cache
    _cache = None
    _cleanup_gpu()
    logger.info("Models unloaded.")


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def _cleanup_gpu() -> None:
    """Flush iGPU memory; safe to call even if ROCm is not present."""
    gc.collect()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.debug("GPU memory cleared.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _collect_audio(generator: Iterator) -> tuple[int, np.ndarray]:
    """
    Consume a get_tts_wav generator and concatenate chunks.

    GPT-SoVITS-v2 yields (sample_rate, audio_chunk) pairs where audio_chunk
    is either float32 in [-1, 1] or int16.  We normalise everything to
    float32 and return a single concatenated array alongside the sample rate.
    """
    chunks: list[np.ndarray] = []
    sr_out: int | None = None

    for item in generator:
        # Some versions yield (sr, audio), others yield just audio
        if isinstance(item, tuple):
            sr, chunk = item
        else:
            sr, chunk = 32000, item

        if sr_out is None:
            sr_out = int(sr)

        arr = np.asarray(chunk)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if arr.ndim > 1:          # collapse to mono
            arr = arr.mean(axis=0) if arr.shape[0] > arr.shape[1] else arr.mean(axis=1)

        chunks.append(arr)

    if not chunks:
        raise RuntimeError("GPT-SoVITS returned no audio data.")

    return sr_out or 32000, np.concatenate(chunks)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resample using scipy if sample rates differ."""
    if src_sr == dst_sr:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        logger.warning("scipy not available — skipping resample (%d → %d).", src_sr, dst_sr)
        return audio


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Clip float32 [-1, 1] to int16 PCM."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def _ensure_wav(ref_audio: Path) -> Path:
    """Return a 16-bit 44.1 kHz mono WAV ready for GPT-SoVITS.

    If *ref_audio* is already a .wav file it is returned unchanged.
    Any other format (mp3, flac, ogg, m4a, aac …) is decoded via pydub
    and written to a sibling temp file ``<stem>_ref_converted.wav`` so the
    original is never modified.

    pydub delegates to ffmpeg, so every format ffmpeg understands is
    automatically supported — including MP3 with its variable-bitrate
    header, M4A/AAC containers, etc.

    Why not pass MP3 directly to GPT-SoVITS?
    - ``soundfile`` (used by some GPT-SoVITS builds) cannot read MP3.
    - MP3 compression artefacts degrade the voice encoder's input quality.
    - Converting to uncompressed PCM first is the safest common denominator.
    """
    if ref_audio.suffix.lower() == ".wav":
        return ref_audio

    out_path = ref_audio.with_name(ref_audio.stem + "_ref_converted.wav")
    if out_path.exists():
        logger.debug("Using cached converted reference: %s", out_path.name)
        return out_path

    logger.info(
        "Converting reference audio %s → WAV (pydub/ffmpeg)…", ref_audio.name
    )
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is required to convert non-WAV reference audio. "
            "Install it with: pip install pydub"
        ) from exc

    audio = AudioSegment.from_file(str(ref_audio))
    audio = (
        audio
        .set_channels(1)           # mono
        .set_frame_rate(44100)     # 44.1 kHz — GPT-SoVITS handles downsampling
        .set_sample_width(2)       # 16-bit PCM
    )
    audio.export(str(out_path), format="wav")
    logger.info("Saved converted reference: %s", out_path.name)
    return out_path


def _write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write 16-bit mono PCM .wav using the stdlib wave module (zero deps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = _to_int16(audio)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    logger.debug("Wrote %d samples @ %d Hz → %s", len(pcm), sample_rate, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_wav(
    text: str,
    output_filename: str | Path,
    profile_path: str | Path,
) -> Path:
    """
    Synthesise *text* using the voice profile at *profile_path* and write the
    result to *output_filename* as a 16-bit mono PCM .wav file.

    No audio is sent to speakers.  GPU memory is flushed after the file is
    saved.

    Parameters
    ----------
    text:
        The text to synthesise.  May contain multiple sentences.
    output_filename:
        Destination .wav path (created with parent dirs as needed).
    profile_path:
        Path to a voice profile directory containing profile.json.

    Returns
    -------
    Path to the written .wav file.
    """
    output_path = Path(output_filename).expanduser().resolve()
    profile = load_profile(profile_path)
    _load_models(profile)

    assert _cache is not None  # guaranteed by _load_models

    logger.info("Synthesising → %s", output_path.name)

    try:
        ref_wav = _ensure_wav(profile.ref_audio)

        gen = _cache.get_tts_wav(
            ref_wav_path      = str(ref_wav),
            prompt_text       = profile.ref_text,
            prompt_language   = profile.ref_language,
            text              = text,
            text_language     = profile.text_language,
            top_k             = profile.top_k,
            top_p             = profile.top_p,
            temperature       = profile.temperature,
            speed             = profile.speed,
        )

        model_sr, audio = _collect_audio(gen)

        # Resample to the profile's target rate if they differ
        if model_sr != profile.sample_rate:
            logger.debug("Resampling %d Hz → %d Hz", model_sr, profile.sample_rate)
            audio = _resample(audio, model_sr, profile.sample_rate)

        _write_wav(audio, profile.sample_rate, output_path)

    finally:
        # Always flush GPU memory — prevents iGPU from hanging on next call
        _cleanup_gpu()

    logger.info("Saved: %s  (%.1f s)", output_path, len(audio) / profile.sample_rate)
    return output_path


def batch_render(
    paragraphs: list[str],
    output_prefix: str | Path,
    profile_path: str | Path,
    *,
    skip_empty: bool = True,
) -> list[Path]:
    """
    Synthesise a list of paragraphs and save them as sequentially numbered
    .wav files.

    Example
    -------
        paths = batch_render(
            paragraphs   = ["Chapter one.", "The darkness fell.", ...],
            output_prefix= "~/audio/chapter_1",
            profile_path = "~/voices/my_voice/",
        )
        # writes: chapter_1_01.wav, chapter_1_02.wav, ...

    Parameters
    ----------
    paragraphs:
        List of text strings, one file per entry.
    output_prefix:
        Path prefix for output files.  A two-digit zero-padded index and
        '.wav' are appended automatically.
    profile_path:
        Voice profile directory.
    skip_empty:
        If True (default) silently skip blank/whitespace-only paragraphs.

    Returns
    -------
    List of Path objects for every file that was written.
    """
    prefix = Path(output_prefix).expanduser().resolve()
    profile = load_profile(profile_path)     # validate once up front
    _load_models(profile)                    # load models once for the batch

    written: list[Path] = []
    index   = 1

    for para in paragraphs:
        if skip_empty and not para.strip():
            continue

        out_path = prefix.parent / f"{prefix.name}_{index:02d}.wav"

        try:
            export_to_wav(para, out_path, profile_path)
            written.append(out_path)
        except Exception as exc:
            logger.error("Paragraph %02d failed: %s — %s", index, out_path.name, exc)
            # Continue with remaining paragraphs rather than aborting the batch

        index += 1

    logger.info("Batch complete: %d/%d files written.", len(written), index - 1)
    return written
