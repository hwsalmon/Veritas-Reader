"""Audio post-processing utilities.

Full narration quality pipeline (applied in order):
  1. remove_silence      — strip dead air, re-pad with short natural gaps
  2. apply_eq            — 80 Hz high-pass + gentle 4 kHz presence boost
  3. apply_compression   — soft-knee dynamic range compression
  4. normalize_loudness  — peak-normalise to target dBFS (-1.5 dBFS headroom)
  5. upsample            — 24 kHz → 44.1 kHz for export compatibility

Each step is also available as a standalone function.
Requires ffmpeg on PATH.  scipy is required for EQ and resampling.
"""

import logging
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

logger = logging.getLogger(__name__)

# Defaults exposed in config/settings.py for user override
DEFAULT_SILENCE_THRESH_DB = -40
DEFAULT_MIN_SILENCE_MS = 300
DEFAULT_KEEP_SILENCE_MS = 50

TARGET_SAMPLE_RATE = 44100   # Hz — standard for audio export
TARGET_PEAK_DBFS   = -1.5    # dBFS headroom after normalisation


class AudioProcessorError(Exception):
    """Raised when audio processing fails."""


# ---------------------------------------------------------------------------
# Step 1 — Silence removal
# ---------------------------------------------------------------------------

def remove_silence(
    input_path: Path,
    output_path: Path,
    silence_thresh_db: int = DEFAULT_SILENCE_THRESH_DB,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    keep_silence_ms: int = DEFAULT_KEEP_SILENCE_MS,
) -> Path:
    """Strip long silences and write a cleaned WAV.

    The raw input is left untouched; the result goes to *output_path*.
    """
    logger.debug(
        "remove_silence: %s → %s (thresh=%d dBFS, min=%d ms, pad=%d ms)",
        input_path, output_path,
        silence_thresh_db, min_silence_ms, keep_silence_ms,
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
        logger.warning("No audio chunks after silence removal — copying raw file.")
        audio.export(str(output_path), format="wav")
        return output_path

    silence_pad = AudioSegment.silent(duration=keep_silence_ms)
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined = combined + silence_pad + chunk

    combined.export(str(output_path), format="wav")
    logger.debug("Silence removed: %s (%.1f s)", output_path, len(combined) / 1000)
    return output_path


# ---------------------------------------------------------------------------
# Step 2 — EQ  (high-pass + presence boost)
# ---------------------------------------------------------------------------

def apply_eq(
    input_path: Path,
    output_path: Path,
    highpass_hz: float = 80.0,
    presence_hz: float = 4000.0,
    presence_gain_db: float = 2.5,
) -> Path:
    """Apply a high-pass filter and gentle presence boost.

    * High-pass at *highpass_hz* removes subsonic/low rumble.
    * Peaking EQ at *presence_hz* adds clarity and air to the voice.

    Requires scipy.
    """
    try:
        from scipy import signal as sig
        import soundfile as sf
    except ImportError as exc:
        raise AudioProcessorError("scipy and soundfile are required for EQ") from exc

    samples, sr = sf.read(str(input_path))
    mono = samples.ndim == 1

    # --- High-pass (4th-order Butterworth) ---
    b_hp, a_hp = sig.butter(4, highpass_hz / (sr / 2), btype="high")
    filtered = sig.filtfilt(b_hp, a_hp, samples, axis=0)

    # --- Presence peak (2nd-order peaking EQ via bilinear transform) ---
    # Build analog prototype and convert to digital
    Q = 1.5
    w0 = 2 * np.pi * presence_hz / sr
    A = 10 ** (presence_gain_db / 40)          # linear amplitude gain
    alpha = np.sin(w0) / (2 * Q)

    b_eq = np.array([
        1 + alpha * A,
        -2 * np.cos(w0),
        1 - alpha * A,
    ])
    a_eq = np.array([
        1 + alpha / A,
        -2 * np.cos(w0),
        1 - alpha / A,
    ])
    boosted = sig.filtfilt(b_eq, a_eq, filtered, axis=0)

    # Prevent clipping introduced by the boost
    peak = np.max(np.abs(boosted))
    if peak > 1.0:
        boosted /= peak

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), boosted, sr)
    logger.debug("EQ applied: hp=%.0f Hz, presence=%.0f Hz +%.1f dB → %s",
                 highpass_hz, presence_hz, presence_gain_db, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 3 — Dynamic range compression
# ---------------------------------------------------------------------------

def apply_compression(
    input_path: Path,
    output_path: Path,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    attack_ms: float = 8.0,
    release_ms: float = 120.0,
    makeup_gain_db: float = 4.0,
) -> Path:
    """Soft-knee feed-forward dynamic range compressor.

    Reduces the loudness gap between quiet and loud passages, making
    narration easier to follow at any volume.  A makeup gain is applied
    after compression to restore perceived loudness.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioProcessorError("soundfile is required for compression") from exc

    samples, sr = sf.read(str(input_path))
    # Work in mono-equivalent for gain computation; apply same gain to stereo
    mono = samples if samples.ndim == 1 else samples.mean(axis=1)

    threshold_lin = 10 ** (threshold_db / 20)
    makeup_lin    = 10 ** (makeup_gain_db / 20)
    attack_coef   = np.exp(-1.0 / (sr * attack_ms  / 1000))
    release_coef  = np.exp(-1.0 / (sr * release_ms / 1000))
    knee_width_db = 6.0

    gain_db = np.zeros(len(mono))
    env = 0.0

    for i, x in enumerate(np.abs(mono)):
        # Peak envelope follower
        if x > env:
            env = attack_coef  * env + (1 - attack_coef)  * x
        else:
            env = release_coef * env + (1 - release_coef) * x

        level_db = 20 * np.log10(max(env, 1e-9))
        over_db  = level_db - threshold_db

        # Soft knee
        if over_db < -(knee_width_db / 2):
            cs = 0.0
        elif over_db < (knee_width_db / 2):
            cs = (over_db + knee_width_db / 2) ** 2 / (2 * knee_width_db)
        else:
            cs = over_db

        gain_db[i] = -cs * (1 - 1 / ratio)

    gain_lin = (10 ** (gain_db / 20)) * makeup_lin
    if samples.ndim == 1:
        compressed = samples * gain_lin
    else:
        compressed = samples * gain_lin[:, np.newaxis]

    # Hard ceiling at ±1.0
    compressed = np.clip(compressed, -1.0, 1.0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(output_path), compressed, sr)
    logger.debug("Compression applied: thresh=%.0f dB ratio=%.1f:1 makeup=+%.1f dB → %s",
                 threshold_db, ratio, makeup_gain_db, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 4 — Loudness / peak normalisation
# ---------------------------------------------------------------------------

def normalize_loudness(
    input_path: Path,
    output_path: Path,
    target_peak_dbfs: float = TARGET_PEAK_DBFS,
) -> Path:
    """Peak-normalise audio so the loudest sample sits at *target_peak_dbfs*.

    Uses pydub — no extra dependencies.
    """
    try:
        audio = AudioSegment.from_file(str(input_path))
    except Exception as exc:
        raise AudioProcessorError(f"Cannot load {input_path}: {exc}") from exc

    headroom = target_peak_dbfs - audio.max_dBFS
    normalised = audio.apply_gain(headroom)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    normalised.export(str(output_path), format="wav")
    logger.debug(
        "Normalised: peak %.1f dBFS → %.1f dBFS → %s",
        audio.max_dBFS, normalised.max_dBFS, output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# Step 5 — Upsample to 44.1 kHz
# ---------------------------------------------------------------------------

def upsample(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Resample audio to *target_sr* using a high-quality polyphase filter.

    Requires scipy.  If the file is already at target_sr, it is copied as-is.
    """
    try:
        from scipy import signal as sig
        import soundfile as sf
    except ImportError as exc:
        raise AudioProcessorError("scipy and soundfile are required for resampling") from exc

    samples, sr = sf.read(str(input_path))

    if sr == target_sr:
        import shutil
        shutil.copy2(str(input_path), str(output_path))
        logger.debug("upsample: already at %d Hz, copied.", target_sr)
        return output_path

    num_samples = int(round(len(samples) * target_sr / sr))
    if samples.ndim == 1:
        resampled = sig.resample_poly(samples, target_sr, sr)
    else:
        resampled = np.column_stack(
            [sig.resample_poly(samples[:, c], target_sr, sr)
             for c in range(samples.shape[1])]
        )
    resampled = resampled[:num_samples] if len(resampled) > num_samples else resampled

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), resampled, target_sr)
    logger.debug("Resampled %d Hz → %d Hz → %s", sr, target_sr, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Full quality pipeline
# ---------------------------------------------------------------------------

def process_audio_pipeline(
    raw_path: Path,
    output_path: Path,
    silence_thresh_db: int = DEFAULT_SILENCE_THRESH_DB,
    min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
) -> Path:
    """Run the full narration quality chain on *raw_path*.

    Steps: silence removal → EQ → compression → normalisation → upsample.
    Intermediate files are written to the same directory as *output_path*
    with step suffixes, then the final result is written to *output_path*.
    """
    stem = Path(output_path).stem
    work_dir = Path(output_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    def _tmp(suffix: str) -> Path:
        return work_dir / f"{stem}_{suffix}.wav"

    logger.info("Audio pipeline start: %s", raw_path)

    p1 = remove_silence(raw_path,  _tmp("s1_silence"),
                        silence_thresh_db, min_silence_ms)
    p2 = apply_eq(p1,              _tmp("s2_eq"))
    p3 = apply_compression(p2,     _tmp("s3_comp"))
    p4 = normalize_loudness(p3,    _tmp("s4_norm"))
    p5 = upsample(p4,              output_path)

    # Clean up intermediate files
    for tmp in [p1, p2, p3, p4]:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    logger.info("Audio pipeline complete: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Legacy helpers (used by export)
# ---------------------------------------------------------------------------

def export_wav(audio: AudioSegment, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(path), format="wav")
    logger.debug("Exported WAV: %s", path)
    return path


def export_mp3(audio: AudioSegment, path: Path, bitrate: str = "192k") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(path), format="mp3", bitrate=bitrate)
    logger.debug("Exported MP3 @ %s: %s", bitrate, path)
    return path


def load_audio(path: Path) -> AudioSegment:
    suffix = path.suffix.lower().lstrip(".")
    fmt = suffix if suffix in ("wav", "mp3", "ogg", "flac") else "wav"
    try:
        return AudioSegment.from_file(str(path), format=fmt)
    except Exception as exc:
        raise AudioProcessorError(f"Cannot load {path}: {exc}") from exc
