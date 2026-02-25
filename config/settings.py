"""Application-wide settings backed by QSettings.

Usage:
    from config.settings import AppSettings
    settings = AppSettings()
    settings.last_model = "llama3.2:latest"
    model = settings.last_model
"""

import logging
from pathlib import Path

from platformdirs import user_data_dir, user_config_dir
from PyQt6.QtCore import QSettings

logger = logging.getLogger(__name__)

APP_NAME = "VeritasReader"
APP_ORG = "VeritasReader"


class AppSettings:
    """Thin wrapper around QSettings with typed property accessors."""

    def __init__(self) -> None:
        self._qs = QSettings(APP_ORG, APP_NAME)

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    @property
    def ollama_host(self) -> str:
        import os
        return os.environ.get("OLLAMA_HOST", self._qs.value("ollama/host", "http://localhost:11434"))

    @ollama_host.setter
    def ollama_host(self, value: str) -> None:
        self._qs.setValue("ollama/host", value)

    @property
    def last_model(self) -> str:
        return self._qs.value("ollama/last_model", "")

    @last_model.setter
    def last_model(self, value: str) -> None:
        self._qs.setValue("ollama/last_model", value)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    @property
    def tts_engine(self) -> str:
        import os
        return os.environ.get("TTS_ENGINE", self._qs.value("tts/engine", "kokoro"))

    @tts_engine.setter
    def tts_engine(self, value: str) -> None:
        self._qs.setValue("tts/engine", value)

    @property
    def last_voice(self) -> str:
        return self._qs.value("tts/last_voice", "")

    @last_voice.setter
    def last_voice(self, value: str) -> None:
        self._qs.setValue("tts/last_voice", value)

    @property
    def tts_speed(self) -> float:
        return float(self._qs.value("tts/speed", 0.9))

    @tts_speed.setter
    def tts_speed(self, value: float) -> None:
        self._qs.setValue("tts/speed", float(value))

    @property
    def pause_sentence_ms(self) -> int:
        return int(self._qs.value("tts/pause_sentence_ms", 500))

    @pause_sentence_ms.setter
    def pause_sentence_ms(self, value: int) -> None:
        self._qs.setValue("tts/pause_sentence_ms", int(value))

    @property
    def pause_mdash_ms(self) -> int:
        return int(self._qs.value("tts/pause_mdash_ms", 500))

    @pause_mdash_ms.setter
    def pause_mdash_ms(self, value: int) -> None:
        self._qs.setValue("tts/pause_mdash_ms", int(value))

    @property
    def pause_paragraph_ms(self) -> int:
        return int(self._qs.value("tts/pause_paragraph_ms", 1000))

    @pause_paragraph_ms.setter
    def pause_paragraph_ms(self, value: int) -> None:
        self._qs.setValue("tts/pause_paragraph_ms", int(value))

    # ------------------------------------------------------------------
    # Audio post-processing
    # ------------------------------------------------------------------

    @property
    def silence_thresh_db(self) -> int:
        return int(self._qs.value("audio/silence_thresh_db", -40))

    @silence_thresh_db.setter
    def silence_thresh_db(self, value: int) -> None:
        self._qs.setValue("audio/silence_thresh_db", value)

    @property
    def min_silence_ms(self) -> int:
        return int(self._qs.value("audio/min_silence_ms", 300))

    @min_silence_ms.setter
    def min_silence_ms(self, value: int) -> None:
        self._qs.setValue("audio/min_silence_ms", value)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def vault_root(self) -> Path:
        default = Path.home() / "Documents" / "VeritasVault"
        raw = self._qs.value("paths/vault_root", str(default))
        return Path(raw)

    @vault_root.setter
    def vault_root(self, value: Path) -> None:
        self._qs.setValue("paths/vault_root", str(value))

    @property
    def output_dir(self) -> Path:
        default = Path(user_data_dir(APP_NAME)) / "exports"
        raw = self._qs.value("paths/output_dir", str(default))
        return Path(raw)

    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        self._qs.setValue("paths/output_dir", str(value))

    @property
    def temp_dir(self) -> Path:
        import tempfile
        return Path(tempfile.gettempdir()) / "veritas_reader"

    @property
    def kb_dir(self) -> Path:
        return Path(user_data_dir(APP_NAME)) / "knowledge_bases"

    @property
    def imports_dir(self) -> Path:
        """Global imports staging folder inside the vault root."""
        return self.vault_root / "imports"

    # ------------------------------------------------------------------
    # Renderer (GPT-SoVITS-v2)
    # ------------------------------------------------------------------

    @property
    def render_profile_path(self) -> str:
        return self._qs.value("renderer/profile_path", "")

    @render_profile_path.setter
    def render_profile_path(self, value: str) -> None:
        self._qs.setValue("renderer/profile_path", value)

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    @property
    def last_file_path(self) -> str:
        return self._qs.value("files/last_file_path", "")

    @last_file_path.setter
    def last_file_path(self, value: str) -> None:
        self._qs.setValue("files/last_file_path", value)

    @property
    def last_open_dir(self) -> str:
        return self._qs.value("files/last_open_dir", "")

    @last_open_dir.setter
    def last_open_dir(self, value: str) -> None:
        self._qs.setValue("files/last_open_dir", value)

    @property
    def last_autosave_path(self) -> str:
        return self._qs.value("files/last_autosave_path", "")

    @last_autosave_path.setter
    def last_autosave_path(self, value: str) -> None:
        self._qs.setValue("files/last_autosave_path", value)

    @property
    def last_vault_path(self) -> str:
        """str(vault.root) of the most recently active project vault."""
        return self._qs.value("files/last_vault_path", "")

    @last_vault_path.setter
    def last_vault_path(self, value: str) -> None:
        self._qs.setValue("files/last_vault_path", value)

    # ------------------------------------------------------------------
    # Window geometry
    # ------------------------------------------------------------------

    @property
    def window_geometry(self) -> bytes | None:
        val = self._qs.value("window/geometry")
        return bytes(val) if val else None

    @window_geometry.setter
    def window_geometry(self, value: bytes) -> None:
        self._qs.setValue("window/geometry", value)

    @property
    def dark_mode(self) -> bool:
        return self._qs.value("ui/dark_mode", False, type=bool)

    @dark_mode.setter
    def dark_mode(self, value: bool) -> None:
        self._qs.setValue("ui/dark_mode", value)

    def sync(self) -> None:
        self._qs.sync()
