"""Toolbar widgets: model selector, voice selector, file naming, TTS/generate buttons."""

import logging

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker boilerplate (reusable across the whole app)
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    progress = pyqtSignal(str)   # token-by-token streaming


class Worker(QRunnable):
    """Generic QRunnable that executes a callable in the thread pool."""

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as exc:
            logger.exception("Worker error: %s", exc)
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


# ---------------------------------------------------------------------------
# Model Selector
# ---------------------------------------------------------------------------

class ModelSelectorCombo(QWidget):
    """Dropdown that lists installed Ollama models, with a refresh button."""

    def __init__(self, ollama_client, parent=None) -> None:
        super().__init__(parent)
        self._client = ollama_client
        self._pool = QThreadPool.globalInstance()
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Model:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(180)
        self._combo.setToolTip("Select the Ollama model for generation")
        layout.addWidget(self._combo)

        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedWidth(28)
        self._refresh_btn.setToolTip("Refresh model list")
        self._refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(self._refresh_btn)

    def refresh(self) -> None:
        self._refresh_btn.setEnabled(False)
        self._combo.setEnabled(False)
        self._combo.clear()
        self._combo.addItem("Loading…")

        worker = Worker(self._client.list_models)
        worker.signals.result.connect(self._on_models_loaded)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(lambda: self._refresh_btn.setEnabled(True))
        self._pool.start(worker)

    def _on_models_loaded(self, models: list[str]) -> None:
        self._combo.clear()
        self._combo.setEnabled(True)
        if models:
            self._combo.addItems(models)
        else:
            self._combo.addItem("(no models found)")
        logger.debug("Loaded %d Ollama models.", len(models))

    def _on_error(self, error: str) -> None:
        self._combo.clear()
        self._combo.addItem("(Ollama unavailable)")
        self._combo.setEnabled(False)
        logger.warning("Could not load Ollama models: %s", error)

    def current_model(self) -> str | None:
        text = self._combo.currentText()
        if text in ("Loading…", "(no models found)", "(Ollama unavailable)"):
            return None
        return text


# ---------------------------------------------------------------------------
# Voice Selector
# ---------------------------------------------------------------------------

class VoiceSelectorCombo(QWidget):
    """Dropdown listing TTS voice profiles."""

    def __init__(self, tts_engine, parent=None) -> None:
        super().__init__(parent)
        self._engine = tts_engine
        self._voices = []
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Voice:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(160)
        self._combo.setToolTip("Select TTS voice profile")
        layout.addWidget(self._combo)

    def _populate(self) -> None:
        self._voices = self._engine.list_voices()
        self._combo.clear()
        for v in self._voices:
            self._combo.addItem(v.name, userData=v)
        logger.debug("Loaded %d TTS voices.", len(self._voices))

    def current_voice(self):
        """Return the currently selected VoiceProfile, or None."""
        idx = self._combo.currentIndex()
        if idx < 0 or not self._voices:
            return None
        return self._combo.itemData(idx)

    def set_voice_by_id(self, voice_id: str) -> bool:
        for i, v in enumerate(self._voices):
            if v.id == voice_id:
                self._combo.setCurrentIndex(i)
                return True
        return False


# ---------------------------------------------------------------------------
# File Name Input
# ---------------------------------------------------------------------------

class FileNameInput(QWidget):
    """Input field for the output file base name (no extension)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("File name:"))
        self._input = QLineEdit()
        self._input.setPlaceholderText("output")
        self._input.setMinimumWidth(140)
        self._input.setToolTip("Base name for exported files (no extension)")
        layout.addWidget(self._input)

    def get_name(self) -> str:
        """Return the entered file base name, falling back to 'output'."""
        text = self._input.text().strip()
        return text if text else "output"

    def set_name(self, name: str) -> None:
        self._input.setText(name)
