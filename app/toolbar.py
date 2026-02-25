"""Toolbar widgets: model selector, voice selector, file naming, TTS/generate buttons."""

import logging

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
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

    def __init__(self, ollama_client, preferred: str = "llama3.2:latest", parent=None) -> None:
        super().__init__(parent)
        self._client = ollama_client
        self._preferred = preferred
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
            idx = self._combo.findText(self._preferred, Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
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
    """Dropdown listing TTS voice profiles from one or more engines."""

    def __init__(self, tts_engine, extra_engines=None, parent=None) -> None:
        super().__init__(parent)
        self._engine = tts_engine
        self._extra_engines = extra_engines or []
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
        for eng in self._extra_engines:
            self._voices.extend(eng.list_voices())
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


# ---------------------------------------------------------------------------
# Pacing Controls
# ---------------------------------------------------------------------------

class PacingControls(QWidget):
    """Spinboxes for TTS speed and inter-segment pause durations."""

    def __init__(self, settings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Thin separator before the controls
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        layout.addWidget(QLabel("Speed:"))
        self._speed = QDoubleSpinBox()
        self._speed.setRange(0.5, 1.5)
        self._speed.setSingleStep(0.05)
        self._speed.setDecimals(2)
        self._speed.setSuffix(" ×")
        self._speed.setFixedWidth(72)
        self._speed.setToolTip("Voice speed multiplier (1.0 = normal)")
        self._speed.setValue(self._settings.tts_speed)
        self._speed.valueChanged.connect(lambda v: setattr(self._settings, "tts_speed", v))
        layout.addWidget(self._speed)

        layout.addWidget(QLabel("Sentence:"))
        self._sentence = QSpinBox()
        self._sentence.setRange(0, 2000)
        self._sentence.setSingleStep(50)
        self._sentence.setSuffix(" ms")
        self._sentence.setFixedWidth(80)
        self._sentence.setToolTip("Silence added after each sentence ending")
        self._sentence.setValue(self._settings.pause_sentence_ms)
        self._sentence.valueChanged.connect(lambda v: setattr(self._settings, "pause_sentence_ms", v))
        layout.addWidget(self._sentence)

        layout.addWidget(QLabel("M-dash:"))
        self._mdash = QSpinBox()
        self._mdash.setRange(0, 2000)
        self._mdash.setSingleStep(50)
        self._mdash.setSuffix(" ms")
        self._mdash.setFixedWidth(80)
        self._mdash.setToolTip("Silence added at em-dash / double-hyphen breaks")
        self._mdash.setValue(self._settings.pause_mdash_ms)
        self._mdash.valueChanged.connect(lambda v: setattr(self._settings, "pause_mdash_ms", v))
        layout.addWidget(self._mdash)

        layout.addWidget(QLabel("Paragraph:"))
        self._paragraph = QSpinBox()
        self._paragraph.setRange(0, 5000)
        self._paragraph.setSingleStep(100)
        self._paragraph.setSuffix(" ms")
        self._paragraph.setFixedWidth(88)
        self._paragraph.setToolTip("Silence added between paragraphs")
        self._paragraph.setValue(self._settings.pause_paragraph_ms)
        self._paragraph.valueChanged.connect(lambda v: setattr(self._settings, "pause_paragraph_ms", v))
        layout.addWidget(self._paragraph)

    # ------------------------------------------------------------------
    # Accessors (read by window.py at synthesis time)
    # ------------------------------------------------------------------

    def speed(self) -> float:
        return self._speed.value()

    def sentence_ms(self) -> int:
        return self._sentence.value()

    def mdash_ms(self) -> int:
        return self._mdash.value()

    def paragraph_ms(self) -> int:
        return self._paragraph.value()
