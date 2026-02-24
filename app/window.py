"""Main application window for Veritas Reader."""

import logging
import tempfile
import threading
from pathlib import Path

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.editor import EditorWidget
from app.player import PlayerWidget
from app.toolbar import FileNameInput, ModelSelectorCombo, PacingControls, VoiceSelectorCombo, Worker
from config.settings import AppSettings
from core.audio_processor import process_audio_pipeline
from core.file_handler import (
    FileHandlerError,
    read_file,
    write_docx,
    write_markdown,
)
from core.ollama_client import OllamaClient, OllamaError
from core.tts_engine import get_engine

logger = logging.getLogger(__name__)


class StreamWorker(QRunnable):
    """Worker that streams Ollama tokens and emits them one by one."""

    class Signals(QObject):
        token = pyqtSignal(str)
        error = pyqtSignal(str)
        finished = pyqtSignal()

    def __init__(self, client: OllamaClient, model: str, prompt: str) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._prompt = prompt
        self._cancelled = threading.Event()
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            for token in self._client.generate(self._model, self._prompt, stream=True):
                if self._cancelled.is_set():
                    break
                self.signals.token.emit(token)
        except OllamaError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:
            logger.exception("Unexpected streaming error: %s", exc)
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class TTSWorker(QRunnable):
    """Worker that runs TTS synthesis off the main thread."""

    class Signals(QObject):
        result = pyqtSignal(str)   # path to raw wav
        error = pyqtSignal(str)
        finished = pyqtSignal()

    def __init__(
        self,
        engine,
        text: str,
        output_path: Path,
        voice,
        speed: float = 0.9,
        pause_sentence_ms: int = 500,
        pause_mdash_ms: int = 500,
        pause_paragraph_ms: int = 1000,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._text = text
        self._output_path = output_path
        self._voice = voice
        self._speed = speed
        self._pause_sentence_ms = pause_sentence_ms
        self._pause_mdash_ms = pause_mdash_ms
        self._pause_paragraph_ms = pause_paragraph_ms
        self.signals = self.Signals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._engine.synthesize(
                self._text,
                self._output_path,
                self._voice,
                speed=self._speed,
                pause_sentence_ms=self._pause_sentence_ms,
                pause_mdash_ms=self._pause_mdash_ms,
                pause_paragraph_ms=self._pause_paragraph_ms,
            )
            self.signals.result.emit(str(result))
        except Exception as exc:
            logger.exception("TTS error: %s", exc)
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = AppSettings()
        self._ollama = OllamaClient(self._settings.ollama_host)
        self._tts_engine = get_engine(self._settings.tts_engine)
        self._pool = QThreadPool.globalInstance()
        self._current_audio_path: Path | None = None
        self._stream_worker: StreamWorker | None = None
        self._generation_cancelled: bool = False

        self.setWindowTitle("Veritas Reader")
        self.resize(1200, 780)
        self._build_ui()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        root.addWidget(self._build_top_bar())
        root.addWidget(self._build_tts_pacing_bar())
        root.addWidget(self._build_filename_bar())
        root.addWidget(self._build_content_area(), stretch=1)
        root.addWidget(self._build_player_bar())
        root.addWidget(self._build_export_bar())

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left: import buttons
        open_btn = QPushButton("Open File…")
        open_btn.setToolTip("Open .md, .txt, or .docx file")
        open_btn.clicked.connect(self._on_open_file)
        layout.addWidget(open_btn)

        paste_btn = QPushButton("Paste Text…")
        paste_btn.setToolTip("Paste plain text")
        paste_btn.clicked.connect(self._on_paste_text)
        layout.addWidget(paste_btn)

        gdocs_btn = QPushButton("Google Docs…")
        gdocs_btn.setToolTip("Import from Google Docs")
        gdocs_btn.clicked.connect(self._on_import_gdocs)
        layout.addWidget(gdocs_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._ai_toggle_btn = QPushButton("✦ AI")
        self._ai_toggle_btn.setToolTip("Show/hide AI panel")
        self._ai_toggle_btn.clicked.connect(self._on_toggle_ai)
        layout.addWidget(self._ai_toggle_btn)

        layout.addStretch()

        # Right: voice selector + TTS button
        self._voice_selector = VoiceSelectorCombo(self._tts_engine)
        layout.addWidget(self._voice_selector)

        self._tts_btn = QPushButton("Synthesize TTS")
        self._tts_btn.setToolTip("Convert editor text to audio")
        self._tts_btn.clicked.connect(self._on_synthesize)
        layout.addWidget(self._tts_btn)

        return bar

    def _build_tts_pacing_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        self._pacing = PacingControls(self._settings)
        layout.addWidget(self._pacing)
        return bar

    def _build_filename_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        self._file_name_input = FileNameInput()
        self._file_name_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._file_name_input)
        return bar

    def _build_content_area(self) -> QSplitter:
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: main editor
        self._editor = EditorWidget()
        self._splitter.addWidget(self._editor)

        # Right: AI panel (hidden by default)
        self._ai_panel = QWidget()
        al = QVBoxLayout(self._ai_panel)
        al.setContentsMargins(4, 0, 0, 0)

        # Model + Generate row
        model_row = QWidget()
        ml = QHBoxLayout(model_row)
        ml.setContentsMargins(0, 0, 0, 0)
        self._model_selector = ModelSelectorCombo(
            self._ollama,
            preferred=self._settings.last_model or "llama3.2:latest",
        )
        ml.addWidget(self._model_selector)
        ml.addStretch()
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setToolTip("Send the prompt to the selected Ollama model")
        self._generate_btn.clicked.connect(self._on_generate)
        ml.addWidget(self._generate_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Stop generation")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_generation)
        ml.addWidget(self._stop_btn)

        al.addWidget(model_row)

        al.addWidget(QLabel("Prompt"))
        self._prompt_input = QTextEdit()
        self._prompt_input.setPlaceholderText("Enter your prompt here…")
        self._prompt_input.setMaximumHeight(120)
        al.addWidget(self._prompt_input)

        al.addWidget(QLabel("AI Output"))
        self._ai_output = QTextEdit()
        self._ai_output.setReadOnly(False)
        self._ai_output.setPlaceholderText("AI-generated text will appear here…")
        al.addWidget(self._ai_output)

        copy_btn = QPushButton("Copy to Editor")
        copy_btn.setToolTip("Copy AI output into the main editor")
        copy_btn.clicked.connect(self._on_copy_ai_to_editor)
        al.addWidget(copy_btn)

        self._ai_panel.hide()
        self._splitter.addWidget(self._ai_panel)
        self._splitter.setSizes([1, 0])

        return self._splitter

    def _build_player_bar(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self._player = PlayerWidget()
        layout.addWidget(self._player)
        return container

    def _build_export_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        export_md = QPushButton("Export .md")
        export_md.clicked.connect(lambda: self._on_export("md"))
        layout.addWidget(export_md)

        export_docx = QPushButton("Export .docx")
        export_docx.clicked.connect(lambda: self._on_export("docx"))
        layout.addWidget(export_docx)

        export_wav = QPushButton("Export .wav")
        export_wav.clicked.connect(self._on_export_audio)
        layout.addWidget(export_wav)

        sync_gdocs = QPushButton("Sync Google Docs")
        sync_gdocs.clicked.connect(self._on_sync_gdocs)
        layout.addWidget(sync_gdocs)

        layout.addStretch()
        return bar

    # ------------------------------------------------------------------
    # AI panel toggle
    # ------------------------------------------------------------------

    def _on_toggle_ai(self) -> None:
        if self._ai_panel.isVisible():
            self._ai_panel.hide()
            self._ai_toggle_btn.setText("✦ AI")
        else:
            self._ai_panel.show()
            self._splitter.setSizes([600, 420])
            self._ai_toggle_btn.setText("✦ AI ✕")

    # ------------------------------------------------------------------
    # File import actions
    # ------------------------------------------------------------------

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            str(self._settings.output_dir),
            "Documents (*.md *.txt *.docx);;All Files (*)",
        )
        if not path:
            return
        try:
            text = read_file(Path(path))
            self._editor.set_text(text)
            stem = Path(path).stem
            self._file_name_input.set_name(stem)
            self.statusBar().showMessage(f"Opened: {path}")
        except FileHandlerError as exc:
            QMessageBox.critical(self, "File Error", str(exc))

    def _on_paste_text(self) -> None:
        from app.dialogs.paste_dialog import PasteDialog
        dlg = PasteDialog(self)
        if dlg.exec() and dlg.get_text().strip():
            self._editor.set_text(dlg.get_text())
            self.statusBar().showMessage("Text pasted into editor.")

    def _on_import_gdocs(self) -> None:
        from app.dialogs.gdocs_dialog import GDocsDialog
        dlg = GDocsDialog(mode="import", parent=self)
        if not dlg.exec():
            return
        doc_id = dlg.get_doc_id()
        if not doc_id:
            return

        self.statusBar().showMessage("Fetching Google Doc…")
        worker = Worker(lambda: __import__("core.gdocs", fromlist=["read_doc"]).read_doc(doc_id))
        worker.signals.result.connect(lambda text: (
            self._editor.set_text(text),
            self.statusBar().showMessage(f"Imported Google Doc: {doc_id}"),
        ))
        worker.signals.error.connect(lambda e: QMessageBox.critical(self, "Google Docs Error", e))
        self._pool.start(worker)

    # ------------------------------------------------------------------
    # AI generation
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        model = self._model_selector.current_model()
        if not model:
            QMessageBox.warning(self, "No Model", "Please select an Ollama model first.")
            return

        prompt = self._prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt.")
            return

        self._settings.last_model = model
        self._ai_output.clear()
        self._generation_cancelled = False
        self._generate_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"Generating with {model}…")

        self._stream_worker = StreamWorker(self._ollama, model, prompt)
        self._stream_worker.signals.token.connect(self._on_stream_token)
        self._stream_worker.signals.error.connect(self._on_stream_error)
        self._stream_worker.signals.finished.connect(self._on_stream_finished)
        self._pool.start(self._stream_worker)

    def _on_stop_generation(self) -> None:
        if self._stream_worker is not None:
            self._generation_cancelled = True
            self._stream_worker.cancel()
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage("Stopping generation…")

    def _on_stream_token(self, token: str) -> None:
        cursor = self._ai_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self._ai_output.setTextCursor(cursor)
        self._ai_output.ensureCursorVisible()

    def _on_stream_error(self, error: str) -> None:
        QMessageBox.critical(self, "Generation Error", error)
        self.statusBar().showMessage("Generation failed.")

    def _on_stream_finished(self) -> None:
        self._generate_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._stream_worker = None
        msg = "Generation stopped." if self._generation_cancelled else "Generation complete."
        self.statusBar().showMessage(msg)

    def _on_copy_ai_to_editor(self) -> None:
        text = self._ai_output.toPlainText().strip()
        if text:
            self._editor.set_text(text)
            self.statusBar().showMessage("AI output copied to editor.")

    # ------------------------------------------------------------------
    # TTS synthesis
    # ------------------------------------------------------------------

    def _on_synthesize(self) -> None:
        text = self._editor.get_text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Editor", "Add some text to the editor before synthesizing.")
            return

        voice = self._voice_selector.current_voice()
        self._settings.last_voice = voice.id if voice else ""

        temp_dir = self._settings.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        base_name = self._file_name_input.get_name()
        raw_path = temp_dir / f"{base_name}_raw.wav"

        self._tts_btn.setEnabled(False)
        self.statusBar().showMessage("Synthesizing audio…")

        worker = TTSWorker(
            self._tts_engine,
            text,
            raw_path,
            voice,
            speed=self._pacing.speed(),
            pause_sentence_ms=self._pacing.sentence_ms(),
            pause_mdash_ms=self._pacing.mdash_ms(),
            pause_paragraph_ms=self._pacing.paragraph_ms(),
        )
        worker.signals.result.connect(self._on_tts_complete)
        worker.signals.error.connect(self._on_tts_error)
        self._pool.start(worker)

    def _on_tts_complete(self, raw_path_str: str) -> None:
        raw_path = Path(raw_path_str)
        processed_path = raw_path.with_name(raw_path.stem.replace("_raw", "_processed") + ".wav")

        self.statusBar().showMessage("Processing audio…")
        worker = Worker(
            process_audio_pipeline,
            raw_path,
            processed_path,
            self._settings.silence_thresh_db,
            self._settings.min_silence_ms,
        )
        worker.signals.result.connect(self._on_processing_complete)
        worker.signals.error.connect(self._on_tts_error)
        self._pool.start(worker)

    def _on_processing_complete(self, output_path) -> None:
        self._current_audio_path = Path(str(output_path))
        self._player.load(self._current_audio_path)
        self._tts_btn.setEnabled(True)
        self.statusBar().showMessage(f"Audio ready: {self._current_audio_path.name}")

    def _on_tts_error(self, error: str) -> None:
        QMessageBox.critical(self, "TTS Error", error)
        self._tts_btn.setEnabled(True)
        self.statusBar().showMessage("TTS failed.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self, fmt: str) -> None:
        text = self._editor.get_text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Editor", "Nothing to export.")
            return

        base = self._file_name_input.get_name()
        ext = f".{fmt}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export as {ext.upper()}",
            str(self._settings.output_dir / (base + ext)),
            f"{ext.upper()[1:]} Files (*{ext})",
        )
        if not path:
            return

        try:
            out = Path(path)
            if fmt == "md":
                write_markdown(text, out)
            elif fmt == "docx":
                write_docx(text, out)
            self._settings.output_dir = out.parent
            self.statusBar().showMessage(f"Exported: {out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_export_audio(self) -> None:
        if not self._current_audio_path or not self._current_audio_path.exists():
            QMessageBox.warning(self, "No Audio", "Synthesize audio first.")
            return

        base = self._file_name_input.get_name()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Audio",
            str(self._settings.output_dir / f"{base}.wav"),
            "WAV Files (*.wav);;MP3 Files (*.mp3)",
        )
        if not path:
            return

        try:
            from pydub import AudioSegment
            from core.audio_processor import export_wav, export_mp3
            audio = AudioSegment.from_wav(str(self._current_audio_path))
            out = Path(path)
            if out.suffix.lower() == ".mp3":
                export_mp3(audio, out)
            else:
                export_wav(audio, out)
            self._settings.output_dir = out.parent
            self.statusBar().showMessage(f"Audio exported: {out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_sync_gdocs(self) -> None:
        text = self._editor.get_text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Editor", "Nothing to sync.")
            return

        from app.dialogs.gdocs_dialog import GDocsDialog
        dlg = GDocsDialog(mode="export", parent=self)
        if not dlg.exec():
            return

        doc_id = dlg.get_doc_id()
        new_title = dlg.get_new_title()

        self.statusBar().showMessage("Syncing to Google Docs…")

        def _sync():
            from core import gdocs
            if doc_id:
                gdocs.write_doc(doc_id, text)
                return doc_id
            else:
                title = new_title or self._file_name_input.get_name()
                return gdocs.create_doc(title, text)

        worker = Worker(_sync)
        worker.signals.result.connect(
            lambda did: self.statusBar().showMessage(f"Synced to Google Doc: {did}")
        )
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(self, "Google Docs Error", e)
        )
        self._pool.start(worker)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geom = self._settings.window_geometry
        if geom:
            self.restoreGeometry(geom)

    def closeEvent(self, event) -> None:
        self._settings.window_geometry = bytes(self.saveGeometry())
        self._settings.sync()
        super().closeEvent(event)
