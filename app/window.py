"""Main application window for Veritas Reader."""

import logging
import tempfile
from pathlib import Path

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
        self.signals = self.Signals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self) -> None:
        try:
            for token in self._client.generate(self._model, self._prompt, stream=True):
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

        root.addWidget(self._build_file_bar())
        root.addWidget(self._build_model_bar())
        root.addWidget(self._build_pacing_bar())
        root.addWidget(self._build_content_area(), stretch=1)
        root.addWidget(self._build_player_bar())
        root.addWidget(self._build_export_bar())

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_file_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

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

        layout.addStretch()

        self._file_name_input = FileNameInput()
        layout.addWidget(self._file_name_input)

        return bar

    def _build_model_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        self._model_selector = ModelSelectorCombo(self._ollama)
        layout.addWidget(self._model_selector)

        self._voice_selector = VoiceSelectorCombo(self._tts_engine)
        layout.addWidget(self._voice_selector)

        layout.addStretch()

        self._generate_btn = QPushButton("Generate with AI")
        self._generate_btn.setToolTip("Send the prompt to the selected Ollama model")
        self._generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._generate_btn)

        self._tts_btn = QPushButton("Synthesize TTS")
        self._tts_btn.setToolTip("Convert editor text to audio")
        self._tts_btn.clicked.connect(self._on_synthesize)
        layout.addWidget(self._tts_btn)

        return bar

    def _build_pacing_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        self._pacing = PacingControls(self._settings)
        layout.addWidget(self._pacing)
        layout.addStretch()
        return bar

    def _build_content_area(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: main editor
        editor_container = QWidget()
        el = QVBoxLayout(editor_container)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(QLabel("Editor"))
        self._editor = EditorWidget()
        el.addWidget(self._editor)
        splitter.addWidget(editor_container)

        # Right: AI prompt + streamed output
        ai_container = QWidget()
        al = QVBoxLayout(ai_container)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(QLabel("AI Prompt"))
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

        splitter.addWidget(ai_container)
        splitter.setSizes([640, 480])
        return splitter

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
        self._generate_btn.setEnabled(False)
        self.statusBar().showMessage(f"Generating with {model}…")

        worker = StreamWorker(self._ollama, model, prompt)
        worker.signals.token.connect(self._on_stream_token)
        worker.signals.error.connect(self._on_stream_error)
        worker.signals.finished.connect(self._on_stream_finished)
        self._pool.start(worker)

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
        self.statusBar().showMessage("Generation complete.")

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
