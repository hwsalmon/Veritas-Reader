"""Main application window for Scriptum Veritas."""

import logging
import tempfile
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, QTimer, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.editor_tabs import EditorTabWidget
from app.player import PlayerWidget
from app.toolbar import FileNameInput, ModelSelectorCombo, PacingControls, VoiceSelectorCombo, Worker
from config.settings import AppSettings
from core.audio_processor import process_audio_pipeline, normalize_loudness, apply_eq
from core.file_handler import (
    FileHandlerError,
    read_file,
    write_docx,
    write_markdown,
)
from core.knowledge_base import (
    KnowledgeBase,
    build_rag_system_prompt,
    chunk_text,
    DEFAULT_EMBED_MODEL,
)
from core.ollama_client import OllamaClient, OllamaError
from core.tts_engine import get_engine, GPTSoVITSEngine
from core.vault import Vault

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class StreamWorker(QRunnable):
    """Streams Ollama generation tokens."""

    class Signals(QObject):
        token = pyqtSignal(str)
        error = pyqtSignal(str)
        finished = pyqtSignal()

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        prompt: str | None = None,
        messages: list | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._prompt = prompt
        self._messages = messages   # if set, use chat() instead of generate()
        self._cancelled = threading.Event()
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self._messages is not None:
                gen = self._client.chat(self._model, self._messages, stream=True)
            else:
                gen = self._client.generate(self._model, self._prompt, stream=True)
            for token in gen:
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
    """Runs TTS synthesis off the main thread."""

    class Signals(QObject):
        result = pyqtSignal(str)
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


class KBBuildWorker(QRunnable):
    """Chunks text and embeds each chunk to build a KnowledgeBase."""

    class Signals(QObject):
        progress = pyqtSignal(int, int)   # current, total
        result = pyqtSignal(object)       # KnowledgeBase
        error = pyqtSignal(str)
        finished = pyqtSignal()

    def __init__(
        self,
        client: OllamaClient,
        text: str,
        name: str,
        embed_model: str,
    ) -> None:
        super().__init__()
        self._client = client
        self._text = text
        self._name = name
        self._embed_model = embed_model
        self._cancelled = threading.Event()
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            chunks = chunk_text(self._text)
            if not chunks:
                self.signals.error.emit("No text chunks found in editor content.")
                return

            embeddings: list[list[float]] = []
            for i, chunk in enumerate(chunks):
                if self._cancelled.is_set():
                    return
                emb = self._client.embed(self._embed_model, chunk)
                embeddings.append(emb)
                self.signals.progress.emit(i + 1, len(chunks))

            kb = KnowledgeBase(
                name=self._name,
                embedding_model=self._embed_model,
                chunks=chunks,
                embeddings=np.array(embeddings, dtype=np.float32),
            )
            self.signals.result.emit(kb)
        except OllamaError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:
            logger.exception("KB build error: %s", exc)
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class KBQueryWorker(QRunnable):
    """Embeds a question, retrieves top chunks, and streams an answer."""

    class Signals(QObject):
        token = pyqtSignal(str)
        error = pyqtSignal(str)
        finished = pyqtSignal()

    def __init__(
        self,
        client: OllamaClient,
        kb: KnowledgeBase,
        question: str,
        chat_model: str,
    ) -> None:
        super().__init__()
        self._client = client
        self._kb = kb
        self._question = question
        self._chat_model = chat_model
        self._cancelled = threading.Event()
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self._cancelled.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            q_emb = self._client.embed(self._kb.embedding_model, self._question)
            top_chunks = self._kb.query(q_emb)
            system = build_rag_system_prompt(top_chunks)
            for token in self._client.generate(
                self._chat_model, self._question, system=system, stream=True
            ):
                if self._cancelled.is_set():
                    break
                self.signals.token.emit(token)
        except OllamaError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:
            logger.exception("KB query error: %s", exc)
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class RenderWorker(QRunnable):
    """Runs GPT-SoVITS-v2 batch render off the main thread."""

    class Signals(QObject):
        progress = pyqtSignal(int, int, str)   # current, total, filename
        result   = pyqtSignal(list)            # list[str] of written paths
        error    = pyqtSignal(str)
        finished = pyqtSignal()

    def __init__(self, paragraphs: list[str], output_prefix: Path, profile_path: str) -> None:
        super().__init__()
        self._paragraphs    = paragraphs
        self._output_prefix = output_prefix
        self._profile_path  = profile_path
        self.signals = self.Signals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self) -> None:
        written: list[str] = []
        try:
            from core.renderer import export_to_wav
            paras = [p for p in self._paragraphs if p.strip()]
            total = len(paras)
            for idx, para in enumerate(paras, start=1):
                out = (
                    self._output_prefix.parent
                    / f"{self._output_prefix.name}_{idx:02d}.wav"
                )
                self.signals.progress.emit(idx, total, out.name)
                export_to_wav(para, out, self._profile_path)
                written.append(str(out))
            self.signals.result.emit(written)
        except Exception as exc:
            logger.exception("Render error: %s", exc)
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = AppSettings()
        self._ollama = OllamaClient(self._settings.ollama_host)
        self._tts_engine = get_engine(self._settings.tts_engine)
        self._gptsovits_engine = GPTSoVITSEngine()
        self._pool = QThreadPool.globalInstance()
        self._current_audio_path: Path | None = None
        self._stream_worker: StreamWorker | None = None
        self._tts_worker: TTSWorker | None = None
        self._generation_cancelled: bool = False
        self._kb: KnowledgeBase | None = None
        self._kb_current_path: Path | None = None
        self._kb_build_worker: KBBuildWorker | None = None
        self._kb_query_worker: KBQueryWorker | None = None
        self._render_worker: RenderWorker | None = None
        self._current_file_path: Path | None = None
        self._autosave_path: Path | None = None
        self._dirty: bool = False
        self._generate_chat_messages: list[dict] = []
        self._stream_buffer: str = ""
        self._vault: Vault | None = None
        self._kb_chat_messages: list[dict] = []
        self._kb_response_buffer: str = ""

        self.setWindowTitle("Scriptum Veritas")
        self.resize(1200, 780)
        self._build_ui()
        self._restore_geometry()
        self._apply_font_size(self._settings.font_size)
        self._restore_last_session()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)  # every 60 seconds
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        root.addWidget(self._build_top_bar())
        root.addWidget(self._build_player_bar())
        self._tts_bar = self._build_tts_bar()
        root.addWidget(self._tts_bar)
        root.addWidget(self._build_filename_bar())
        root.addWidget(self._build_content_area(), stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self._editor.text_changed.connect(self._mark_dirty)
        self._editor.grammar_check_requested.connect(self._on_grammar_check)
        self._editor.clone_requested.connect(self._on_clone_tab)

        # Global text-size shortcuts (work regardless of which widget has focus)
        zoom_in_sc = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        zoom_in_sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_in_sc.activated.connect(self._zoom_in)
        # Ctrl+= (no-shift) as an additional zoom-in binding
        zoom_in_eq = QShortcut(QKeySequence("Ctrl+="), self)
        zoom_in_eq.setContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_in_eq.activated.connect(self._zoom_in)
        zoom_out_sc = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        zoom_out_sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        zoom_out_sc.activated.connect(self._zoom_out)

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("File")

        new_action = QAction("New File", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_file)
        file_menu.addAction(new_action)

        file_menu.addSeparator()
        file_menu.addAction("Open File…", self._on_open_file)
        file_menu.addAction("Paste Text…", self._on_paste_text)
        file_menu.addAction("Google Docs…", self._on_import_gdocs)
        file_menu.addSeparator()
        file_menu.addAction("Import Documents…", self._on_import_documents)
        file_menu.addAction("Open from Imports…", self._on_open_from_imports)
        file_menu.addSeparator()
        file_menu.addAction("Choose Vault…", self._on_choose_vault)
        file_menu.addSeparator()
        file_menu.addAction("Open Project…", self._on_open_project)
        file_menu.addAction("Close Project",  self._on_close_project)
        file_menu.addSeparator()
        file_menu.addAction("Close All Tabs", self._on_close_all_tabs)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self._on_exit)
        file_menu.addAction(exit_action)

        export_menu = mb.addMenu("Export")
        export_menu.addAction("Commit Version", self._on_commit_version)
        export_menu.addSeparator()
        export_menu.addAction("Export .md…", lambda: self._on_export("md"))
        export_menu.addAction("Export .docx…", lambda: self._on_export("docx"))
        export_menu.addAction("Export .wav…", self._on_export_audio)
        export_menu.addSeparator()
        export_menu.addAction("Sync Google Docs…", self._on_sync_gdocs)

        view_menu = mb.addMenu("View")
        self._theme_action = view_menu.addAction("Dark Mode")
        self._theme_action.setCheckable(True)
        self._theme_action.setChecked(self._settings.dark_mode)
        self._theme_action.triggered.connect(self._on_toggle_theme)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()

        self._tts_toggle_btn = QPushButton("TTS")
        self._tts_toggle_btn.setToolTip("Show/hide TTS controls")
        self._tts_toggle_btn.clicked.connect(self._on_toggle_tts)
        layout.addWidget(self._tts_toggle_btn)

        self._ai_toggle_btn = QPushButton("✦ AI")
        self._ai_toggle_btn.setToolTip("Show/hide AI panel")
        self._ai_toggle_btn.clicked.connect(self._on_toggle_ai)
        layout.addWidget(self._ai_toggle_btn)

        return bar

    def _build_tts_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._voice_selector = VoiceSelectorCombo(self._tts_engine, extra_engines=[self._gptsovits_engine])
        layout.addWidget(self._voice_selector)

        self._pacing = PacingControls(self._settings)
        layout.addWidget(self._pacing)

        layout.addStretch()

        self._tts_btn = QPushButton("Synthesize")
        self._tts_btn.setToolTip("Convert editor text to audio")
        self._tts_btn.clicked.connect(self._on_synthesize)
        layout.addWidget(self._tts_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._render_profile_btn = QPushButton("⋯")
        self._render_profile_btn.setFixedWidth(28)
        self._render_profile_btn.clicked.connect(self._on_pick_render_profile)
        self._update_render_profile_tooltip()
        layout.addWidget(self._render_profile_btn)

        self._render_btn = QPushButton("Render")
        self._render_btn.setToolTip("Batch-render editor text to WAV via GPT-SoVITS-v2")
        self._render_btn.clicked.connect(self._on_render)
        layout.addWidget(self._render_btn)

        bar.hide()
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

        self._editor = EditorTabWidget()
        self._editor.tabs_changed.connect(self._refresh_kb_source_combo)
        self._splitter.addWidget(self._editor)

        # AI panel with tabs
        self._ai_panel = QWidget()
        al = QVBoxLayout(self._ai_panel)
        al.setContentsMargins(4, 0, 0, 0)

        # AI service quick-launch buttons (open persistent browser tabs)
        _AI_SERVICES = [
            ("NotebookLM", "https://notebooklm.google.com"),
            ("ChatGPT",    "https://chat.openai.com"),
            ("Gemini",     "https://gemini.google.com"),
            ("Claude AI",  "https://claude.ai"),
        ]
        web_row = QWidget()
        wrl = QHBoxLayout(web_row)
        wrl.setContentsMargins(0, 2, 0, 2)
        wrl.setSpacing(4)
        for label, url in _AI_SERVICES:
            btn = QPushButton(label)
            btn.setToolTip(f"Open {label}")
            btn.clicked.connect(
                lambda checked, u=url, l=label: self._open_ai_browser(u, l)
            )
            wrl.addWidget(btn)
        wrl.addStretch()
        al.addWidget(web_row)

        self._ai_tabs = QTabWidget()
        self._ai_tabs.setTabsClosable(True)
        self._ai_tabs.tabCloseRequested.connect(self._on_close_ai_tab)
        self._ai_tabs.addTab(self._build_generate_tab(), "Generate")
        self._ai_tabs.addTab(self._build_kb_tab(), "KB Chat")
        self._ai_tabs.addTab(self._build_history_tab(), "History")
        # Remove close buttons from the 3 permanent tabs
        from PyQt6.QtWidgets import QTabBar
        for i in range(3):
            self._ai_tabs.tabBar().setTabButton(i, QTabBar.ButtonPosition.RightSide, None)
            self._ai_tabs.tabBar().setTabButton(i, QTabBar.ButtonPosition.LeftSide, None)
        al.addWidget(self._ai_tabs)

        self._ai_panel.hide()
        self._splitter.addWidget(self._ai_panel)
        self._splitter.setSizes([1, 0])

        # Populate KB source combo with initial tab list
        QTimer.singleShot(0, self._refresh_kb_source_combo)

        return self._splitter

    def _build_generate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Model + Generate + Stop row
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
        layout.addWidget(model_row)

        layout.addWidget(QLabel("Prompt"))
        self._prompt_input = QTextEdit()
        self._prompt_input.setPlaceholderText("Enter your prompt here…")
        self._prompt_input.setMaximumHeight(120)
        layout.addWidget(self._prompt_input)

        layout.addWidget(QLabel("AI Output"))
        self._ai_output = QTextEdit()
        self._ai_output.setReadOnly(False)
        self._ai_output.setPlaceholderText("AI-generated text will appear here…")
        layout.addWidget(self._ai_output)

        action_row = QWidget()
        arl = QHBoxLayout(action_row)
        arl.setContentsMargins(0, 0, 0, 0)
        copy_btn = QPushButton("Copy to Editor")
        copy_btn.setToolTip("Copy AI output into the main editor")
        copy_btn.clicked.connect(self._on_copy_ai_to_editor)
        arl.addWidget(copy_btn)
        tab_btn = QPushButton("Open in New Tab")
        tab_btn.setToolTip("Open AI output in a new editor tab for tweaking")
        tab_btn.clicked.connect(self._on_ai_output_to_tab)
        arl.addWidget(tab_btn)
        arl.addStretch()
        layout.addWidget(action_row)

        # --- Follow-up chat row ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        followup_row = QWidget()
        frl = QHBoxLayout(followup_row)
        frl.setContentsMargins(0, 0, 0, 0)
        frl.setSpacing(4)
        frl.addWidget(QLabel("Follow-up:"))
        self._followup_input = QLineEdit()
        self._followup_input.setPlaceholderText("Ask a follow-up question…")
        self._followup_input.returnPressed.connect(self._on_followup)
        self._followup_input.setEnabled(False)
        frl.addWidget(self._followup_input)
        self._followup_btn = QPushButton("Ask")
        self._followup_btn.setEnabled(False)
        self._followup_btn.clicked.connect(self._on_followup)
        frl.addWidget(self._followup_btn)
        layout.addWidget(followup_row)

        return tab

    def _build_kb_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Source tab selector
        src_row = QWidget()
        srl = QHBoxLayout(src_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(4)
        srl.addWidget(QLabel("Source tab:"))
        self._kb_source_combo = QComboBox()
        self._kb_source_combo.setToolTip("Which editor tab to build the knowledge base from")
        self._kb_source_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        srl.addWidget(self._kb_source_combo)
        layout.addWidget(src_row)

        # Embed model + action buttons
        ctrl_row = QWidget()
        cl = QHBoxLayout(ctrl_row)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(QLabel("Embed model:"))
        self._embed_model_input = QLineEdit(DEFAULT_EMBED_MODEL)
        self._embed_model_input.setFixedWidth(160)
        self._embed_model_input.setToolTip(
            "Ollama embedding model (run: ollama pull nomic-embed-text)"
        )
        cl.addWidget(self._embed_model_input)
        self._kb_build_btn = QPushButton("Build from Editor")
        self._kb_build_btn.setToolTip("Embed the selected source tab into a knowledge base")
        self._kb_build_btn.clicked.connect(self._on_build_kb)
        cl.addWidget(self._kb_build_btn)
        self._kb_load_btn = QPushButton("Load…")
        self._kb_load_btn.setToolTip("Load a saved .vkb knowledge base file")
        self._kb_load_btn.clicked.connect(self._on_load_kb)
        cl.addWidget(self._kb_load_btn)
        self._kb_save_btn = QPushButton("Save…")
        self._kb_save_btn.setToolTip("Save the current knowledge base to disk")
        self._kb_save_btn.setEnabled(False)
        self._kb_save_btn.clicked.connect(self._on_save_kb)
        cl.addWidget(self._kb_save_btn)
        layout.addWidget(ctrl_row)

        # KB status + progress
        self._kb_status = QLabel("No knowledge base loaded.")
        self._kb_status.setWordWrap(True)
        layout.addWidget(self._kb_status)

        self._kb_progress = QProgressBar()
        self._kb_progress.setVisible(False)
        self._kb_progress.setTextVisible(True)
        layout.addWidget(self._kb_progress)

        # Chat history
        layout.addWidget(QLabel("Chat"))
        self._kb_chat = QTextEdit()
        self._kb_chat.setReadOnly(True)
        self._kb_chat.setPlaceholderText(
            "Build or load a knowledge base, then ask a question below…"
        )
        layout.addWidget(self._kb_chat)

        # Question input + Ask + Stop
        q_row = QWidget()
        ql = QHBoxLayout(q_row)
        ql.setContentsMargins(0, 0, 0, 0)
        self._kb_question = QLineEdit()
        self._kb_question.setPlaceholderText("Ask a question about the document…")
        self._kb_question.returnPressed.connect(self._on_ask_kb)
        ql.addWidget(self._kb_question)
        self._kb_ask_btn = QPushButton("Ask")
        self._kb_ask_btn.setEnabled(False)
        self._kb_ask_btn.clicked.connect(self._on_ask_kb)
        ql.addWidget(self._kb_ask_btn)
        self._kb_stop_btn = QPushButton("Stop")
        self._kb_stop_btn.setEnabled(False)
        self._kb_stop_btn.clicked.connect(self._on_stop_kb_query)
        ql.addWidget(self._kb_stop_btn)
        layout.addWidget(q_row)

        return tab

    def _build_history_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("AI Session History"))

        self._history_list = QListWidget()
        self._history_list.setToolTip("Double-click a session to reload it in the Generate tab")
        self._history_list.itemDoubleClicked.connect(self._on_load_history_item)
        layout.addWidget(self._history_list)

        btn_row = QWidget()
        brl = QHBoxLayout(btn_row)
        brl.setContentsMargins(0, 0, 0, 0)
        load_btn = QPushButton("Load Session")
        load_btn.setToolTip("Reload the selected session into the Generate tab")
        load_btn.clicked.connect(self._on_load_history_item)
        brl.addWidget(load_btn)
        del_btn = QPushButton("Delete")
        del_btn.setToolTip("Permanently delete the selected session file")
        del_btn.clicked.connect(self._on_delete_history_item)
        brl.addWidget(del_btn)
        brl.addStretch()
        layout.addWidget(btn_row)

        return tab

    def _build_player_bar(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self._player = PlayerWidget()
        layout.addWidget(self._player)
        return container

    # ------------------------------------------------------------------
    # AI panel toggle
    # ------------------------------------------------------------------

    def _on_toggle_tts(self) -> None:
        if self._tts_bar.isVisible():
            self._tts_bar.hide()
            self._tts_toggle_btn.setText("TTS")
        else:
            self._tts_bar.show()
            self._tts_toggle_btn.setText("TTS ✕")

    def _on_toggle_ai(self) -> None:
        if self._ai_panel.isVisible():
            self._ai_panel.hide()
            self._ai_toggle_btn.setText("✦ AI")
        else:
            self._ai_panel.show()
            self._splitter.setSizes([600, 420])
            self._ai_toggle_btn.setText("✦ AI ✕")

    def _on_clone_tab(self) -> None:
        """Clone active editor tab and save a vault version of its content."""
        text, new_widget = self._editor.clone_current_tab()
        if self._vault and text.strip():
            suffix = self._current_file_path.suffix if self._current_file_path else ".md"
            version_path = self._vault.next_version_path(suffix)
            try:
                version_path.parent.mkdir(parents=True, exist_ok=True)
                version_path.write_text(text, encoding="utf-8")
                # Register the file so renaming the tab also renames the file
                self._editor.set_tab_path(new_widget, version_path)
                self.statusBar().showMessage(
                    f"Cloned + version saved: {version_path.name}"
                )
            except Exception as exc:
                logger.warning("Clone vault save failed: %s", exc)
        else:
            self.statusBar().showMessage("Tab cloned.")

    def _on_close_ai_tab(self, idx: int) -> None:
        """Close a dynamically-added AI browser tab; protect the 3 permanent tabs."""
        if idx >= 3:
            self._ai_tabs.removeTab(idx)

    def _open_ai_browser(self, url: str, title: str) -> None:
        """Open a persistent browser tab in the AI panel."""
        from app.web_tab import BrowserTab
        from PyQt6.QtWidgets import QTabBar
        browser = BrowserTab(url)
        self._ai_tabs.addTab(browser, title)
        new_idx = self._ai_tabs.count() - 1
        self._ai_tabs.setCurrentIndex(new_idx)
        if not self._ai_panel.isVisible():
            self._ai_panel.show()
            self._splitter.setSizes([600, 420])
            self._ai_toggle_btn.setText("✦ AI ✕")

    # ------------------------------------------------------------------
    # Text size zoom
    # ------------------------------------------------------------------

    _FONT_MIN = 7
    _FONT_MAX = 28

    def _zoom_in(self) -> None:
        self._apply_font_size(self._settings.font_size + 1)

    def _zoom_out(self) -> None:
        self._apply_font_size(self._settings.font_size - 1)

    def _apply_font_size(self, size: int) -> None:
        """Apply *size* (clamped to 7–28 pt) to every text widget in the app."""
        size = max(self._FONT_MIN, min(self._FONT_MAX, size))
        self._settings.font_size = size

        # Editor tabs (all existing + future tabs via EditorTabWidget.set_font_size)
        self._editor.set_font_size(size)

        # AI / KB panel text widgets — use the app's default sans-serif at new size
        panel_font = QFont()
        panel_font.setPointSize(size)
        for w in (
            self._prompt_input,
            self._ai_output,
            self._kb_chat,
            self._kb_question,
            self._followup_input,
        ):
            w.setFont(panel_font)

        self.statusBar().showMessage(f"Text size: {size} pt", 2000)

    def _on_toggle_theme(self) -> None:
        from PyQt6.QtWidgets import QApplication
        from app.theme import apply_dark, apply_light
        dark = not self._settings.dark_mode
        self._settings.dark_mode = dark
        app = QApplication.instance()
        if dark:
            apply_dark(app)
        else:
            apply_light(app)
        self._theme_action.setChecked(dark)

    # ------------------------------------------------------------------
    # File import actions
    # ------------------------------------------------------------------

    def _on_open_file(self) -> None:
        start_dir = self._settings.last_open_dir or str(self._settings.output_dir)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            start_dir,
            "Documents (*.md *.txt *.docx);;All Files (*)",
        )
        if not path:
            return
        try:
            p = Path(path)
            text = read_file(p)
            self._editor.set_text(text)
            self._editor.set_main_tab_title(p.stem)
            self._file_name_input.set_name(p.stem)
            self._settings.last_file_path = path
            self._settings.last_open_dir = str(p.parent)
            self._current_file_path = p
            self._init_vault(p)
            self._autosave_path = self._compute_autosave_path(p)
            self._dirty = False
            self.statusBar().showMessage(f"Opened: {path}")
            logger.info("Session autosave target: %s", self._autosave_path)
        except FileHandlerError as exc:
            QMessageBox.critical(self, "File Error", str(exc))

    def _on_new_file(self) -> None:
        if self._dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before creating a new document?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Save:
                self._autosave()

        name, ok = QInputDialog.getText(
            self,
            "New Project",
            "Project name:",
            QLineEdit.EchoMode.Normal,
            "",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            name = "Untitled"

        # Build a logical file path inside the vault root so vault + autosave work
        new_file_path = self._settings.vault_root / name / f"{name}.md"

        self._editor.set_text("")
        self._editor.set_main_tab_title(name)
        self._file_name_input.set_name(name)
        self._current_file_path = new_file_path
        self._dirty = False
        self._kb = None
        self._kb_current_path = None
        self._settings.last_file_path = str(new_file_path)
        self._settings.last_autosave_path = ""
        self._init_vault(new_file_path)
        self._autosave_path = self._compute_autosave_path(new_file_path)
        self.statusBar().showMessage(f"New project '{name}' created.")

    def _on_close_all_tabs(self) -> None:
        self._editor.close_all_non_primary_tabs()

    def _on_import_documents(self) -> None:
        start_dir = self._settings.last_open_dir or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Documents",
            start_dir,
            "Documents (*.md *.txt *.docx);;All Files (*)",
        )
        if not paths:
            return
        imports_dir = self._settings.imports_dir
        imports_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped: list[str] = []
        for src_str in paths:
            src = Path(src_str)
            dest = imports_dir / src.name
            # Collision-avoidance
            n = 1
            while dest.exists():
                dest = imports_dir / f"{src.stem}_{n}{src.suffix}"
                n += 1
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception as exc:
                logger.warning("Could not import %s: %s", src, exc)
                skipped.append(src.name)
        msg = f"{copied} file{'s' if copied != 1 else ''} imported to imports folder."
        if skipped:
            QMessageBox.warning(
                self,
                "Import Warning",
                f"Could not import: {', '.join(skipped)}",
            )
        self.statusBar().showMessage(msg)

    def _on_open_from_imports(self) -> None:
        from app.dialogs.imports_dialog import ImportsBrowserDialog
        dlg = ImportsBrowserDialog(self._settings.imports_dir, parent=self)
        if not dlg.exec():
            return
        p = dlg.get_selected_path()
        if not p:
            return
        try:
            text = read_file(p)
            self._editor.set_text(text)
            self._editor.set_main_tab_title(p.stem)
            self._file_name_input.set_name(p.stem)
            self._settings.last_file_path = str(p)
            self._settings.last_open_dir = str(p.parent)
            self._current_file_path = p
            self._init_vault(p)
            self._autosave_path = self._compute_autosave_path(p)
            self._dirty = False
            self.statusBar().showMessage(f"Opened from imports: {p.name}")
        except FileHandlerError as exc:
            QMessageBox.critical(self, "File Error", str(exc))

    def _on_exit(self) -> None:
        self._autosave()
        self.close()

    def _reopen_last_file(self) -> None:
        """Silently reload the last opened file on startup.

        Prefers the last autosave over the original so edits are never lost.
        """
        path_str = self._settings.last_file_path
        if not path_str:
            return
        p = Path(path_str)
        if not p.exists():
            return
        try:
            # Init vault first so _compute_autosave_path uses vault/versions/
            self._init_vault(p)

            # Restore from the last autosave if it exists (more recent than original)
            load_path = p
            autosave_str = self._settings.last_autosave_path
            if autosave_str:
                autosave = Path(autosave_str)
                if autosave.exists():
                    load_path = autosave

            text = read_file(load_path)
            self._editor.set_text(text)
            self._editor.set_main_tab_title(p.stem)
            self._file_name_input.set_name(p.stem)
            self._current_file_path = p          # always the original
            self._autosave_path = self._compute_autosave_path(p)
            self._dirty = False
            msg = f"Reopened: {p.name}"
            if load_path != p:
                msg += f"  (restored from {load_path.name})"
            self.statusBar().showMessage(msg)
            logger.info("Reopened %s | loaded from %s | next autosave → %s",
                        p.name, load_path.name, self._autosave_path)
        except Exception as exc:
            logger.warning("Could not reopen last file %s: %s", p, exc)

    def _init_vault(self, path: Path) -> None:
        """Create (or re-open) the vault for the document at *path*."""
        self._vault = Vault(self._settings.vault_root, path.stem)
        self._vault.init()
        self._settings.last_vault_path = str(self._vault.root)
        self._refresh_history_tab()
        logger.info("Vault: %s", self._vault.root)

    def _on_choose_vault(self) -> None:
        """Let user pick a new vault root folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Vault Root Folder",
            str(self._settings.vault_root),
        )
        if not folder:
            return
        self._settings.vault_root = Path(folder)
        # Re-init vault for current document if one is open
        if self._current_file_path:
            self._init_vault(self._current_file_path)
        self.statusBar().showMessage(f"Vault root: {folder}")

    def _refresh_history_tab(self) -> None:
        """Repopulate the History list from the vault's ai_history/ folder."""
        if not hasattr(self, "_history_list"):
            return
        self._history_list.clear()
        if self._vault is None:
            return
        for path in self._vault.list_ai_sessions():
            try:
                data = self._vault.load_ai_session(path)
                ts = data.get("timestamp", path.stem)
                stype = data.get("type", "?")
                model = data.get("model", "?")
                label = f"{ts}  [{stype}]  {model}"
            except Exception:
                label = path.stem
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._history_list.addItem(item)

    def _compute_autosave_path(self, original: Path) -> Path:
        """Return the next unused versioned path.

        Uses vault/versions/ when a vault is active, otherwise falls back to
        the same directory as the original file.
        """
        if self._vault:
            return self._vault.next_version_path(original.suffix or ".md")
        # Legacy fallback: same directory as original
        stem = original.stem
        suffix = original.suffix or ".md"
        parent = original.parent
        n = 1
        while True:
            candidate = parent / f"{stem}-{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _autosave(self) -> None:
        """Write a versioned auto-save copy if the editor content has changed."""
        if self._autosave_path is None or not self._dirty:
            return
        text = self._editor.get_text()
        if not text.strip():
            return
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            self._autosave_path.write_text(text, encoding="utf-8")
            self._dirty = False
            self._settings.last_autosave_path = str(self._autosave_path)
            self.statusBar().showMessage(
                f"Auto-saved: {self._autosave_path.name}", 3000
            )
            logger.debug("Auto-saved to %s", self._autosave_path)
            self._save_session()
        except Exception as exc:
            logger.warning("Auto-save failed: %s", exc)

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
        self._generate_chat_messages = [{"role": "user", "content": prompt}]
        self._stream_buffer = ""
        self._generation_cancelled = False
        self._generate_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._followup_input.setEnabled(False)
        self._followup_btn.setEnabled(False)
        self.statusBar().showMessage(f"Generating with {model}…")

        self._stream_worker = StreamWorker(
            self._ollama, model, messages=self._generate_chat_messages
        )
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
        self._stream_buffer += token
        cursor = self._ai_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self._ai_output.setTextCursor(cursor)
        self._ai_output.ensureCursorVisible()

    def _on_stream_error(self, error: str) -> None:
        QMessageBox.critical(self, "Generation Error", error)
        self.statusBar().showMessage("Generation failed.")

    def _on_stream_finished(self) -> None:
        # Capture assistant response in conversation history
        if self._stream_buffer.strip():
            self._generate_chat_messages.append(
                {"role": "assistant", "content": self._stream_buffer.strip()}
            )
        self._stream_buffer = ""
        self._generate_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._stream_worker = None
        # Enable follow-up only when there's a conversation to continue
        has_chat = len(self._generate_chat_messages) >= 2
        self._followup_input.setEnabled(has_chat)
        self._followup_btn.setEnabled(has_chat)
        # Persist to vault
        if self._vault and has_chat:
            self._vault.save_ai_session(
                session_type="generate",
                model=self._model_selector.current_model() or "unknown",
                messages=self._generate_chat_messages,
                document_name=self._file_name_input.get_name(),
            )
            self._refresh_history_tab()
        msg = "Generation stopped." if self._generation_cancelled else "Generation complete."
        self.statusBar().showMessage(msg)

    def _on_followup(self) -> None:
        question = self._followup_input.text().strip()
        if not question or not self._generate_chat_messages:
            return
        model = self._model_selector.current_model()
        if not model:
            QMessageBox.warning(self, "No Model", "Please select an Ollama model first.")
            return

        self._generate_chat_messages.append({"role": "user", "content": question})
        self._followup_input.clear()
        self._stream_buffer = ""

        # Append visual separator in the output area
        cursor = self._ai_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(f"\n\n{'─' * 36}\n{question}\n{'─' * 36}\n\n")
        self._ai_output.setTextCursor(cursor)
        self._ai_output.ensureCursorVisible()

        self._generation_cancelled = False
        self._generate_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._followup_input.setEnabled(False)
        self._followup_btn.setEnabled(False)
        self.statusBar().showMessage(f"Generating follow-up with {model}…")

        self._stream_worker = StreamWorker(
            self._ollama, model, messages=self._generate_chat_messages
        )
        self._stream_worker.signals.token.connect(self._on_stream_token)
        self._stream_worker.signals.error.connect(self._on_stream_error)
        self._stream_worker.signals.finished.connect(self._on_stream_finished)
        self._pool.start(self._stream_worker)

    def _on_grammar_check(self, text: str) -> None:
        """Pre-fill AI panel with a grammar-check prompt and open it."""
        prompt = (
            "You are a copy-editor specialising in US English. "
            "Review the following text for grammar, punctuation, spelling, "
            "and style issues. List each problem with its location (quote the "
            "relevant phrase), explain the issue, and provide a corrected version. "
            "Be concise. If no issues are found, say so.\n\n"
            "TEXT TO REVIEW:\n\n"
            f"{text}"
        )
        self._prompt_input.setPlainText(prompt)
        # Open AI panel if hidden
        if not self._ai_panel.isVisible():
            self._ai_panel.show()
            self._splitter.setSizes([600, 420])
            self._ai_toggle_btn.setText("✦ AI ✕")
        # Switch to Generate tab
        self._ai_tabs.setCurrentIndex(0)
        self._on_generate()

    def _on_copy_ai_to_editor(self) -> None:
        text = self._ai_output.toPlainText().strip()
        if text:
            self._editor.set_text(text)
            self.statusBar().showMessage("AI output copied to editor.")

    def _on_ai_output_to_tab(self) -> None:
        text = self._ai_output.toPlainText().strip()
        if text:
            self._editor.open_in_new_tab(text)  # auto-generates versioned title
            self.statusBar().showMessage("AI output opened in new editor tab.")

    def _on_load_history_item(self) -> None:
        """Reload a saved AI session back into the Generate tab."""
        items = self._history_list.selectedItems()
        if not items:
            return
        path_str = items[0].data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        try:
            data = self._vault.load_ai_session(Path(path_str))
            messages = data.get("messages", [])
            if not messages:
                return
            self._generate_chat_messages = list(messages)
            self._ai_output.clear()
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                cursor = self._ai_output.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                if self._ai_output.toPlainText():
                    cursor.insertText("\n\n")
                if role == "user":
                    cursor.insertText(f"[Prompt]\n{content}\n\n[Response]\n")
                elif role == "assistant":
                    cursor.insertText(content)
                self._ai_output.setTextCursor(cursor)
            self._ai_output.ensureCursorVisible()
            has_chat = len(messages) >= 2
            self._followup_input.setEnabled(has_chat)
            self._followup_btn.setEnabled(has_chat)
            self._ai_tabs.setCurrentIndex(0)
            self.statusBar().showMessage(f"Session loaded: {Path(path_str).stem}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def _on_delete_history_item(self) -> None:
        """Delete the selected AI session file from the vault."""
        items = self._history_list.selectedItems()
        if not items:
            return
        path_str = items[0].data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        p = Path(path_str)
        reply = QMessageBox.question(
            self,
            "Delete Session",
            f"Permanently delete session '{p.stem}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            p.unlink()
            self._refresh_history_tab()
            self.statusBar().showMessage(f"Deleted: {p.stem}")
        except Exception as exc:
            QMessageBox.critical(self, "Delete Error", str(exc))

    # ------------------------------------------------------------------
    # Knowledge base — source tab combo
    # ------------------------------------------------------------------

    def _refresh_kb_source_combo(self) -> None:
        """Repopulate the KB source tab combo with current editor tabs."""
        from app.editor import EditorWidget
        prev_idx = self._kb_source_combo.currentData()
        self._kb_source_combo.blockSignals(True)
        self._kb_source_combo.clear()
        for i in range(self._editor.tab_count()):
            w = self._editor.tab_widget_at(i)
            if isinstance(w, EditorWidget):
                title = self._editor._tabs.tabText(i)
                self._kb_source_combo.addItem(title, userData=i)
        # Restore previous selection if it still exists
        if prev_idx is not None:
            for j in range(self._kb_source_combo.count()):
                if self._kb_source_combo.itemData(j) == prev_idx:
                    self._kb_source_combo.setCurrentIndex(j)
                    break
        self._kb_source_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Knowledge base — build
    # ------------------------------------------------------------------

    def _on_build_kb(self) -> None:
        # Use whichever editor tab the source combo points to
        source_idx = self._kb_source_combo.currentData()
        if source_idx is not None:
            text = self._editor.get_tab_text(source_idx).strip()
        else:
            text = self._editor.get_text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Editor", "Add text to the editor before building a KB.")
            return

        embed_model = self._embed_model_input.text().strip() or DEFAULT_EMBED_MODEL
        name = self._file_name_input.get_name()

        self._kb_build_btn.setEnabled(False)
        self._kb_load_btn.setEnabled(False)
        self._kb_save_btn.setEnabled(False)
        self._kb_progress.setVisible(True)
        self._kb_progress.setValue(0)
        self._kb_status.setText(f"Building knowledge base '{name}'…")
        self.statusBar().showMessage("Building knowledge base…")

        self._kb_build_worker = KBBuildWorker(self._ollama, text, name, embed_model)
        self._kb_build_worker.signals.progress.connect(self._on_kb_build_progress)
        self._kb_build_worker.signals.result.connect(self._on_kb_built)
        self._kb_build_worker.signals.error.connect(self._on_kb_build_error)
        self._kb_build_worker.signals.finished.connect(self._on_kb_build_finished)
        self._pool.start(self._kb_build_worker)

    def _on_kb_build_progress(self, current: int, total: int) -> None:
        self._kb_progress.setMaximum(total)
        self._kb_progress.setValue(current)
        self._kb_status.setText(f"Embedding chunk {current}/{total}…")

    def _on_kb_built(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        n = len(kb.chunks)
        self._kb_status.setText(
            f"KB ready: '{kb.name}' — {n} chunk{'s' if n != 1 else ''} "
            f"({kb.embedding_model})"
        )
        self._kb_save_btn.setEnabled(True)
        self._kb_ask_btn.setEnabled(True)
        self._kb_chat.clear()
        self.statusBar().showMessage(f"Knowledge base built: {n} chunks.")
        # Auto-save to vault so the KB survives session close/reopen
        if self._vault:
            try:
                kb_path = self._vault.kb_dir / f"{kb.name}.vkb"
                kb_path.parent.mkdir(parents=True, exist_ok=True)
                kb.save(kb_path)
                self._kb_current_path = kb_path
                logger.info("KB auto-saved: %s", kb_path)
            except Exception as exc:
                logger.warning("KB auto-save failed: %s", exc)

    def _on_kb_build_error(self, error: str) -> None:
        QMessageBox.critical(self, "KB Build Error", error)
        self._kb_status.setText("Build failed.")
        self.statusBar().showMessage("KB build failed.")

    def _on_kb_build_finished(self) -> None:
        self._kb_build_btn.setEnabled(True)
        self._kb_load_btn.setEnabled(True)
        self._kb_progress.setVisible(False)
        self._kb_build_worker = None

    # ------------------------------------------------------------------
    # Knowledge base — save / load
    # ------------------------------------------------------------------

    def _on_save_kb(self) -> None:
        if self._kb is None:
            return
        default_dir = self._vault.kb_dir if self._vault else self._settings.kb_dir
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Knowledge Base",
            str(default_dir / f"{self._kb.name}.vkb"),
            "Veritas Knowledge Base (*.vkb)",
        )
        if not path:
            return
        try:
            self._kb.save(Path(path))
            self.statusBar().showMessage(f"KB saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_load_kb(self) -> None:
        self._settings.kb_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Knowledge Base",
            str(self._settings.kb_dir),
            "Veritas Knowledge Base (*.vkb);;All Files (*)",
        )
        if not path:
            return
        try:
            self._kb = KnowledgeBase.load(Path(path))
            self._kb_current_path = Path(path)
            n = len(self._kb.chunks)
            self._kb_status.setText(
                f"KB loaded: '{self._kb.name}' — {n} chunk{'s' if n != 1 else ''} "
                f"({self._kb.embedding_model})"
            )
            self._kb_save_btn.setEnabled(True)
            self._kb_ask_btn.setEnabled(True)
            self._kb_chat.clear()
            self.statusBar().showMessage(f"KB loaded: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    # ------------------------------------------------------------------
    # Knowledge base — chat
    # ------------------------------------------------------------------

    def _on_ask_kb(self) -> None:
        if self._kb is None:
            return
        question = self._kb_question.text().strip()
        if not question:
            return

        model = self._model_selector.current_model()
        if not model:
            QMessageBox.warning(self, "No Model", "Select a model in the Generate tab first.")
            return

        self._kb_question.clear()
        self._kb_ask_btn.setEnabled(False)
        self._kb_stop_btn.setEnabled(True)
        self.statusBar().showMessage("Querying knowledge base…")

        # Track conversation for vault history
        self._kb_chat_messages.append({"role": "user", "content": question})
        self._kb_response_buffer = ""

        # Append user turn to chat display
        cursor = self._kb_chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if self._kb_chat.toPlainText():
            cursor.insertText("\n\n")
        cursor.insertText(f"You: {question}\n\nAssistant: ")
        self._kb_chat.setTextCursor(cursor)
        self._kb_chat.ensureCursorVisible()

        self._kb_query_worker = KBQueryWorker(self._ollama, self._kb, question, model)
        self._kb_query_worker.signals.token.connect(self._on_kb_token)
        self._kb_query_worker.signals.error.connect(self._on_kb_query_error)
        self._kb_query_worker.signals.finished.connect(self._on_kb_query_finished)
        self._pool.start(self._kb_query_worker)

    def _on_stop_kb_query(self) -> None:
        if self._kb_query_worker is not None:
            self._kb_query_worker.cancel()
        self._kb_stop_btn.setEnabled(False)
        self.statusBar().showMessage("Stopping KB query…")

    def _on_kb_token(self, token: str) -> None:
        self._kb_response_buffer += token
        cursor = self._kb_chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self._kb_chat.setTextCursor(cursor)
        self._kb_chat.ensureCursorVisible()

    def _on_kb_query_error(self, error: str) -> None:
        QMessageBox.critical(self, "KB Query Error", error)
        self.statusBar().showMessage("KB query failed.")

    def _on_kb_query_finished(self) -> None:
        self._kb_ask_btn.setEnabled(True)
        self._kb_stop_btn.setEnabled(False)
        self._kb_query_worker = None
        # Append separator after response
        cursor = self._kb_chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText("\n\n" + "─" * 40)
        self._kb_chat.setTextCursor(cursor)
        # Persist conversation turn to vault
        if self._kb_response_buffer.strip():
            self._kb_chat_messages.append(
                {"role": "assistant", "content": self._kb_response_buffer.strip()}
            )
        self._kb_response_buffer = ""
        if self._vault and len(self._kb_chat_messages) >= 2:
            self._vault.save_ai_session(
                session_type="kb_chat",
                model=self._model_selector.current_model() or "unknown",
                messages=self._kb_chat_messages,
                document_name=self._file_name_input.get_name(),
            )
            self._refresh_history_tab()
        self.statusBar().showMessage("Ready")

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

        base_name = self._file_name_input.get_name()
        if self._vault:
            audio_dir = self._vault.audio_dir
            audio_dir.mkdir(parents=True, exist_ok=True)
            raw_path = self._vault.audio_path(f"{base_name}_raw.wav")
        else:
            temp_dir = self._settings.temp_dir
            temp_dir.mkdir(parents=True, exist_ok=True)
            raw_path = temp_dir / f"{base_name}_raw.wav"

        self._tts_btn.setEnabled(False)
        self.statusBar().showMessage("Synthesizing audio…")

        engine = self._gptsovits_engine if (voice and voice.engine == "gptsovits") else self._tts_engine
        self._current_voice_engine = voice.engine if voice else "kokoro"
        self._tts_worker = TTSWorker(
            engine,
            text,
            raw_path,
            voice,
            speed=self._pacing.speed(),
            pause_sentence_ms=self._pacing.sentence_ms(),
            pause_mdash_ms=self._pacing.mdash_ms(),
            pause_paragraph_ms=self._pacing.paragraph_ms(),
        )
        self._tts_worker.signals.result.connect(self._on_tts_complete)
        self._tts_worker.signals.error.connect(self._on_tts_error)
        self._pool.start(self._tts_worker)

    def _on_tts_complete(self, raw_path_str: str) -> None:
        raw_path = Path(raw_path_str)

        # GPT-SoVITS handles its own pacing and timing — skip silence removal and
        # compression which clip words, but still apply EQ and loudness normalisation.
        if getattr(self, "_current_voice_engine", "kokoro") == "gptsovits":
            eq_path   = raw_path.with_name(raw_path.stem + "_eq.wav")
            norm_path = raw_path.with_name(raw_path.stem + "_norm.wav")
            def _eq_then_norm(src, eq, norm):
                apply_eq(src, eq, highpass_hz=80.0, presence_hz=6000.0, presence_gain_db=1.5)
                return normalize_loudness(eq, norm)
            worker = Worker(_eq_then_norm, raw_path, eq_path, norm_path)
            worker.signals.result.connect(self._on_processing_complete)
            worker.signals.error.connect(self._on_tts_error)
            self._pool.start(worker)
            return

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
        self._tts_worker = None
        self._player.load(self._current_audio_path)
        self._tts_btn.setEnabled(True)
        self.statusBar().showMessage(f"Audio ready: {self._current_audio_path.name}")

    def _on_tts_error(self, error: str) -> None:
        self._tts_worker = None
        QMessageBox.critical(self, "TTS Error", error)
        self._tts_btn.setEnabled(True)
        self.statusBar().showMessage("TTS failed.")

    # ------------------------------------------------------------------
    # GPT-SoVITS-v2 offline render
    # ------------------------------------------------------------------

    def _update_render_profile_tooltip(self) -> None:
        path = self._settings.render_profile_path
        if path:
            tip = f"Voice profile: {Path(path).name}\nClick to change"
        else:
            tip = "No voice profile set — click to choose"
        self._render_profile_btn.setToolTip(tip)

    def _on_pick_render_profile(self) -> None:
        start = self._settings.render_profile_path or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Voice Profile Directory", start
        )
        if not folder:
            return
        self._settings.render_profile_path = folder
        self._update_render_profile_tooltip()
        self.statusBar().showMessage(f"Render profile: {Path(folder).name}")

    def _on_render(self) -> None:
        profile_path = self._settings.render_profile_path
        if not profile_path:
            QMessageBox.warning(
                self,
                "No Voice Profile",
                "Choose a GPT-SoVITS-v2 voice profile first (⋯ button).",
            )
            return

        text = self._editor.get_text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Editor", "Nothing to render.")
            return

        # Split into paragraphs on blank lines; fall back to single-newline split
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        base_name = self._file_name_input.get_name() or "render"
        default_dir = str(
            self._vault.audio_dir if self._vault else self._settings.output_dir
        )
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose Output Directory for Rendered WAVs", default_dir
        )
        if not out_dir:
            return

        output_prefix = Path(out_dir) / base_name

        self._render_btn.setEnabled(False)
        self._render_btn.setText("Rendering…")
        self.statusBar().showMessage(
            f"Rendering {len(paragraphs)} paragraph(s) via GPT-SoVITS-v2…"
        )

        self._render_worker = RenderWorker(paragraphs, output_prefix, profile_path)
        self._render_worker.signals.progress.connect(self._on_render_progress)
        self._render_worker.signals.result.connect(self._on_render_done)
        self._render_worker.signals.error.connect(self._on_render_error)
        self._render_worker.signals.finished.connect(
            lambda: self._render_btn.__setattr__("_finished", True)
        )
        self._pool.start(self._render_worker)

    def _on_render_progress(self, current: int, total: int, filename: str) -> None:
        self._render_btn.setText(f"Rendering {current}/{total}")
        self.statusBar().showMessage(f"Rendering {current}/{total}: {filename}")

    def _on_render_done(self, paths: list) -> None:
        self._render_worker = None
        self._render_btn.setEnabled(True)
        self._render_btn.setText("Render")
        n = len(paths)
        self.statusBar().showMessage(f"Render complete — {n} file{'s' if n != 1 else ''} written.")
        QMessageBox.information(
            self,
            "Render Complete",
            f"{n} WAV file{'s' if n != 1 else ''} written to:\n{Path(paths[0]).parent}",
        )

    def _on_render_error(self, error: str) -> None:
        self._render_worker = None
        self._render_btn.setEnabled(True)
        self._render_btn.setText("Render")
        QMessageBox.critical(self, "Render Error", error)
        self.statusBar().showMessage("Render failed.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_commit_version(self) -> None:
        """Save the primary editor tab content as a new versioned file in the vault."""
        text = self._editor.get_main_text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Editor", "Nothing to commit.")
            return
        if self._vault is None:
            QMessageBox.warning(
                self,
                "No Vault",
                "Open a document first so a vault is created, then commit.",
            )
            return
        suffix = self._current_file_path.suffix if self._current_file_path else ".md"
        version_path = self._vault.next_version_path(suffix)
        try:
            version_path.parent.mkdir(parents=True, exist_ok=True)
            version_path.write_text(text, encoding="utf-8")
            self.statusBar().showMessage(f"Version committed: {version_path.name}")
            logger.info("Committed version: %s", version_path)
        except Exception as exc:
            QMessageBox.critical(self, "Commit Error", str(exc))

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
    # Session — collect / save / restore
    # ------------------------------------------------------------------

    def _collect_session(self) -> dict:
        """Snapshot complete workspace state into a JSON-serialisable dict."""
        from app.web_tab import BrowserTab

        # Editor tabs
        tabs_data = []
        for i in range(self._editor.tab_count()):
            w = self._editor.tab_widget_at(i)
            title = self._editor._tabs.tabText(i)
            if isinstance(w, BrowserTab):
                tabs_data.append({
                    "type": "browser",
                    "title": title,
                    "content": "",
                    "url": w.current_url(),
                    "is_primary": (i == 0),
                })
            else:
                tabs_data.append({
                    "type": "editor",
                    "title": title,
                    "content": w.get_text() if hasattr(w, "get_text") else "",
                    "url": "",
                    "is_primary": (i == 0),
                })

        # AI browser tabs (indices >= 3)
        ai_browser_tabs = []
        for i in range(3, self._ai_tabs.count()):
            w = self._ai_tabs.widget(i)
            ai_browser_tabs.append({
                "title": self._ai_tabs.tabText(i),
                "url": w.current_url() if isinstance(w, BrowserTab) else "",
            })

        return {
            "schema_version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "project": {
                "name": self._file_name_input.get_name(),
                "vault_path": str(self._vault.root) if self._vault else "",
                "file_path": str(self._current_file_path) if self._current_file_path else "",
            },
            "editor": {
                "file_name": self._file_name_input.get_name(),
                "active_tab": self._editor.active_tab_index(),
                "tabs": tabs_data,
            },
            "ai": {
                "panel_visible": self._ai_panel.isVisible(),
                "active_tab": self._ai_tabs.currentIndex(),
                "tts_bar_visible": self._tts_bar.isVisible(),
                "splitter_sizes": self._splitter.sizes(),
                "browser_tabs": ai_browser_tabs,
                "generate": {
                    "model": self._model_selector.current_model() or "",
                    "prompt": self._prompt_input.toPlainText(),
                    "output": self._ai_output.toPlainText(),
                    "messages": list(self._generate_chat_messages),
                },
                "kb": {
                    "embed_model": self._embed_model_input.text(),
                    "kb_path": str(self._kb_current_path) if self._kb_current_path else "",
                    "chat_text": self._kb_chat.toPlainText(),
                    "chat_messages": list(self._kb_chat_messages),
                },
            },
        }

    def _save_session(self) -> None:
        """Write session.json to the active vault; no-op if no vault."""
        if self._vault is None:
            return
        try:
            self._vault.save_session(self._collect_session())
        except Exception as exc:
            logger.warning("Session save failed: %s", exc)

    def _restore_session(self, data: dict) -> None:
        """Apply a session dict to the workspace.

        Disconnects text_changed → _mark_dirty for the duration so restoring
        content does not spuriously mark the document dirty.
        """
        try:
            self._editor.text_changed.disconnect(self._mark_dirty)
        except TypeError:
            pass

        try:
            self._restore_editor_tabs(data.get("editor", {}))
            self._restore_ai_state(data.get("ai", {}))

            proj = data.get("project", {})
            file_name = data.get("editor", {}).get("file_name", "") or proj.get("name", "")
            if file_name:
                self._file_name_input.set_name(file_name)

            file_path_str = proj.get("file_path", "")
            if file_path_str:
                self._current_file_path = Path(file_path_str)
                self._settings.last_file_path = file_path_str
                self._autosave_path = self._compute_autosave_path(self._current_file_path)

            self._dirty = False
            # Auto-restore KB from vault if available
            self._try_restore_kb(data.get("ai", {}).get("kb", {}).get("kb_path", ""))
            self.statusBar().showMessage(
                f"Session restored: {proj.get('name', 'project')}"
            )
        except Exception as exc:
            logger.warning("Session restore failed: %s", exc)
            self.statusBar().showMessage("Session restore failed — starting fresh.")
        finally:
            self._editor.text_changed.connect(self._mark_dirty)

    def _try_restore_kb(self, saved_kb_path: str) -> None:
        """Auto-load the KB for the current project on session restore."""
        # Prefer the explicitly-saved path; fall back to the default vault KB location
        kb_path: Path | None = None
        if saved_kb_path:
            p = Path(saved_kb_path)
            if p.exists():
                kb_path = p
        if kb_path is None and self._vault:
            name = self._file_name_input.get_name()
            if name:
                candidate = self._vault.kb_dir / f"{name}.vkb"
                if candidate.exists():
                    kb_path = candidate
        if kb_path is None:
            return
        try:
            self._kb = KnowledgeBase.load(kb_path)
            self._kb_current_path = kb_path
            n = len(self._kb.chunks)
            self._kb_status.setText(
                f"KB: '{self._kb.name}' — {n} chunk{'s' if n != 1 else ''} (auto-loaded)"
            )
            self._kb_save_btn.setEnabled(True)
            self._kb_ask_btn.setEnabled(True)
            logger.info("KB auto-loaded: %s", kb_path)
        except Exception as exc:
            logger.warning("KB auto-load failed: %s", exc)

    def _restore_editor_tabs(self, editor_data: dict) -> None:
        """Recreate editor tabs from session data without prompting for saves."""
        from app.web_tab import BrowserTab

        tabs = editor_data.get("tabs", [])
        if not tabs:
            return

        # Restore primary tab (index 0, always type="editor")
        primary = next((t for t in tabs if t.get("is_primary")), tabs[0])
        if primary.get("type") == "editor":
            primary_widget = self._editor.tab_widget_at(0)
            if hasattr(primary_widget, "set_text"):
                primary_widget.set_text(primary.get("content", ""))
            self._editor.set_main_tab_title(primary.get("title", "Document"))

        # Remove any extra tabs silently (no unsaved-content prompts during restore)
        while self._editor.tab_count() > 1:
            self._editor._tabs.removeTab(self._editor.tab_count() - 1)

        # Recreate tabs 1+
        for tab_data in tabs[1:]:
            if tab_data.get("type") == "browser":
                url = tab_data.get("url", "about:blank")
                self._editor.open_url_tab(url, tab_data.get("title", url))
            else:
                self._editor.open_in_new_tab(
                    tab_data.get("content", ""),
                    tab_data.get("title", "Draft"),
                )

        active = editor_data.get("active_tab", 0)
        self._editor.set_active_tab(active)

    def _restore_ai_state(self, ai_data: dict) -> None:
        """Restore AI panel generate/KB state, browser tabs, and UI visibility."""
        from app.web_tab import BrowserTab

        if not ai_data:
            return

        # Generate tab
        gen = ai_data.get("generate", {})
        if gen.get("prompt"):
            self._prompt_input.setPlainText(gen["prompt"])
        if gen.get("output"):
            self._ai_output.setPlainText(gen["output"])
        self._generate_chat_messages = list(gen.get("messages", []))
        has_chat = len(self._generate_chat_messages) >= 2
        self._followup_input.setEnabled(has_chat)
        self._followup_btn.setEnabled(has_chat)

        # Model — defer because ModelSelectorCombo loads models asynchronously
        model = gen.get("model", "")
        if model:
            self._try_set_model(model)

        # KB Chat tab
        kb = ai_data.get("kb", {})
        if kb.get("embed_model"):
            self._embed_model_input.setText(kb["embed_model"])
        if kb.get("chat_text"):
            self._kb_chat.setPlainText(kb["chat_text"])
        self._kb_chat_messages = list(kb.get("chat_messages", []))

        # AI browser tabs (> the 3 permanent ones)
        for tab_data in ai_data.get("browser_tabs", []):
            url = tab_data.get("url", "about:blank")
            title = tab_data.get("title", url)
            if url:
                browser = BrowserTab(url)
                self._ai_tabs.addTab(browser, title)

        # Active AI sub-tab
        active_ai_tab = ai_data.get("active_tab", 0)
        if 0 <= active_ai_tab < self._ai_tabs.count():
            self._ai_tabs.setCurrentIndex(active_ai_tab)

        # TTS bar
        if ai_data.get("tts_bar_visible"):
            self._tts_bar.show()
            self._tts_toggle_btn.setText("TTS ✕")
        else:
            self._tts_bar.hide()
            self._tts_toggle_btn.setText("TTS")

        # AI panel + splitter (defer setSizes so Qt layout settles first)
        panel_visible = ai_data.get("panel_visible", False)
        splitter_sizes = ai_data.get("splitter_sizes")
        if panel_visible:
            self._ai_panel.show()
            self._ai_toggle_btn.setText("✦ AI ✕")
            if splitter_sizes and len(splitter_sizes) == 2:
                sizes = splitter_sizes
                QTimer.singleShot(0, lambda: self._splitter.setSizes(sizes))
        else:
            self._ai_panel.hide()
            self._ai_toggle_btn.setText("✦ AI")

    def _try_set_model(self, model: str, attempt: int = 0) -> None:
        """Select *model* in the combo, retrying while models are still loading."""
        self._model_selector.set_preferred(model)
        # If the combo still shows "Loading…", poll up to 10× at 300 ms intervals
        if attempt < 10 and self._model_selector.current_model() is None:
            QTimer.singleShot(300, lambda: self._try_set_model(model, attempt + 1))

    # ------------------------------------------------------------------
    # Session — startup restore and project management
    # ------------------------------------------------------------------

    def _restore_last_session(self) -> None:
        """On startup: restore the last session.json, falling back to legacy file reopen."""
        vault_path_str = self._settings.last_vault_path
        if vault_path_str:
            vault_root_path = Path(vault_path_str)
            if vault_root_path.is_dir():
                vault = Vault(vault_root_path.parent, vault_root_path.name)
                data = vault.load_session()
                if data and data.get("schema_version") == 1:
                    self._vault = vault
                    self._vault.init()
                    self._restore_session(data)
                    logger.info("Full session restored from %s", vault_root_path)
                    return
        # Fallback: legacy file-only restore
        self._reopen_last_file()

    def _on_open_project(self) -> None:
        """Show project picker and restore the selected project's session."""
        from app.dialogs.project_picker import ProjectPickerDialog
        dlg = ProjectPickerDialog(self._settings.vault_root, parent=self)
        if not dlg.exec():
            return
        selected = dlg.get_selected_vault_root()
        if not selected:
            return

        # Save current session before switching
        self._save_session()

        vault = Vault(selected.parent, selected.name)
        data = vault.load_session()
        if data is None:
            QMessageBox.warning(
                self, "No Session",
                f"Project '{selected.name}' has no saved session.",
            )
            return

        self._vault = vault
        self._vault.init()
        self._settings.last_vault_path = str(selected)
        self._refresh_history_tab()
        self._restore_session(data)

    def _on_close_project(self) -> None:
        """Save session and reset workspace to a clean state."""
        if self._vault is None and not self._dirty:
            self.statusBar().showMessage("No project open.")
            return

        self._autosave()
        self._save_session()

        # Remove extra editor tabs silently
        while self._editor.tab_count() > 1:
            self._editor._tabs.removeTab(self._editor.tab_count() - 1)

        # Remove AI browser tabs
        for i in range(self._ai_tabs.count() - 1, 2, -1):
            self._ai_tabs.removeTab(i)

        self._editor.set_text("")
        self._editor.set_main_tab_title("Document")
        self._file_name_input.set_name("")
        self._prompt_input.clear()
        self._ai_output.clear()
        self._kb_chat.clear()
        self._kb_status.setText("No knowledge base loaded.")
        self._kb_save_btn.setEnabled(False)
        self._kb_ask_btn.setEnabled(False)
        self._followup_input.setEnabled(False)
        self._followup_btn.setEnabled(False)
        self._history_list.clear()

        self._ai_panel.hide()
        self._ai_toggle_btn.setText("✦ AI")
        self._tts_bar.hide()
        self._tts_toggle_btn.setText("TTS")

        self._generate_chat_messages = []
        self._kb_chat_messages = []
        self._kb_response_buffer = ""
        self._stream_buffer = ""
        self._vault = None
        self._kb = None
        self._kb_current_path = None
        self._current_file_path = None
        self._autosave_path = None
        self._dirty = False
        self._settings.last_vault_path = ""
        self._settings.last_file_path = ""
        self._settings.last_autosave_path = ""

        self.statusBar().showMessage("Project closed.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geom = self._settings.window_geometry
        if geom:
            self.restoreGeometry(geom)

    def closeEvent(self, event) -> None:
        self._autosave()       # flush document content
        self._save_session()   # flush full workspace state
        self._settings.window_geometry = bytes(self.saveGeometry())
        self._settings.sync()
        super().closeEvent(event)
