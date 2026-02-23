"""Markdown / plain-text editor widget."""

import logging
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class EditorWidget(QWidget):
    """A markdown-aware editor with a simple formatting toolbar.

    Signals:
        text_changed: Emitted whenever the document content changes.
    """

    text_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # --- Formatting mini-toolbar ---
        fmt_bar = QToolBar()
        fmt_bar.setMovable(False)

        bold_act = fmt_bar.addAction("B")
        bold_act.setToolTip("Bold (Ctrl+B)")
        bold_act.triggered.connect(self._toggle_bold)

        italic_act = fmt_bar.addAction("I")
        italic_act.setToolTip("Italic (Ctrl+I)")
        italic_act.triggered.connect(self._toggle_italic)

        fmt_bar.addSeparator()

        h1_act = fmt_bar.addAction("H1")
        h1_act.setToolTip("Heading 1")
        h1_act.triggered.connect(lambda: self._insert_heading(1))

        h2_act = fmt_bar.addAction("H2")
        h2_act.setToolTip("Heading 2")
        h2_act.triggered.connect(lambda: self._insert_heading(2))

        h3_act = fmt_bar.addAction("H3")
        h3_act.setToolTip("Heading 3")
        h3_act.triggered.connect(lambda: self._insert_heading(3))

        fmt_bar.addSeparator()

        clear_act = fmt_bar.addAction("Clear")
        clear_act.setToolTip("Clear all text")
        clear_act.triggered.connect(self.clear)

        layout.addWidget(fmt_bar)

        # --- Main text area ---
        self._editor = QPlainTextEdit()
        self._editor.setFont(QFont("Monospace", 11))
        self._editor.setPlaceholderText(
            "Import a file, paste text, or generate content with AI to get started..."
        )
        self._editor.textChanged.connect(self.text_changed)
        layout.addWidget(self._editor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        """Replace editor content."""
        self._editor.setPlainText(text)

    def get_text(self) -> str:
        """Return current editor content as plain text."""
        return self._editor.toPlainText()

    def append_text(self, text: str) -> None:
        """Append text at the current cursor position."""
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._editor.setTextCursor(cursor)

    def clear(self) -> None:
        self._editor.clear()

    def is_empty(self) -> bool:
        return not self._editor.toPlainText().strip()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _toggle_bold(self) -> None:
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            selected = cursor.selectedText()
            if selected.startswith("**") and selected.endswith("**"):
                cursor.insertText(selected[2:-2])
            else:
                cursor.insertText(f"**{selected}**")
        else:
            cursor.insertText("****")
            cursor.movePosition(QTextCursor.MoveOperation.Left, n=2)
            self._editor.setTextCursor(cursor)

    def _toggle_italic(self) -> None:
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            selected = cursor.selectedText()
            if selected.startswith("_") and selected.endswith("_"):
                cursor.insertText(selected[1:-1])
            else:
                cursor.insertText(f"_{selected}_")
        else:
            cursor.insertText("__")
            cursor.movePosition(QTextCursor.MoveOperation.Left, n=1)
            self._editor.setTextCursor(cursor)

    def _insert_heading(self, level: int) -> None:
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.insertText("#" * level + " ")
        self._editor.setTextCursor(cursor)
