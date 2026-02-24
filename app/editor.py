"""Markdown / plain-text editor widget."""

import logging
import re

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_STYLE_ITEMS = ["Normal text", "Heading 1", "Heading 2", "Heading 3"]


class EditorWidget(QWidget):
    """A markdown-aware editor with a Substack-style formatting toolbar.

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

        # --- Formatting toolbar ---
        toolbar = QWidget()
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(2, 2, 2, 2)
        tl.setSpacing(4)

        # Style dropdown
        self._style_combo = QComboBox()
        self._style_combo.addItems(_STYLE_ITEMS)
        self._style_combo.setFixedWidth(120)
        self._style_combo.setToolTip("Paragraph style")
        self._style_combo.currentIndexChanged.connect(self._on_style_changed)
        tl.addWidget(self._style_combo)

        tl.addSpacing(6)

        # Bold
        bold_btn = QPushButton("B")
        bold_btn.setFixedWidth(28)
        bold_btn.setToolTip("Bold")
        bold_btn.setStyleSheet("font-weight: bold;")
        bold_btn.clicked.connect(self._toggle_bold)
        tl.addWidget(bold_btn)

        # Italic
        italic_btn = QPushButton("I")
        italic_btn.setFixedWidth(28)
        italic_btn.setToolTip("Italic")
        italic_btn.setStyleSheet("font-style: italic;")
        italic_btn.clicked.connect(self._toggle_italic)
        tl.addWidget(italic_btn)

        # Strikethrough
        strike_btn = QPushButton("S")
        strike_btn.setFixedWidth(28)
        strike_btn.setToolTip("Strikethrough")
        strike_btn.setStyleSheet("text-decoration: line-through;")
        strike_btn.clicked.connect(self._toggle_strikethrough)
        tl.addWidget(strike_btn)

        # Visual separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        tl.addWidget(sep)

        # Blockquote
        quote_btn = QPushButton('"')
        quote_btn.setFixedWidth(28)
        quote_btn.setToolTip("Blockquote")
        quote_btn.clicked.connect(lambda: self._toggle_prefix("> "))
        tl.addWidget(quote_btn)

        # Bullet list
        bullet_btn = QPushButton("•")
        bullet_btn.setFixedWidth(28)
        bullet_btn.setToolTip("Bullet list")
        bullet_btn.clicked.connect(lambda: self._toggle_prefix("- "))
        tl.addWidget(bullet_btn)

        # Horizontal divider
        div_btn = QPushButton("—")
        div_btn.setFixedWidth(28)
        div_btn.setToolTip("Horizontal divider")
        div_btn.clicked.connect(self._insert_divider)
        tl.addWidget(div_btn)

        tl.addStretch()
        layout.addWidget(toolbar)

        # --- Main text area ---
        self._editor = QPlainTextEdit()
        self._editor.setFont(QFont("Monospace", 11))
        self._editor.setPlaceholderText(
            "Import a file, paste text, or generate content with AI to get started..."
        )
        self._editor.textChanged.connect(self.text_changed)
        self._editor.cursorPositionChanged.connect(self._update_style_combo)
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
    # Style dropdown helpers
    # ------------------------------------------------------------------

    def _current_heading_level(self) -> int:
        """Return 0 for normal text, 1–3 for H1–H3."""
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        line = cursor.selectedText()
        for level in (3, 2, 1):
            if line.startswith("#" * level + " "):
                return level
        return 0

    def _set_heading(self, level: int) -> None:
        """Apply heading level to the current line, replacing any existing one."""
        self._remove_heading()
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.insertText("#" * level + " ")
        self._editor.setTextCursor(cursor)

    def _remove_heading(self) -> None:
        """Strip any leading # prefix from the current line."""
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        line = cursor.selectedText()
        stripped = re.sub(r"^#{1,3} ", "", line)
        if stripped != line:
            cursor.insertText(stripped)
            self._editor.setTextCursor(cursor)

    def _update_style_combo(self) -> None:
        """Sync style combo to the heading level of the current line."""
        level = self._current_heading_level()
        self._style_combo.blockSignals(True)
        self._style_combo.setCurrentIndex(level)  # 0=Normal, 1=H1, 2=H2, 3=H3
        self._style_combo.blockSignals(False)

    def _on_style_changed(self, index: int) -> None:
        if index == 0:
            self._remove_heading()
        else:
            self._set_heading(index)

    # ------------------------------------------------------------------
    # Inline formatting helpers
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

    def _toggle_strikethrough(self) -> None:
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            selected = cursor.selectedText()
            if selected.startswith("~~") and selected.endswith("~~"):
                cursor.insertText(selected[2:-2])
            else:
                cursor.insertText(f"~~{selected}~~")
        else:
            cursor.insertText("~~~~")
            cursor.movePosition(QTextCursor.MoveOperation.Left, n=2)
            self._editor.setTextCursor(cursor)

    def _toggle_prefix(self, prefix: str) -> None:
        """Toggle a line-start prefix (e.g. '> ' or '- ')."""
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        line = cursor.selectedText()
        if line.startswith(prefix):
            cursor.insertText(line[len(prefix):])
        else:
            cursor.insertText(prefix + line)
        self._editor.setTextCursor(cursor)

    def _insert_divider(self) -> None:
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine)
        cursor.insertText("\n\n---\n\n")
        self._editor.setTextCursor(cursor)
