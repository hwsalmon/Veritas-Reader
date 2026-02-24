"""Markdown / plain-text editor widget with Markup ⇄ Formatted toggle."""

import logging
import re

import markdown2
from markdownify import markdownify as html_to_md

from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import (
    QFont,
    QTextCharFormat,
    QTextBlockFormat,
    QTextListFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_STYLE_ITEMS = ["Normal text", "Heading 1", "Heading 2", "Heading 3"]
# Font sizes for formatted-mode headings
_HEADING_SIZES = {0: 12, 1: 24, 2: 18, 3: 14}


class EditorWidget(QWidget):
    """Editor widget with Markup (raw markdown) and Formatted (rich text) modes.

    Toolbar buttons work in both modes.  Switching modes converts content
    bi-directionally using markdown2 and markdownify.

    Signals:
        text_changed: Emitted whenever the document content changes.
    """

    text_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode = "markup"  # "markup" | "formatted"
        self._build_ui()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

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

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        tl.addWidget(sep1)

        # Blockquote / indent block
        quote_btn = QPushButton('"')
        quote_btn.setFixedWidth(28)
        quote_btn.setToolTip("Blockquote")
        quote_btn.clicked.connect(self._toggle_quote)
        tl.addWidget(quote_btn)

        # Bullet list
        bullet_btn = QPushButton("•")
        bullet_btn.setFixedWidth(28)
        bullet_btn.setToolTip("Bullet list")
        bullet_btn.clicked.connect(self._toggle_bullet)
        tl.addWidget(bullet_btn)

        # Increase indent
        indent_btn = QPushButton("→|")
        indent_btn.setFixedWidth(32)
        indent_btn.setToolTip("Increase indent  (Tab)")
        indent_btn.clicked.connect(self._increase_indent)
        tl.addWidget(indent_btn)

        # Decrease indent
        outdent_btn = QPushButton("|←")
        outdent_btn.setFixedWidth(32)
        outdent_btn.setToolTip("Decrease indent  (Shift+Tab)")
        outdent_btn.clicked.connect(self._decrease_indent)
        tl.addWidget(outdent_btn)

        # Horizontal divider
        div_btn = QPushButton("—")
        div_btn.setFixedWidth(28)
        div_btn.setToolTip("Horizontal divider")
        div_btn.clicked.connect(self._insert_divider)
        tl.addWidget(div_btn)

        tl.addStretch()

        # Mode toggle
        self._mode_btn = QPushButton("Formatted")
        self._mode_btn.setToolTip("Switch to formatted (rich text) view")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(False)
        self._mode_btn.clicked.connect(self._toggle_mode)
        tl.addWidget(self._mode_btn)

        layout.addWidget(toolbar)

        # --- Stacked editor panes ---
        self._stack = QStackedWidget()

        # Page 0: Markup (QPlainTextEdit)
        self._markup_editor = QPlainTextEdit()
        self._markup_editor.setFont(QFont("Monospace", 11))
        self._markup_editor.setPlaceholderText(
            "Import a file, paste text, or generate content with AI to get started..."
        )
        self._markup_editor.textChanged.connect(self.text_changed)
        self._markup_editor.cursorPositionChanged.connect(self._update_style_combo)
        self._markup_editor.installEventFilter(self)

        # Page 1: Formatted (QTextEdit)
        self._rich_editor = QTextEdit()
        self._rich_editor.setFont(QFont("Georgia", 12))
        self._rich_editor.setPlaceholderText(
            "Formatted view — edit here or switch to Markup to see raw text."
        )
        self._rich_editor.textChanged.connect(self.text_changed)
        self._rich_editor.cursorPositionChanged.connect(self._update_style_combo)

        self._stack.addWidget(self._markup_editor)   # index 0
        self._stack.addWidget(self._rich_editor)     # index 1
        layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _toggle_mode(self) -> None:
        if self._mode == "markup":
            self._switch_to_formatted()
        else:
            self._switch_to_markup()

    def _switch_to_formatted(self) -> None:
        md = self._markup_editor.toPlainText()
        html = markdown2.markdown(
            md,
            extras=["tables", "fenced-code-blocks", "header-ids"],
        )
        self._rich_editor.setHtml(html)
        self._stack.setCurrentIndex(1)
        self._mode = "formatted"
        self._mode_btn.setText("Markup")
        self._mode_btn.setToolTip("Switch to markup (raw text) view")
        self._mode_btn.setChecked(True)

    def _switch_to_markup(self) -> None:
        html = self._rich_editor.toHtml()
        md = html_to_md(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["a"],
        ).strip()
        self._markup_editor.setPlainText(md)
        self._stack.setCurrentIndex(0)
        self._mode = "markup"
        self._mode_btn.setText("Formatted")
        self._mode_btn.setToolTip("Switch to formatted (rich text) view")
        self._mode_btn.setChecked(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        """Replace editor content.  Always accepts markdown source."""
        self._markup_editor.setPlainText(text)
        if self._mode == "formatted":
            html = markdown2.markdown(
                text,
                extras=["tables", "fenced-code-blocks", "header-ids"],
            )
            self._rich_editor.setHtml(html)

    def get_text(self) -> str:
        """Return current content as markdown (from whichever pane is active)."""
        if self._mode == "markup":
            return self._markup_editor.toPlainText()
        else:
            html = self._rich_editor.toHtml()
            return html_to_md(
                html,
                heading_style="ATX",
                bullets="-",
                strip=["a"],
            ).strip()

    def append_text(self, text: str) -> None:
        """Append text at the end of the markup editor."""
        cursor = self._markup_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._markup_editor.setTextCursor(cursor)

    def clear(self) -> None:
        self._markup_editor.clear()
        self._rich_editor.clear()

    def is_empty(self) -> bool:
        return not self.get_text().strip()

    # ------------------------------------------------------------------
    # Event filter — Tab / Shift+Tab for indent in markup mode
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        if obj is self._markup_editor and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            mods = event.modifiers()
            if key == Qt.Key.Key_Tab:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self._decrease_indent()
                else:
                    self._increase_indent()
                return True
            if key == Qt.Key.Key_Backtab:
                self._decrease_indent()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Style dropdown
    # ------------------------------------------------------------------

    def _current_heading_level(self) -> int:
        """Return 0 for normal text, 1–3 for H1–H3 (markup mode)."""
        cursor = self._markup_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        line = cursor.selectedText()
        for level in (3, 2, 1):
            if line.startswith("#" * level + " "):
                return level
        return 0

    def _set_heading_markup(self, level: int) -> None:
        self._remove_heading_markup()
        cursor = self._markup_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.insertText("#" * level + " ")
        self._markup_editor.setTextCursor(cursor)

    def _remove_heading_markup(self) -> None:
        cursor = self._markup_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        line = cursor.selectedText()
        stripped = re.sub(r"^#{1,3} ", "", line)
        if stripped != line:
            cursor.insertText(stripped)
            self._markup_editor.setTextCursor(cursor)

    def _update_style_combo(self) -> None:
        if self._mode == "markup":
            idx = self._current_heading_level()
        else:
            level = self._rich_editor.textCursor().blockFormat().headingLevel()
            idx = min(level, 3)
        self._style_combo.blockSignals(True)
        self._style_combo.setCurrentIndex(idx)
        self._style_combo.blockSignals(False)

    def _on_style_changed(self, index: int) -> None:
        if self._mode == "markup":
            if index == 0:
                self._remove_heading_markup()
            else:
                self._set_heading_markup(index)
        else:
            cursor = self._rich_editor.textCursor()
            block_fmt = QTextBlockFormat()
            block_fmt.setHeadingLevel(index)
            char_fmt = QTextCharFormat()
            char_fmt.setFontPointSize(_HEADING_SIZES.get(index, 12))
            char_fmt.setFontWeight(
                QFont.Weight.Bold if index > 0 else QFont.Weight.Normal
            )
            cursor.mergeBlockFormat(block_fmt)
            cursor.mergeCharFormat(char_fmt)
            self._rich_editor.setTextCursor(cursor)

    # ------------------------------------------------------------------
    # Inline formatting
    # ------------------------------------------------------------------

    def _toggle_bold(self) -> None:
        if self._mode == "markup":
            cursor = self._markup_editor.textCursor()
            if cursor.hasSelection():
                sel = cursor.selectedText()
                cursor.insertText(sel[2:-2] if (sel.startswith("**") and sel.endswith("**")) else f"**{sel}**")
            else:
                cursor.insertText("****")
                cursor.movePosition(QTextCursor.MoveOperation.Left, n=2)
                self._markup_editor.setTextCursor(cursor)
        else:
            fmt = QTextCharFormat()
            is_bold = self._rich_editor.fontWeight() >= 700
            fmt.setFontWeight(QFont.Weight.Normal if is_bold else QFont.Weight.Bold)
            self._rich_editor.mergeCurrentCharFormat(fmt)

    def _toggle_italic(self) -> None:
        if self._mode == "markup":
            cursor = self._markup_editor.textCursor()
            if cursor.hasSelection():
                sel = cursor.selectedText()
                cursor.insertText(sel[1:-1] if (sel.startswith("_") and sel.endswith("_")) else f"_{sel}_")
            else:
                cursor.insertText("__")
                cursor.movePosition(QTextCursor.MoveOperation.Left, n=1)
                self._markup_editor.setTextCursor(cursor)
        else:
            fmt = QTextCharFormat()
            fmt.setFontItalic(not self._rich_editor.fontItalic())
            self._rich_editor.mergeCurrentCharFormat(fmt)

    def _toggle_strikethrough(self) -> None:
        if self._mode == "markup":
            cursor = self._markup_editor.textCursor()
            if cursor.hasSelection():
                sel = cursor.selectedText()
                cursor.insertText(sel[2:-2] if (sel.startswith("~~") and sel.endswith("~~")) else f"~~{sel}~~")
            else:
                cursor.insertText("~~~~")
                cursor.movePosition(QTextCursor.MoveOperation.Left, n=2)
                self._markup_editor.setTextCursor(cursor)
        else:
            fmt = QTextCharFormat()
            fmt.setFontStrikeOut(not self._rich_editor.currentCharFormat().fontStrikeOut())
            self._rich_editor.mergeCurrentCharFormat(fmt)

    # ------------------------------------------------------------------
    # Block / structural formatting
    # ------------------------------------------------------------------

    def _toggle_quote(self) -> None:
        if self._mode == "markup":
            self._toggle_markup_prefix("> ")
        else:
            cursor = self._rich_editor.textCursor()
            block_fmt = cursor.blockFormat()
            # Toggle between 40px indent and 0
            block_fmt.setLeftMargin(0.0 if block_fmt.leftMargin() > 0 else 40.0)
            cursor.setBlockFormat(block_fmt)

    def _toggle_bullet(self) -> None:
        if self._mode == "markup":
            self._toggle_markup_prefix("- ")
        else:
            cursor = self._rich_editor.textCursor()
            if cursor.currentList():
                # Remove from list
                lst = cursor.currentList()
                lst.remove(cursor.block())
                block_fmt = QTextBlockFormat()
                cursor.setBlockFormat(block_fmt)
            else:
                list_fmt = QTextListFormat()
                list_fmt.setStyle(QTextListFormat.Style.ListDisc)
                cursor.createList(list_fmt)

    def _increase_indent(self) -> None:
        if self._mode == "markup":
            cursor = self._markup_editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
            cursor.movePosition(
                QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
            )
            line = cursor.selectedText()
            cursor.insertText("  " + line)
            self._markup_editor.setTextCursor(cursor)
        else:
            cursor = self._rich_editor.textCursor()
            if cursor.currentList():
                lst = cursor.currentList()
                new_fmt = QTextListFormat()
                new_fmt.setStyle(lst.format().style())
                new_fmt.setIndent(lst.format().indent() + 1)
                cursor.createList(new_fmt)
            else:
                block_fmt = cursor.blockFormat()
                block_fmt.setLeftMargin(block_fmt.leftMargin() + 20.0)
                cursor.setBlockFormat(block_fmt)

    def _decrease_indent(self) -> None:
        if self._mode == "markup":
            cursor = self._markup_editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
            cursor.movePosition(
                QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
            )
            line = cursor.selectedText()
            if line.startswith("  "):
                cursor.insertText(line[2:])
                self._markup_editor.setTextCursor(cursor)
        else:
            cursor = self._rich_editor.textCursor()
            if cursor.currentList():
                lst = cursor.currentList()
                indent = max(1, lst.format().indent() - 1)
                new_fmt = QTextListFormat()
                new_fmt.setStyle(lst.format().style())
                new_fmt.setIndent(indent)
                cursor.createList(new_fmt)
            else:
                block_fmt = cursor.blockFormat()
                block_fmt.setLeftMargin(max(0.0, block_fmt.leftMargin() - 20.0))
                cursor.setBlockFormat(block_fmt)

    def _toggle_markup_prefix(self, prefix: str) -> None:
        cursor = self._markup_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor
        )
        line = cursor.selectedText()
        cursor.insertText(line[len(prefix):] if line.startswith(prefix) else prefix + line)
        self._markup_editor.setTextCursor(cursor)

    def _insert_divider(self) -> None:
        if self._mode == "markup":
            cursor = self._markup_editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.EndOfLine)
            cursor.insertText("\n\n---\n\n")
            self._markup_editor.setTextCursor(cursor)
        else:
            cursor = self._rich_editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
            cursor.insertHtml("<br/><hr/><br/>")
            self._rich_editor.setTextCursor(cursor)
