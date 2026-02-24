"""Multi-tab editor container — wraps multiple EditorWidget instances."""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QPushButton,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.editor import EditorWidget

logger = logging.getLogger(__name__)


class EditorTabWidget(QWidget):
    """QTabWidget containing EditorWidget tabs.

    Exposes the same public API as EditorWidget (get_text, set_text, etc.)
    delegated to whichever tab is currently active, so window.py needs
    minimal changes.

    Extra API
    ---------
    open_in_new_tab(text, title)  Open text in a new scratch tab.
    set_main_tab_title(title)     Rename the primary (document) tab.
    get_main_text()               Content of the primary tab only.
    """

    text_changed = pyqtSignal()
    grammar_check_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._on_close_tab)

        # "+" corner button to open a blank scratch tab
        add_btn = QPushButton("+")
        add_btn.setFixedSize(22, 22)
        add_btn.setToolTip("New scratch tab")
        add_btn.clicked.connect(lambda: self.open_in_new_tab("", "Draft"))
        self._tabs.setCornerWidget(add_btn)

        # Primary document tab (index 0) — always present, not closable
        self._add_editor_tab("", "Document")
        self._set_tab_closable(0, False)

        layout.addWidget(self._tabs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_editor_tab(self, text: str, title: str) -> int:
        editor = EditorWidget()
        if text:
            editor.set_text(text)
        editor.text_changed.connect(self.text_changed)
        editor.grammar_check_requested.connect(self.grammar_check_requested)
        idx = self._tabs.addTab(editor, title)
        self._tabs.setCurrentIndex(idx)
        return idx

    def _set_tab_closable(self, idx: int, closable: bool) -> None:
        btn = self._tabs.tabBar().tabButton(idx, QTabBar.ButtonPosition.RightSide)
        if btn:
            btn.setVisible(closable)

    def _on_close_tab(self, idx: int) -> None:
        if idx == 0:
            return  # never close the primary document tab
        self._tabs.removeTab(idx)

    def _active(self) -> EditorWidget:
        w = self._tabs.currentWidget()
        # Fallback to first tab if somehow currentWidget is None
        return w if isinstance(w, EditorWidget) else self._tabs.widget(0)

    # ------------------------------------------------------------------
    # Public API — mirrors EditorWidget
    # ------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        """Set content of the active tab."""
        self._active().set_text(text)

    def get_text(self) -> str:
        """Get content of the active tab as markdown."""
        return self._active().get_text()

    def append_text(self, text: str) -> None:
        self._active().append_text(text)

    def clear(self) -> None:
        self._active().clear()

    def is_empty(self) -> bool:
        return self._active().is_empty()

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    @property
    def current_tab_title(self) -> str:
        return self._tabs.tabText(self._tabs.currentIndex())

    def set_main_tab_title(self, title: str) -> None:
        """Rename the primary (index 0) tab."""
        self._tabs.setTabText(0, title)

    def open_in_new_tab(self, text: str = "", title: str = "Draft") -> None:
        """Open text in a new scratch tab and switch to it."""
        self._add_editor_tab(text, title)

    def get_main_text(self) -> str:
        """Return content of the primary document tab (index 0)."""
        editor = self._tabs.widget(0)
        return editor.get_text() if isinstance(editor, EditorWidget) else ""
