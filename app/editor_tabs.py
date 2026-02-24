"""Multi-tab editor container — wraps multiple EditorWidget instances."""

import logging

from PyQt6.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMenu,
    QMessageBox,
    QPushButton,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.editor import EditorWidget

logger = logging.getLogger(__name__)


class _TabBar(QTabBar):
    """QTabBar subclass — emits layout_changed and context_menu_requested."""

    layout_changed = pyqtSignal()
    context_menu_requested = pyqtSignal(int, QPoint)   # (tab_index, global_pos)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setExpanding(False)

    def tabLayoutChange(self) -> None:
        super().tabLayoutChange()
        self.layout_changed.emit()

    def contextMenuEvent(self, event) -> None:
        idx = self.tabAt(event.pos())
        if idx >= 0:
            self.context_menu_requested.emit(idx, self.mapToGlobal(event.pos()))


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
    clone_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(False)   # close via right-click menu only
        self._tabs.tabCloseRequested.connect(self._on_close_tab)

        # Custom tab bar (must be installed before any addTab calls)
        self._tab_bar = _TabBar()
        self._tabs.setTabBar(self._tab_bar)

        layout.addWidget(self._tabs)

        # -------------------------------------------------------------------
        # Action buttons as children of EditorTabWidget (NOT QTabBar).
        # QTabBar clips its own children; by parenting to the outer widget we
        # avoid that entirely and use coordinate-mapped positioning instead.
        # -------------------------------------------------------------------
        specs = [
            ("+",  "New scratch tab",
             lambda: self.open_in_new_tab("", "Draft")),
            ("⊞",  "Clone tab — copies content to new tab and saves a vault version",
             self.clone_requested),
            ("S",  "Open Substack dashboard",
             lambda: self.open_url_tab(
                 "https://serenityfinch.substack.com/publish/home", "Substack")),
        ]
        self._overlay_btns: list[QPushButton] = []
        for label, tip, handler in specs:
            btn = QPushButton(label, self)
            btn.setToolTip(tip)
            btn.setCursor(Qt.CursorShape.ArrowCursor)
            btn.clicked.connect(handler)
            self._overlay_btns.append(btn)

        # Reposition whenever tab layout changes or current tab switches
        self._tab_bar.layout_changed.connect(self._reposition_btns)
        self._tab_bar.context_menu_requested.connect(self._on_tab_context_menu)
        self._tabs.currentChanged.connect(lambda _: self._reposition_btns())

        # Primary document tab (index 0) — always present, not closable
        self._add_editor_tab("", "Document")

        # Defer initial position until Qt has finished laying everything out
        QTimer.singleShot(0, self._reposition_btns)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------

    def _on_tab_context_menu(self, idx: int, global_pos: QPoint) -> None:
        menu = QMenu(self)
        clone_action = menu.addAction("Clone Tab")
        menu.addSeparator()
        close_action = menu.addAction("Close Tab")
        close_action.setEnabled(idx != 0)   # primary tab is never closable

        action = menu.exec(global_pos)

        if action == clone_action:
            self._clone_tab_by_index(idx)
        elif action == close_action:
            self._close_tab_with_warning(idx)

    def _close_tab_with_warning(self, idx: int) -> None:
        if idx == 0:
            return
        widget = self._tabs.widget(idx)
        has_content = isinstance(widget, EditorWidget) and bool(widget.get_text().strip())
        if has_content:
            reply = QMessageBox.question(
                self,
                "Close Tab",
                f"'{self._tabs.tabText(idx)}' has unsaved content. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._tabs.removeTab(idx)

    def _clone_tab_by_index(self, idx: int) -> None:
        widget = self._tabs.widget(idx)
        text = widget.get_text() if isinstance(widget, EditorWidget) else ""
        title = self._tabs.tabText(idx) + " (copy)"
        self._add_editor_tab(text, title)

    # ------------------------------------------------------------------
    # Overlay button positioning
    # ------------------------------------------------------------------

    def _reposition_btns(self) -> None:
        tb = self._tab_bar
        h = tb.height()
        if h == 0:
            return

        bh = max(h - 4, 18)
        bw = bh

        count = tb.count()
        x_in_tb = (tb.tabRect(count - 1).right() + 4) if count > 0 else 4

        # Map from tab-bar-local coordinates into EditorTabWidget coordinates
        origin: QPoint = tb.mapTo(self, QPoint(x_in_tb, (h - bh) // 2))
        x, y = origin.x(), origin.y()

        for btn in self._overlay_btns:
            btn.setFixedSize(bw, bh)
            btn.move(x, y)
            btn.raise_()
            x += bw + 2

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_btns()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reposition_btns()

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

    def clone_current_tab(self) -> str:
        """Copy active tab text into a new tab. Returns the cloned text."""
        text = self.get_text()
        title = self.current_tab_title + " (copy)"
        self._add_editor_tab(text, title)
        return text

    def open_url_tab(self, url: str, title: str) -> None:
        """Open an embedded browser tab at *url*."""
        from app.web_tab import BrowserTab
        browser = BrowserTab(url)
        idx = self._tabs.addTab(browser, title)
        self._tabs.setCurrentIndex(idx)
