"""Multi-tab editor container — wraps multiple EditorWidget instances."""

import logging
import re

from PyQt6.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Matches the version suffix appended to tab titles (e.g. " — V3")
_V_RE = re.compile(r" — V(\d+)$")

from app.editor import EditorWidget

logger = logging.getLogger(__name__)


class _TabBar(QTabBar):
    """QTabBar subclass — emits layout_changed, context_menu_requested, rename_requested."""

    layout_changed = pyqtSignal()
    context_menu_requested = pyqtSignal(int, QPoint)   # (tab_index, global_pos)
    rename_requested = pyqtSignal(int)                  # (tab_index)

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

    def mouseDoubleClickEvent(self, event) -> None:
        idx = self.tabAt(event.pos())
        if idx >= 0:
            self.rename_requested.emit(idx)
        else:
            super().mouseDoubleClickEvent(event)


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
    tabs_changed = pyqtSignal()   # emitted when tabs are added / removed / renamed

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._last_editor_idx: int = 0
        self._base_title: str = ""          # project name without version suffix
        self._widget_paths: dict = {}       # EditorWidget → Path (vault version file)
        self._font_size: int = 11           # current font size, applied to new tabs
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
             lambda: self.open_in_new_tab("")),
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
        self._tab_bar.rename_requested.connect(self._on_rename_tab)
        self._tabs.currentChanged.connect(self._on_current_tab_changed)

        # Primary document tab (index 0) — always present, not closable
        self._add_editor_tab("", "V1")

        # Ctrl+Shift+C — copy active tab contents to clipboard
        copy_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_shortcut.activated.connect(
            lambda: self._copy_tab_by_index(self._tabs.currentIndex())
        )

        # Defer initial position until Qt has finished laying everything out
        QTimer.singleShot(0, self._reposition_btns)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------

    def _on_tab_context_menu(self, idx: int, global_pos: QPoint) -> None:
        menu = QMenu(self)
        copy_action = menu.addAction("Copy Tab Contents")
        copy_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        clone_action = menu.addAction("Clone Tab")
        rename_action = menu.addAction("Rename Tab…")
        menu.addSeparator()
        close_action = menu.addAction("Close Tab")
        close_action.setEnabled(idx != 0)   # primary tab is never closable

        action = menu.exec(global_pos)

        if action == copy_action:
            self._copy_tab_by_index(idx)
        elif action == clone_action:
            self._clone_tab_by_index(idx)
        elif action == rename_action:
            self._on_rename_tab(idx)
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

    def close_all_non_primary_tabs(self) -> None:
        """Close all tabs except index 0, prompting for unsaved content."""
        for idx in range(self._tabs.count() - 1, 0, -1):
            self._close_tab_with_warning(idx)
        self.tabs_changed.emit()

    def _copy_tab_by_index(self, idx: int) -> None:
        widget = self._tabs.widget(idx)
        if isinstance(widget, EditorWidget):
            text = widget.get_text()
        elif hasattr(widget, "_view"):          # BrowserTab — copy current URL
            text = widget._view.url().toString()
        else:
            return
        if text:
            QApplication.clipboard().setText(text)

    def _clone_tab_by_index(self, idx: int) -> None:
        widget = self._tabs.widget(idx)
        text = widget.get_text() if isinstance(widget, EditorWidget) else ""
        self._add_editor_tab(text, self._versioned_title())

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
        editor.set_font_size(self._font_size)
        if text:
            editor.set_text(text)
        editor.text_changed.connect(self.text_changed)
        editor.grammar_check_requested.connect(self.grammar_check_requested)
        idx = self._tabs.addTab(editor, title)
        self._tabs.setCurrentIndex(idx)
        self.tabs_changed.emit()
        return idx

    def set_font_size(self, size: int) -> None:
        """Apply *size* to all existing editor tabs and remember it for new ones."""
        self._font_size = size
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if isinstance(w, EditorWidget):
                w.set_font_size(size)

    def _on_close_tab(self, idx: int) -> None:
        if idx == 0:
            return  # never close the primary document tab
        self._tabs.removeTab(idx)
        self.tabs_changed.emit()

    def _on_current_tab_changed(self, idx: int) -> None:
        """Track the last-active EditorWidget tab and reposition overlay buttons."""
        w = self._tabs.widget(idx)
        if isinstance(w, EditorWidget):
            self._last_editor_idx = idx
        self._reposition_btns()

    def _active(self) -> EditorWidget:
        w = self._tabs.currentWidget()
        if isinstance(w, EditorWidget):
            return w
        # Current tab is a browser tab — fall back to the last known editor tab
        fb = self._tabs.widget(self._last_editor_idx)
        return fb if isinstance(fb, EditorWidget) else self._tabs.widget(0)

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
        """Set the primary (index 0) tab title and update the stored base name.

        Strips any existing ' — V{n}' suffix from *title* before storing
        so calling this with a previously-displayed title is always safe.
        """
        base = _V_RE.sub("", title).rstrip()
        self._base_title = base
        self._tabs.setTabText(0, f"{base} — V1" if base else "V1")
        self.tabs_changed.emit()

    def open_in_new_tab(self, text: str = "", title: str | None = None) -> None:
        """Open text in a new scratch tab and switch to it.

        If *title* is None the tab receives an auto-generated versioned name
        (e.g. 'Project — V3').  Pass an explicit string to override.
        """
        if title is None:
            title = self._versioned_title()
        self._add_editor_tab(text, title)

    def get_main_text(self) -> str:
        """Return content of the primary document tab (index 0)."""
        editor = self._tabs.widget(0)
        return editor.get_text() if isinstance(editor, EditorWidget) else ""

    def clone_current_tab(self) -> tuple:
        """Copy active tab text into a new versioned tab.

        Returns ``(text, new_editor_widget)`` so the caller can register
        a vault file path against the new widget via ``set_tab_path()``.
        """
        text = self.get_text()
        title = self._versioned_title()
        editor = EditorWidget()
        if text:
            editor.set_text(text)
        editor.text_changed.connect(self.text_changed)
        editor.grammar_check_requested.connect(self.grammar_check_requested)
        idx = self._tabs.addTab(editor, title)
        self._tabs.setCurrentIndex(idx)
        self.tabs_changed.emit()
        return text, editor

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def _next_v_num(self) -> int:
        """Scan all tab titles for ' — V{n}' and return max(n) + 1."""
        max_v = 0
        for i in range(self._tabs.count()):
            m = _V_RE.search(self._tabs.tabText(i))
            if m:
                max_v = max(max_v, int(m.group(1)))
        return max_v + 1

    def _versioned_title(self) -> str:
        v = self._next_v_num()
        return f"{self._base_title} — V{v}" if self._base_title else f"V{v}"

    # ------------------------------------------------------------------
    # Per-tab vault file path registration
    # ------------------------------------------------------------------

    def set_tab_path(self, widget, path) -> None:
        """Register a vault version file path for a given tab widget."""
        self._widget_paths[widget] = path

    # ------------------------------------------------------------------
    # Rename (double-click or context menu)
    # ------------------------------------------------------------------

    def _on_rename_tab(self, idx: int) -> None:
        current = self._tabs.tabText(idx)
        new_title, ok = QInputDialog.getText(
            self, "Rename Tab", "New name:", QLineEdit.EchoMode.Normal, current
        )
        if not ok or not new_title.strip() or new_title.strip() == current:
            return
        new_title = new_title.strip()
        self._tabs.setTabText(idx, new_title)
        # Keep base title in sync if renaming the primary tab
        if idx == 0:
            self._base_title = _V_RE.sub("", new_title).rstrip()
        # Rename backing vault version file if one was registered
        widget = self._tabs.widget(idx)
        if widget in self._widget_paths:
            old_path = self._widget_paths[widget]
            new_path = old_path.parent / (new_title + old_path.suffix)
            try:
                old_path.rename(new_path)
                self._widget_paths[widget] = new_path
                logger.info("Renamed version file: %s → %s", old_path.name, new_path.name)
            except Exception as exc:
                logger.warning("Could not rename version file %s: %s", old_path, exc)
        self.tabs_changed.emit()

    # ------------------------------------------------------------------
    # Low-level accessors (used by window.py session restore)
    # ------------------------------------------------------------------

    def tab_count(self) -> int:
        return self._tabs.count()

    def tab_widget_at(self, index: int) -> QWidget:
        return self._tabs.widget(index)

    def active_tab_index(self) -> int:
        return self._tabs.currentIndex()

    def set_active_tab(self, index: int) -> None:
        if 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)

    def open_url_tab(self, url: str, title: str) -> None:
        """Open an embedded browser tab at *url*."""
        from app.web_tab import BrowserTab
        browser = BrowserTab(url)
        idx = self._tabs.addTab(browser, title)
        self._tabs.setCurrentIndex(idx)
        self.tabs_changed.emit()

    def get_tab_text(self, idx: int) -> str:
        """Return the text content of the editor tab at *idx* (empty for browser tabs)."""
        w = self._tabs.widget(idx)
        return w.get_text() if isinstance(w, EditorWidget) else ""
