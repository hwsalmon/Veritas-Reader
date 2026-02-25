"""Imports browser dialog — browse staged import documents."""

import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

_SUPPORTED = {".md", ".txt", ".docx"}


class ImportsBrowserDialog(QDialog):
    """Browse documents staged in the imports folder."""

    def __init__(self, imports_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._imports_dir = imports_dir
        self.setWindowTitle("Open from Imports")
        self.resize(480, 360)
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open")
        self._open_btn.setDefault(True)
        self._open_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._open_btn)

        reveal_btn = QPushButton("Reveal Folder…")
        reveal_btn.clicked.connect(self._reveal_folder)
        btn_row.addWidget(reveal_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _populate(self) -> None:
        self._list.clear()
        if not self._imports_dir.exists():
            item = QListWidgetItem("(imports folder does not exist yet)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
            self._open_btn.setEnabled(False)
            return

        files = sorted(
            p for p in self._imports_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _SUPPORTED
        )
        if not files:
            item = QListWidgetItem("(no supported files in imports folder)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
            self._open_btn.setEnabled(False)
            return

        for p in files:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self._list.addItem(item)
        self._list.setCurrentRow(0)
        self._open_btn.setEnabled(True)

    def _reveal_folder(self) -> None:
        self._imports_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(self._imports_dir)])

    def get_selected_path(self) -> Path | None:
        items = self._list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)
