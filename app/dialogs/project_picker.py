"""Project picker dialog — lists all vault projects that have a session.json."""

import json
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class ProjectPickerDialog(QDialog):
    """Lists sub-directories inside vault_root that contain a session.json,
    showing project name and last-saved timestamp.  Double-click or Open
    to select."""

    def __init__(self, vault_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._vault_root = vault_root
        self.setWindowTitle("Open Project")
        self.resize(520, 360)
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a project to restore:"))

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open")
        self._open_btn.setDefault(True)
        self._open_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _populate(self) -> None:
        self._list.clear()
        if not self._vault_root.exists():
            item = QListWidgetItem("(vault root does not exist)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
            self._open_btn.setEnabled(False)
            return

        projects = []
        for sub in sorted(self._vault_root.iterdir()):
            if not sub.is_dir():
                continue
            session_file = sub / "session.json"
            if not session_file.exists():
                continue
            saved_at = ""
            name = sub.name
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                name = data.get("project", {}).get("name", sub.name) or sub.name
                raw_ts = data.get("saved_at", "")
                if raw_ts:
                    dt = datetime.fromisoformat(raw_ts)
                    saved_at = dt.strftime("%Y-%m-%d  %H:%M")
            except Exception:
                pass
            projects.append((name, saved_at, sub))

        if not projects:
            item = QListWidgetItem("(no projects with saved sessions found)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
            self._open_btn.setEnabled(False)
            return

        for name, saved_at, path in projects:
            label = name
            if saved_at:
                label += f"    —    {saved_at}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._list.addItem(item)

        self._list.setCurrentRow(0)
        self._open_btn.setEnabled(True)

    def get_selected_vault_root(self) -> Path | None:
        """Return vault.root path (the project sub-directory), or None."""
        items = self._list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)
