"""Google Docs import/export dialog."""

import re

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


_DOC_ID_RE = re.compile(r"/document/d/([a-zA-Z0-9_-]+)")


def _extract_doc_id(text: str) -> str | None:
    """Extract a document ID from a full URL or return the raw ID."""
    m = _DOC_ID_RE.search(text)
    if m:
        return m.group(1)
    stripped = text.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]+", stripped):
        return stripped
    return None


class GDocsDialog(QDialog):
    """Dialog for Google Docs import (read) or export (create/update)."""

    def __init__(self, mode: str = "import", parent=None) -> None:
        """
        Args:
            mode: "import" to read a doc, "export" to create/update a doc.
        """
        super().__init__(parent)
        self._mode = mode
        self.setWindowTitle("Google Docs — Import" if mode == "import" else "Google Docs — Export")
        self.resize(480, 200)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText(
            "Paste the Google Docs URL or document ID"
        )
        form.addRow("Document URL / ID:", self._url_input)

        if self._mode == "export":
            self._title_input = QLineEdit()
            self._title_input.setPlaceholderText("Leave blank to update existing doc")
            form.addRow("New document title:", self._title_input)

        layout.addLayout(form)

        auth_btn = QPushButton("Authenticate with Google")
        auth_btn.clicked.connect(self._authenticate)
        layout.addWidget(auth_btn)

        layout.addWidget(
            QLabel(
                "Note: The first time you connect, a browser window will open "
                "to authorize access."
            )
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _authenticate(self) -> None:
        try:
            from core.gdocs import authenticate
            authenticate()
            QMessageBox.information(self, "Google Docs", "Authentication successful.")
        except Exception as exc:
            QMessageBox.critical(self, "Authentication Failed", str(exc))

    def _on_accept(self) -> None:
        doc_id = _extract_doc_id(self._url_input.text())
        if not doc_id and self._mode == "import":
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid Google Docs URL or document ID.")
            return
        self.accept()

    def get_doc_id(self) -> str | None:
        return _extract_doc_id(self._url_input.text())

    def get_new_title(self) -> str:
        if self._mode == "export" and hasattr(self, "_title_input"):
            return self._title_input.text().strip()
        return ""
