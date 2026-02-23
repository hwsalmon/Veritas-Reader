"""Paste Plain Text dialog."""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)


class PasteDialog(QDialog):
    """Modal dialog for pasting plain text into the editor."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paste Plain Text")
        self.resize(600, 400)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Paste your text below:"))

        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText("Paste or type text here...")
        layout.addWidget(self._text_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_text(self) -> str:
        """Return the text entered by the user."""
        return self._text_edit.toPlainText()
