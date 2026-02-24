"""Persistent embedded browser widget for Veritas Editor."""

from pathlib import Path

from platformdirs import user_data_dir
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_PROFILE_PATH = Path(user_data_dir("VeritasReader")) / "webengine"
_profile: QWebEngineProfile | None = None


def _get_profile() -> QWebEngineProfile:
    global _profile
    if _profile is None:
        _profile = QWebEngineProfile("veritas-editor")
        _profile.setPersistentStoragePath(str(_PROFILE_PATH))
        _profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
        )
    return _profile


class BrowserTab(QWidget):
    """Embedded browser with address bar, back/forward/reload."""

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        nav = QWidget()
        nav.setFixedHeight(26)
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(2, 0, 2, 0)
        nl.setSpacing(2)
        self._back_btn = QPushButton("‹")
        self._back_btn.setFixedSize(24, 24)
        self._fwd_btn = QPushButton("›")
        self._fwd_btn.setFixedSize(24, 24)
        self._reload_btn = QPushButton("↺")
        self._reload_btn.setFixedSize(24, 24)
        self._address = QLineEdit(url)
        self._address.setFixedHeight(24)
        nl.addWidget(self._back_btn)
        nl.addWidget(self._fwd_btn)
        nl.addWidget(self._reload_btn)
        nl.addWidget(self._address)
        layout.addWidget(nav)

        self._view = QWebEngineView()
        page = QWebEnginePage(_get_profile(), self._view)
        self._view.setPage(page)
        layout.addWidget(self._view, stretch=1)

        self._back_btn.clicked.connect(self._view.back)
        self._fwd_btn.clicked.connect(self._view.forward)
        self._reload_btn.clicked.connect(self._view.reload)
        self._address.returnPressed.connect(
            lambda: self._view.setUrl(QUrl(self._address.text()))
        )
        self._view.urlChanged.connect(lambda u: self._address.setText(u.toString()))
        self._view.setUrl(QUrl(url))
