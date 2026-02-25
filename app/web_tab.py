"""Persistent embedded browser widget for Veritas Editor."""

import subprocess
from pathlib import Path

from platformdirs import user_data_dir
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineSettings,
)
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
        s = _profile.settings()
        # Allow JS to read/write clipboard (needed for image paste on Substack etc.)
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        # Don't require a user gesture before media plays (allows audio on NotebookLM etc.)
        s.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        # Allow JS to open new windows (useful for OAuth pop-ups)
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, True)
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
        self._ext_btn = QPushButton("⧉")
        self._ext_btn.setFixedSize(24, 24)
        self._ext_btn.setToolTip("Open current URL in system browser")
        self._address = QLineEdit(url)
        self._address.setFixedHeight(24)
        nl.addWidget(self._back_btn)
        nl.addWidget(self._fwd_btn)
        nl.addWidget(self._reload_btn)
        nl.addWidget(self._ext_btn)
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

        self._ext_btn.clicked.connect(self._on_open_in_browser)

        # Auto-grant feature permissions (clipboard, media, notifications)
        page.featurePermissionRequested.connect(self._on_permission_requested)

        self._view.setUrl(QUrl(url))

    def current_url(self) -> str:
        """Return the current page URL as a string."""
        return self._view.url().toString()

    def _on_open_in_browser(self) -> None:
        url = self.current_url()
        if url and url not in ("about:blank", ""):
            subprocess.Popen(["xdg-open", url])

    def _on_permission_requested(self, url, feature) -> None:
        self._view.page().setFeaturePermission(
            url, feature, QWebEnginePage.PermissionPolicy.PermissionGrantedByUser
        )
