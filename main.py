"""Scriptum Veritas — entry point.

Run with:
    python main.py
"""

import logging
import os
import sys

from PyQt6.QtWebEngineWidgets import QWebEngineView  # must precede QApplication
from PyQt6.QtWidgets import QApplication

# Configure logging before any app imports
_level = logging.DEBUG if os.environ.get("VERITAS_DEBUG") == "1" else logging.INFO
logging.basicConfig(
    level=_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Must be set before QApplication is created — Chromium reads these at process start.
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    # Allow media to play without a prior user click
    "--autoplay-policy=no-user-gesture-required"
    # Disable the out-of-process audio service and its sandbox — fixes audio on Wayland/Linux
    " --disable-features=AudioServiceOutOfProcess,AudioServiceSandbox"
    # Disable cross-origin isolation enforcement that blocks media on Google CDN URLs
    " --disable-features=CrossOriginOpenerPolicy,CrossOriginEmbedderPolicy"
    # Use the Chromium audio backend directly (avoids PipeWire handshake issues)
    " --use-fake-ui-for-media-stream"
)
# Enable remote DevTools on port 9222 so you can inspect exactly what's failing
# (open chrome://inspect or http://localhost:9222 in Chrome while the app is running)
os.environ.setdefault("QTWEBENGINE_REMOTE_DEBUGGING", "9222")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Scriptum Veritas")
    app.setOrganizationName("VeritasReader")
    app.setStyle("Fusion")

    from config.settings import AppSettings
    from app.theme import apply_dark, apply_light
    settings = AppSettings()
    if settings.dark_mode:
        apply_dark(app)
    else:
        apply_light(app)

    from app.window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("Scriptum Veritas started.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
