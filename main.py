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
