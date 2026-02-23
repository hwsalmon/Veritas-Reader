"""Veritas Reader — entry point.

Run with:
    python main.py
"""

import logging
import os
import sys

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
    app.setApplicationName("Veritas Reader")
    app.setOrganizationName("VeritasReader")

    # Apply a clean style
    app.setStyle("Fusion")

    from app.window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("Veritas Reader started.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
