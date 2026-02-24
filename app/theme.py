"""Application theme helpers — light and dark Fusion palettes."""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


def apply_dark(app: QApplication) -> None:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(40,  40,  40))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,            QColor(28,  28,  28))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(48,  48,  48))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(28,  28,  28))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,          QColor(55,  55,  55))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 100, 100))
    p.setColor(QPalette.ColorRole.Link,            QColor(88,  166, 255))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(58,  120, 200))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 120))
    # Disabled variants
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(100, 100, 100))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(100, 100, 100))
    app.setPalette(p)


def apply_light(app: QApplication) -> None:
    app.setPalette(app.style().standardPalette())
