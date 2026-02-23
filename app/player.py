"""Built-in audio player widget using QMediaPlayer."""

import logging
from pathlib import Path

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


def _ms_to_str(ms: int) -> str:
    total_sec = max(0, ms) // 1000
    m, s = divmod(total_sec, 60)
    return f"{m:02d}:{s:02d}"


class PlayerWidget(QWidget):
    """Compact audio player bar (play/pause/stop, seek slider, time label)."""

    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_path: Path | None = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.setEnabled(False)
        layout.addWidget(self._play_btn)

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setToolTip("Stop")
        self._stop_btn.setEnabled(False)
        layout.addWidget(self._stop_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        layout.addWidget(self._slider)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setFixedWidth(90)
        layout.addWidget(self._time_label)

        # Qt multimedia
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(1.0)

    def _connect_signals(self) -> None:
        self._play_btn.clicked.connect(self._on_play_pause)
        self._stop_btn.clicked.connect(self._on_stop)
        self._slider.sliderMoved.connect(self._on_seek)

        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.errorOccurred.connect(self._on_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Path) -> None:
        """Load an audio file for playback.

        Args:
            path: Path to .wav or .mp3 file.
        """
        self._current_path = Path(path)
        self._player.setSource(QUrl.fromLocalFile(str(self._current_path)))
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._slider.setEnabled(True)
        self._play_btn.setText("▶")
        logger.info("Player loaded: %s", path)

    def unload(self) -> None:
        """Stop playback and clear the loaded file."""
        self._player.stop()
        self._player.setSource(QUrl())
        self._play_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._slider.setEnabled(False)
        self._slider.setValue(0)
        self._time_label.setText("00:00 / 00:00")
        self._current_path = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_play_pause(self) -> None:
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()
            self.playback_started.emit()

    def _on_stop(self) -> None:
        self._player.stop()
        self.playback_stopped.emit()

    def _on_seek(self, position_ms: int) -> None:
        self._player.setPosition(position_ms)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._slider.setRange(0, duration_ms)
        self._update_time_label(0, duration_ms)

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._slider.isSliderDown():
            self._slider.setValue(position_ms)
        self._update_time_label(position_ms, self._player.duration())

    def _update_time_label(self, pos_ms: int, dur_ms: int) -> None:
        self._time_label.setText(f"{_ms_to_str(pos_ms)} / {_ms_to_str(dur_ms)}")

    def _on_error(self, error, error_string: str) -> None:
        logger.error("Media player error: %s", error_string)
