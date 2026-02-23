"""Tests for core.tts_engine."""

from unittest.mock import MagicMock, patch

import pytest

from core.tts_engine import CoquiEngine, KokoroEngine, VoiceProfile, get_engine


class TestGetEngine:
    def test_returns_kokoro_by_default(self):
        engine = get_engine("kokoro")
        assert isinstance(engine, KokoroEngine)

    def test_returns_coqui(self):
        engine = get_engine("coqui")
        assert isinstance(engine, CoquiEngine)

    def test_unknown_falls_back_to_kokoro(self):
        engine = get_engine("banana")
        assert isinstance(engine, KokoroEngine)


class TestKokoroEngine:
    def test_list_voices_returns_profiles(self):
        engine = KokoroEngine()
        voices = engine.list_voices()
        assert len(voices) > 0
        assert all(isinstance(v, VoiceProfile) for v in voices)
        assert all(v.engine == "kokoro" for v in voices)

    def test_synthesize_calls_pipeline(self, tmp_path):
        engine = KokoroEngine()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([("gs", "ps", None)])
        engine._pipeline = mock_pipeline

        import numpy as np
        mock_pipeline.return_value = iter([("gs", "ps", np.zeros(24000, dtype="float32"))])

        with patch("soundfile.write") as mock_sf:
            out = tmp_path / "test_raw.wav"
            engine.synthesize("Hello world", out)
            mock_sf.assert_called_once()


class TestCoquiEngine:
    def test_list_voices_returns_profiles(self):
        engine = CoquiEngine()
        voices = engine.list_voices()
        assert len(voices) > 0
        assert all(v.engine == "coqui" for v in voices)
