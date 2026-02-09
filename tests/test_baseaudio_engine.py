from typing import Self
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from system.audio.baseaudio_engine import BaseAudioEngine


class TestBaseAudioEngine:
    @patch("pyaudio.PyAudio")
    def test_stop_capture_data_integrity(self: Self, mock_pyaudio: MagicMock) -> None:
        engine = BaseAudioEngine()
        mock_data = b"\xff\x7f\x00\x80"
        engine._frames = [mock_data]  # type: ignore
        engine.stream = MagicMock()
        payload = engine.stop_capture()

        assert isinstance(payload, np.ndarray)
        assert payload.dtype == np.float32
        assert len(payload) == 2
        assert pytest.approx(payload[0], abs=1e-4) == 1.0  # type: ignore
        assert payload[1] == -1.0

    @patch("pyaudio.PyAudio")
    def test_unaligned_buffer_handling(self: Self, mock_pyaudio: MagicMock) -> None:
        engine = BaseAudioEngine()
        engine._frames = [b"\xff\x7f\x00"]  # type: ignore
        engine.stream = MagicMock()
        payload = engine.stop_capture()

        assert len(payload) == 1
        assert pytest.approx(payload[0], abs=1e-4) == 1.0  # type: ignore
