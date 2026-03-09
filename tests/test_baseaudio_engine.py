from typing import Self
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from system.audio.baseaudio_engine import BaseAudioEngine


class TestBaseAudioEngine:
    @pytest.fixture()
    def engine(self: Self) -> BaseAudioEngine:
        with patch("pyaudio.PyAudio"):
            return BaseAudioEngine()

    def test_stop_capture_happy_path_stereo_to_mono_decimation(self: Self, engine: BaseAudioEngine) -> None:
        frame_max = b"\xff\x7f\xff\x7f"

        frame_min = b"\x00\x80\x00\x80"
        frame_zero = b"\x00\x00\x00\x00"

        engine._frames = [frame_max, frame_min, frame_zero, frame_min, frame_max, frame_zero]  # type: ignore
        engine.stream = MagicMock()

        payload = engine.stop_capture()

        assert isinstance(payload, np.ndarray)
        assert payload.dtype == np.float32
        assert len(payload) == 2
        assert pytest.approx(payload[0], abs=1e-4) == 1.0  # type: ignore
        assert pytest.approx(payload[1], abs=1e-4) == -1.0  # type: ignore

    def test_unaligned_bytes_handling_odd(self: Self, engine: BaseAudioEngine) -> None:
        engine._frames = [b"\xff\x7f\xff\x7f\xff"]  # type: ignore
        engine.stream = MagicMock()

        payload = engine.stop_capture()

        assert len(payload) == 1
        assert pytest.approx(payload[0], abs=1e-4) == 1.0  # type: ignore

    def test_incomplete_stereo_frame_handling(self: Self, engine: BaseAudioEngine) -> None:
        engine._frames = [b"\xff\x7f\xff\x7f\x00\x80"]  # type: ignore
        engine.stream = MagicMock()
        payload = engine.stop_capture()

        assert len(payload) == 1
        assert pytest.approx(payload[0], abs=1e-4) == 1.0  # type: ignore

    def test_empty_buffer_handling(self: Self, engine: BaseAudioEngine) -> None:
        engine._frames = []  # type: ignore
        engine.stream = MagicMock()

        payload = engine.stop_capture()

        assert isinstance(payload, np.ndarray)
        assert len(payload) == 0

    def test_silence_processing(self: Self, engine: BaseAudioEngine) -> None:
        engine._frames = [b"\x00" * 12]  # type: ignore
        engine.stream = MagicMock()

        payload = engine.stop_capture()

        assert len(payload) == 1
        assert payload[0] == 0.0
