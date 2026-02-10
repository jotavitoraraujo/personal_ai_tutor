### --- IMPORTS --- ###
import asyncio
from unittest.mock import MagicMock, patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor
from system.utils.enum_state import State


def test_full_flow_integration() -> None:
    with (
        patch("system.core.linguist_conductor.BAE") as MockBAE,
        patch("system.core.linguist_conductor.STT") as MockSTT,
    ):
        mock_audio = MockBAE.return_value
        mock_audio.stop_capture.return_value = np.zeros(16000, dtype=np.float32)

        mock_stt = MockSTT.return_value
        mock_stt.transcribe = MagicMock()

        async def mock_transcribe_call(_: np.ndarray) -> str:
            return "Test Transcription"

        mock_stt.transcribe.side_effect = mock_transcribe_call

        conductor = LinguistConductor()

        conductor.on_press()
        conductor.on_release()

        async def run_integration_logic() -> None:
            try:
                await asyncio.wait_for(conductor.run(), timeout=0.2)
            except TimeoutError:
                pass

        asyncio.run(run_integration_logic())
        assert conductor.state == State.IDLE
        mock_stt.transcribe.assert_called_once()
        assert conductor.audio_queue.empty()
