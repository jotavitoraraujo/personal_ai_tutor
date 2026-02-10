### --- IMPORTS --- ###
import asyncio
from unittest.mock import patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor
from system.utils.enum_state import State


def test_conductor_logic_and_handoff() -> None:
    with (
        patch("system.core.linguist_conductor.BAE") as MockBAE,
        patch("system.core.linguist_conductor.STT") as MockSTT,
    ):
        # Configuração Mock BAE
        mock_engine_instance = MockBAE.return_value
        mock_engine_instance.stop_capture.return_value = np.zeros(
            16000, dtype=np.float32
        )
        mock_stt_instance = MockSTT.return_value

        async def mock_transcribe(_: np.ndarray) -> str:
            return "Test Transcription"

        mock_stt_instance.transcribe = mock_transcribe

        conductor = LinguistConductor()

        assert conductor.state == State.IDLE
        assert conductor.audio_queue.empty()

        conductor.on_press()
        assert conductor.state == State.RECORDING
        mock_engine_instance.start_capture.assert_called_once()

        conductor.on_release()
        assert conductor.state == State.TRANSCRIBING
        mock_engine_instance.stop_capture.assert_called_once()
        assert conductor.audio_queue.qsize() == 1

        async def validate_run() -> bool:
            try:
                await asyncio.wait_for(conductor.run(), timeout=0.1)
            except TimeoutError:
                pass
            return conductor.audio_queue.empty()

        is_queue_empty = asyncio.run(validate_run())

        assert is_queue_empty
        assert conductor.state == State.IDLE
