### --- IMPORTS --- ###
import asyncio
from unittest.mock import patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor
from system.utils.enum_state import State

###


def test_full_flow_integration() -> None:
    with (
        patch("system.core.linguist_conductor.BAE") as MockBAE,
        patch("system.core.linguist_conductor.STT") as MockSTT,
    ):
        mock_audio = MockBAE.return_value
        mock_audio.stop_capture.return_value = np.zeros(16000, dtype=np.float32)

        mock_stt = MockSTT.return_value

        async def mock_transcribe_call(_: np.ndarray) -> str:
            return "Test Transcription"

        mock_stt.transcribe.side_effect = mock_transcribe_call

        async def run_integration_logic() -> None:
            conductor = LinguistConductor()

            conductor.on_press()
            conductor.on_release()
            await asyncio.sleep(0)

            try:
                await asyncio.wait_for(conductor.run(), timeout=0.2)
            except TimeoutError:
                pass
            assert conductor.state == State.IDLE
            assert conductor.audio_queue.empty()

        asyncio.run(run_integration_logic())
