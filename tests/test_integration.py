### --- IMPORTS --- ###
import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor
from system.utils.enum_state import State

###


def test_full_flow_integration() -> None:
    with (
        patch("system.core.linguist_conductor.BAE") as MockBAE,
        patch("system.core.linguist_conductor.STT") as MockSTT,
        patch("system.core.linguist_conductor.LLM") as MockLLM,
    ):
        mock_audio = MockBAE.return_value
        mock_audio.stop_capture.return_value = np.zeros(16000, dtype=np.float32)
        mock_stt = MockSTT.return_value
        mock_stt.transcribe = AsyncMock(return_value="Test Transcription")
        mock_llm = MockLLM.return_value
        mock_llm.think = AsyncMock(
            return_value={
                "text": "Test Response",
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_time_ms": 0,
            }
        )
        mock_llm.initialize_layered_mentor = AsyncMock(return_value=True)

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
            mock_llm.think.assert_called_once_with("Test Transcription")
            mock_llm.initialize_layered_mentor.assert_called_once()

        asyncio.run(run_integration_logic())
