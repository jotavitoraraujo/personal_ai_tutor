### --- IMPORTS --- ###
import asyncio
from unittest.mock import patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor
from system.utils.enum_state import State

###


def test_conductor_logic_and_handoff() -> None:
    with (
        patch("system.core.linguist_conductor.BAE") as MockBAE,
        patch("system.core.linguist_conductor.STT") as MockSTT,
        patch("system.core.linguist_conductor.LLM") as MockLLM,
    ):
        mock_audio = MockBAE.return_value
        mock_audio.stop_capture.return_value = np.zeros(16000, dtype=np.float32)

        mock_stt = MockSTT.return_value

        async def mock_transcribe(_: np.ndarray) -> str:
            return "Test Transcription"

        mock_stt.transcribe = mock_transcribe
        mock_llm = MockLLM.return_value

        async def mock_think(_: str) -> dict[str, str | int]:
            return {
                "text": "Test Response",
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_time_ms": 0,
            }

        async def mock_initialize() -> bool:
            return True

        mock_llm.think = mock_think
        mock_llm.initialize_layered_mentor = mock_initialize

        async def run_async_test() -> bool:
            conductor = LinguistConductor()

            assert conductor.state == State.IDLE

            conductor.on_press()
            conductor.on_release()
            await asyncio.sleep(0)

            assert conductor.audio_queue.qsize() == 1
            assert conductor.state == State.TRANSCRIBING
            try:
                await asyncio.wait_for(conductor.run(), timeout=0.1)
            except TimeoutError:
                pass

            return conductor.audio_queue.empty() and conductor.state == State.IDLE

        assert asyncio.run(run_async_test())
