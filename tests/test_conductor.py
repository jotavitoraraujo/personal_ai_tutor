import asyncio
from typing import Any, Self
from unittest.mock import AsyncMock, patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor
from system.utils.enum_state import State


class MockResponse:
    def __init__(self: Self) -> None:
        self.structured_result: dict[str, Any] = {
            "structured_data": {"conversation_response": "Test response", "accuracy_score": 100, "corrections": [], "pedagogical_tip": "None", "proficiency_assessment": "A1"},
            "prompt_tokens": 10,
            "output_tokens": 10,
            "total_time_ms": 100,
        }


def test_conductor_success_flow() -> None:
    async def run_test() -> None:
        with (
            patch("system.core.linguist_conductor.BAE") as _mock_bae,
            patch("system.core.linguist_conductor.STT") as mock_stt,
            patch("system.core.linguist_conductor.LLM") as mock_llm,
            patch("system.ui.display_manager.DisplayManager.show_mentor_response") as mock_dm,
            patch("system.ui.display_manager.DisplayManager.print_gpu_status"),
            patch("system.ui.display_manager.DisplayManager.show_user_transcription"),
        ):
            conductor: LinguistConductor = LinguistConductor()
            mock_llm.return_value.initialize_layered_mentor = AsyncMock(return_value=True)
            mock_stt.return_value.transcribe = AsyncMock(return_value="Hello world")
            mock_llm.return_value.think = AsyncMock(return_value=MockResponse().structured_result)

            task: asyncio.Task[None] = asyncio.create_task(conductor.run())
            await asyncio.sleep(0.05)

            conductor.on_press()
            conductor.on_release()

            await asyncio.sleep(0.1)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            assert conductor.state == State.IDLE
            mock_dm.assert_called_once()

    asyncio.run(run_test())


def test_conductor_transcription_silence() -> None:
    async def run_test() -> None:
        with (
            patch("system.core.linguist_conductor.BAE"),
            patch("system.core.linguist_conductor.STT") as mock_stt,
            patch("system.core.linguist_conductor.LLM") as mock_llm,
            patch("system.ui.display_manager.DisplayManager.print_gpu_status"),
        ):
            conductor: LinguistConductor = LinguistConductor()
            mock_llm.return_value.initialize_layered_mentor = AsyncMock(return_value=True)
            mock_stt.return_value.transcribe = AsyncMock(return_value=None)

            task: asyncio.Task[None] = asyncio.create_task(conductor.run())
            await asyncio.sleep(0.05)

            conductor.audio_queue.put_nowait(np.zeros(512, dtype=np.float32))
            await asyncio.sleep(0.1)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            assert conductor.state == State.IDLE
            mock_llm.return_value.think.assert_not_called()

    asyncio.run(run_test())
