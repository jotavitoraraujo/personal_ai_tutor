import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np

from system.core.linguist_conductor import LinguistConductor


def test_system_integration_contract() -> None:
    with (
        patch("system.core.linguist_conductor.BAE") as _mock_bae,
        patch("system.core.linguist_conductor.STT") as mock_stt,
        patch("system.core.linguist_conductor.LLM") as mock_llm,
        patch("system.core.linguist_conductor.DM") as mock_dm,
    ):
        mock_stt.return_value.transcribe = AsyncMock(return_value="Valid transcription")
        mock_llm.return_value.initialize_layered_mentor = AsyncMock(return_value=True)

        structured_payload: dict[str, Any] = {
            "structured_data": {"speech": "Integration Success", "template": "Template Sucess"},
            "prompt_tokens": 50,
            "output_tokens": 50,
            "total_time_ms": 500,
        }
        mock_llm.return_value.think = AsyncMock(return_value=structured_payload)

        async def run_integration() -> None:
            conductor: LinguistConductor = LinguistConductor()
            conductor.audio_queue.put_nowait(np.zeros(1024, dtype=np.float32))

            task: asyncio.Task[None] = asyncio.create_task(conductor.run())
            await asyncio.sleep(0.1)
            task.cancel()

            mock_dm.show_mentor_response.assert_called_with(structured_payload["structured_data"], structured_payload)

        asyncio.run(run_integration())
