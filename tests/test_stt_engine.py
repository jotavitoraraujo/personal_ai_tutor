### --- IMPORTS --- ###
import asyncio
from unittest.mock import MagicMock, patch

import numpy as np

from system.brain.stt_engine import STTEngine

###


def test_stt_engine_lifecycle() -> None:
    with patch("system.brain.stt_engine.WhisperModel") as MockModel:
        mock_segment = MagicMock()
        mock_segment.text = " TEST"
        instance = MockModel.return_value
        instance.transcribe.return_value = ([mock_segment], None)

        stt = STTEngine()
        assert stt.model is None

        async def run_transcription() -> str | None:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            return await stt.transcribe(dummy_audio)

        result = asyncio.run(run_transcription())

        assert result == "TEST"
        assert stt.model is None

        MockModel.assert_called_once_with(
            "large-v3", device="cuda", compute_type="float16"
        )
