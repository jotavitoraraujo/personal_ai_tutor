### --- IMPORTS --- ###
import gc
from collections.abc import Iterable
from typing import Self

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from faster_whisper.transcribe import TranscriptionInfo as TInfo
from torch import cuda

###


class STTEngine:
    __slots__ = ("model", "device", "compute_type")

    def __init__(self: Self) -> None:
        self.model = None
        self.device = "cuda"
        self.compute_type = "float16"

    async def transcribe(self: Self, arr_float32: np.ndarray) -> str | None:
        if self.model is None:
            self.model = WhisperModel(
                "large-v3", device=self.device, compute_type=self.compute_type
            )
            segments: Iterable[Segment]
            _info: TInfo
            segments, _info = self.model.transcribe(arr_float32, beam_size=5)  # type: ignore
            final_string = " ".join(segment.text for segment in segments).strip()
            del self.model
            self.model = None
            gc.collect()
            cuda.empty_cache()
            return final_string
