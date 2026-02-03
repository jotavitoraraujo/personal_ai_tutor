### --- IMPORTS --- ###
import math
import struct
from collections.abc import AsyncGenerator, Sequence
from typing import Final, Self

###


class BaseAudioEngine:
    """
    Core engine for handling raw audio signals and voice activity detection.
    Optimized for 16-bit PCM at 16kHz as required by high-fidelity STT models.
    """

    SAMPLE_RATE: Final[int] = 16000
    CHUNK_SIZE: Final[int] = 1024
    CHANNELS: Final[int] = 1
    RMS_THRESHOLD: Final[float] = 300.0
    UNPACK_FORMAT: Final[str] = f"<{CHUNK_SIZE}h"

    __slots__ = ("is_active",)

    def __init__(self: Self) -> None:
        self.is_active: bool = False

    def voice_active_detection(self: Self, chunk_audio: Sequence[int]) -> bool:
        "Calculates the RMS to detect speech activity"

        if not chunk_audio:
            return False

        sum_squares: float = sum(float(sample) ** 2 for sample in chunk_audio)
        mean_square: float = sum_squares / len(chunk_audio)
        rms: float = math.sqrt(mean_square)

        return rms > self.RMS_THRESHOLD

    async def _listen(self: Self) -> AsyncGenerator[bytes, None]:
        "High-level orchestration of the audio ingestion loop."

        self.is_active: bool = True
        try:
            while self.is_active:
                raw_data: bytes = b""

                if not raw_data:
                    continue

                numeric_samples: tuple[int, ...] = struct.unpack(
                    self.UNPACK_FORMAT, raw_data
                )

                if self.voice_active_detection(numeric_samples):
                    yield raw_data

        finally:
            self.is_active: bool = False
