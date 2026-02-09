import threading
from typing import Final, Self

import numpy as np
import pyaudio


class BaseAudioEngine:
    SAMPLE_RATE: Final[int] = 16000
    CHUNK_SIZE: Final[int] = 1024
    CHANNELS: Final[int] = 1
    FORMAT: Final[int] = pyaudio.paInt16

    def __init__(self: Self) -> None:
        self.pa = pyaudio.PyAudio()
        self.stream: pyaudio.Stream | None = None
        self._frames: list[bytes] = []
        self._is_recording: bool = False
        self._thread: threading.Thread | None = None

    def _recording_loop(self: Self) -> None:
        if self.stream is None:
            return

        while self._is_recording:
            try:
                data: bytes = self.stream.read(
                    self.CHUNK_SIZE, exception_on_overflow=False
                )
                self._frames.append(data)
            except OSError:
                break

    def stop_capture(self: Self) -> np.ndarray:
        self._is_recording = False

        if self._thread is not None:
            self._thread.join()

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        combined_bytes: bytes = b"".join(self._frames)
        remainder: int = len(combined_bytes) % 2

        if remainder != 0:
            combined_bytes = combined_bytes[:-remainder]

        audio_int16: np.ndarray = np.frombuffer(combined_bytes, dtype=np.int16)
        audio_float32: np.ndarray = audio_int16.astype(np.float32) / 32768.0

        return audio_float32
