### --- IMPORTS --- ###
import logging as log
from asyncio import Queue
from typing import Self

import numpy as np

from system.audio.baseaudio_engine import BaseAudioEngine as BAE

###
from system.brain.stt_engine import STTEngine as STT
from system.utils.enum_state import State


class LinguistConductor:
    __slots__ = (
        "stt_engine",
        "state",
        "audio_engine",
        "audio_queue",
    )

    def __init__(self: Self) -> None:
        self.stt_engine = STT()
        self.state = State.IDLE
        self.audio_engine = BAE()
        self.audio_queue = Queue[np.ndarray]()

    def on_press(self: Self) -> None:
        if not self.state == State.IDLE:
            return
        else:
            self.state = State.RECORDING
            self.audio_engine.start_capture()

    def on_release(self: Self) -> None:
        if not self.state == State.RECORDING:
            return
        else:
            self.state = State.TRANSCRIBING
            arr_float32 = self.audio_engine.stop_capture()
            self.audio_queue.put_nowait(arr_float32)
            return

    async def run(self: Self) -> None:
        while True:
            payload = await self.audio_queue.get()
            transcription = await self.stt_engine.transcribe(payload)
            if transcription:
                log.info(f"[INFO] Transcription (Your audio): {transcription}")
            else:
                log.warning("[WARNING] Audio not understood or silence detect...")
            self.state = State.IDLE
            log.info(f"[INFO] The system is found in: {self.state}")
