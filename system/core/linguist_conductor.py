### --- IMPORTS --- ###
import logging as log
from asyncio import Queue
from typing import Self

import numpy as np

###
from system.audio.baseaudio_engine import BaseAudioEngine as BAE
from system.utils.enum_state import State


class LinguistConductor:
    __slots__ = (
        "state",
        "audio_engine",
        "audio_queue",
    )

    def __init__(self: Self) -> None:
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
            _payload = await self.audio_queue.get()
            log.info(f"[INFO] The system is found in: {self.state}")
