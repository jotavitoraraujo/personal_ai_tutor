### --- IMPORTS --- ###
import asyncio
import logging as log
from asyncio import Queue
from typing import Self, cast

import numpy as np

from system.audio.baseaudio_engine import BaseAudioEngine as BAE
from system.brain.llm_engine import LLMEngine as LLM

###
from system.brain.stt_engine import STTEngine as STT
from system.ui.display_manager import DisplayManager as DM
from system.utils.enum_state import State


class LinguistConductor:
    __slots__ = (
        "state",
        "audio_engine",
        "stt_engine",
        "llm_engine",
        "audio_queue",
        "loop",
    )

    def __init__(self: Self) -> None:
        self.state = State.IDLE
        self.audio_engine = BAE()
        self.stt_engine = STT()
        self.llm_engine = LLM()
        self.audio_queue = Queue[np.ndarray]()
        self.loop = asyncio.get_event_loop()

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
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, arr_float32)
            return

    async def run(self: Self) -> None:
        while True:
            payload = await self.audio_queue.get()
            transcription = await self.stt_engine.transcribe(payload)

            if transcription:
                self.state = State.THINKING
                DM.print_gpu_status("LLAMA 3.1: Thinking...")
                result = await self.llm_engine.think(transcription)
                DM.show_mentor_response(cast(str, result["text"]), result)
            else:
                log.warning("[WARNING] Audio not understood or silence detect...")
                DM.print_gpu_status("Status: IDLE (No Speech detected)")
            self.state = State.IDLE
