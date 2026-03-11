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
from system.ui.display_manager import MentorPayload
from system.utils.config import Config
from system.utils.enum_state import State


class LinguistConductor:
    __slots__ = ("config", "state", "audio_engine", "stt_engine", "llm_engine", "audio_queue", "loop", "initialization")

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        self.state = State.IDLE
        self.audio_engine = BAE()
        self.stt_engine = STT()
        self.llm_engine = LLM(self.config)
        self.audio_queue = Queue[np.ndarray]()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.initialization: bool = False

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
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, arr_float32)

    async def run(self: Self) -> None:
        self.loop = asyncio.get_running_loop()

        if not self.initialization:
            log.info("[SYSTEM] Initialize bootstrapping SLP... ")
            is_ready = await self.llm_engine.initialize_layered_mentor()
            if is_ready:
                self.initialization = True
                log.info("[SYSTEM] Mentor successfully anchored in VRAM.")
                log.info("[CONTROL] Ready! Hold 'F8' to talk.")
            else:
                log.critical("[CRITICAL] Calibration failed. Aborting...")
                return

        try:
            while True:
                payload = await self.audio_queue.get()
                transcription = await self.stt_engine.transcribe(payload)

                if transcription:
                    self.state = State.THINKING
                    DM.print_gpu_status("LLAMA 3.1: Thinking...")
                    DM.show_user_transcription(transcription)

                    metrics_llm = await self.llm_engine.think(transcription)
                    json_schema_llm = cast(MentorPayload, metrics_llm["structured_data"])
                    DM.show_mentor_response(json_schema_llm, metrics_llm)
                    self.state = State.IDLE
                else:
                    log.warning("[WARNING] Audio not understood or silence detect...")
                    DM.print_gpu_status("Status: IDLE (No Speech detected)")
        finally:
            self.state = State.IDLE
