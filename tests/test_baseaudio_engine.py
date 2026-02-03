### --- IMPORTS --- ###
import struct
from typing import Self

from system.audio.baseaudio_engine import BaseAudioEngine

###


class AudioEngineMock(BaseAudioEngine):
    def __init__(self: Self) -> None:
        super().__init__()


def generate_audio_chunk(amplitude: int) -> bytes:
    return struct.pack("<1024h", *([amplitude] * 1024))


def test_vad_silence_detection() -> None:
    engine = AudioEngineMock()
    silent_data: list[int] = [0] * engine.CHUNK_SIZE

    assert engine.voice_active_detection(silent_data) is False


def test_vad_voice_detection() -> None:
    engine = AudioEngineMock()
    voice_data: list[int] = [500] * engine.CHUNK_SIZE

    assert engine.voice_active_detection(voice_data) is True


def test_struct_unpack_integration() -> None:
    engine = AudioEngineMock()
    raw_bytes: bytes = generate_audio_chunk(400)
    numeric_sample: tuple[int, ...] = struct.unpack(engine.UNPACK_FORMAT, raw_bytes)

    assert numeric_sample[0] == 400
    assert engine.voice_active_detection(numeric_sample) is True
