import sys
from typing import Any, cast

import pyaudio


def check_hardware() -> None:
    print(f"--- AMBIENTE: Python {sys.version} ---")
    p = pyaudio.PyAudio()
    print("\n--- SENSORES: Dispositivos de Áudio ---")

    for i in range(p.get_device_count()):
        dev = cast(dict[str, Any], p.get_device_info_by_index(i))
        name = dev.get("name")
        inputs = dev.get("maxInputChannels")

        if isinstance(inputs, int) and inputs > 0:
            print(f"ID {i}: {name} | Canais: {inputs}")

    p.terminate()

    print("\n--- MOTOR: GPU & CUDA ---")
    try:
        import torch

        if torch.cuda.is_available():  # type: ignore
            print(f"SUCESSO: GPU Detectada -> {torch.cuda.get_device_name(0)}")  # type: ignore
            v = torch.version.cuda  # type: ignore
            print(f"Versão CUDA: {v}")
        else:
            print("AVISO: CUDA não disponível.")
    except ImportError:
        print("INFO: Módulo 'torch' não instalado.")


if __name__ == "__main__":
    check_hardware()
