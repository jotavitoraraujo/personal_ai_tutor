### --- IMPORTS --- ###
from typing import Any, Self, TypedDict, cast


class Style:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class GPUTelemetry(TypedDict):
    name: str
    used: float
    total: float


class DisplayManager:
    _nvml_initialized = False

    @classmethod
    def get_gpu_telemetry(cls: type[Self]) -> dict[str, str | float]:
        try:
            if not cls._nvml_initialized:
                import pynvml

                pynvml.nvmlInit()
                cls._nvml_initialized = True

            from pynvml import nvmlDeviceGetHandleByIndex as d_handle  # type: ignore
            from pynvml import nvmlDeviceGetMemoryInfo as mem_info  # type: ignore
            from pynvml import nvmlDeviceGetName as d_name  # type: ignore

            handle = cast(Any, d_handle(0))
            name: bytes = d_name(handle)
            info = cast(Any, mem_info(handle))

            return {
                "name": name.decode("utf-8") if hasattr(name, "decode") else str(name),
                "used": float(info.used) / (1024**3),
                "total": float(info.total) / (1024**3),
            }
        except Exception:
            return {"name": "N/A", "used": 0.0, "total": 1.0}

    @classmethod
    def print_gpu_status(cls: type[Self], action_msg: str) -> None:
        gpu = cls.get_gpu_telemetry()
        if isinstance(gpu["used"], float) and isinstance(gpu["total"], float):
            total = gpu["total"] if gpu["total"] > 0 else 1.0
            used = float(gpu["used"])
            bar_length = 20
            filled = int(round(bar_length * used / total))
            bar = "█" * filled + "░" * (bar_length - filled)

            print(f"{Style.YELLOW}[GPU]{Style.RESET}{Style.BOLD}{gpu['name']}{Style.RESET}{gpu['used']:.1f}/{gpu['total']:.1f} GB | {bar} | {Style.CYAN}{action_msg}{Style.RESET}")

    @staticmethod
    def show_mentor_response(response: str, metrics: dict[str, str | int]) -> None:
        print(f"\n{Style.BOLD}{Style.CYAN}═══ MENTOR ═══{Style.RESET}")
        print(f"{Style.BOLD}{response}{Style.RESET}")
        print(f"{Style.CYAN}══════════════{Style.RESET}")
        print(
            f"""\n{Style.YELLOW}[METRICS]{Style.RESET} 
            Prompt: {metrics["prompt_tokens"]} tks | "
            Output: {metrics["output_tokens"]} tks | 
            Latency: {metrics["total_time_ms"]}ms"""
        )

        print(
            f"""\n{Style.GREEN}[CONTROL]{Style.RESET} 
            {Style.BOLD}Hold 'F8' to talk.{Style.RESET}"""
        )

    @staticmethod
    def show_user_transcription(text: str) -> None:
        print(f"\n{Style.BOLD}{Style.RED}═══ USER ═══{Style.RESET}")
        print(f"{Style.BOLD}{text}{Style.RESET}")
        print(f"{Style.RED}════════════{Style.RESET}")
