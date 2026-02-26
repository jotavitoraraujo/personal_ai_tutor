### --- IMPORTS --- ###
import logging as log
from typing import Any, Self, TypedDict, cast


class Style:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    RESET = "\033[0m"


class Correction(TypedDict):
    original: str
    improved: str
    reason: str


class MentorPayload(TypedDict):
    conversation_response: str
    perfect_version: str
    accuracy_score: int
    corrections: list[Correction]
    pedagogical_tip: str
    proficiency_assessment: str


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
    def show_mentor_response(response: MentorPayload, metrics: dict[str, Any]) -> None:
        try:
            print(f"\n{Style.BOLD}{Style.CYAN}═══ MENTOR ═══{Style.RESET}")
            print(f"{Style.BOLD}{response.get('conversation_response', 'No message content.')}{Style.RESET}")
            reconstruction = response.get("natural_reconstruction")
            if reconstruction:
                print(f"\n{Style.YELLOW}Natural Reconstruction:{Style.RESET}")
                print(f'{Style.ITALIC}{Style.GREEN}"{reconstruction}"{Style.RESET}')

            score = response.get("accuracy_score", 0)
            level = response.get("proficiency_assessment", "N/A")
            print(f"\n{Style.GREEN}Accuracy: {score}% | Level: {level}{Style.RESET}")

            corrections = response.get("corrections", [])
            if corrections:
                print(f"\n{Style.YELLOW}CORRECTIONS:{Style.RESET}")
                for corr in corrections:
                    print(f"  • {Style.RED}{corr.get('original')}{Style.RESET} -> {Style.GREEN}{corr.get('improved')}{Style.RESET}")
                    print(f"    {Style.CYAN}Reason: {corr.get('reason')}{Style.RESET}")

            if response.get("pedagogical_tip"):
                print(f"\n{Style.BOLD}Tip:{Style.RESET} {response.get('pedagogical_tip')}")

            print(f"{Style.CYAN}══════════════{Style.RESET}")

            print(f"{Style.YELLOW}[METRICS]{Style.RESET} Prompt: {metrics.get('prompt_tokens', 0)} tks | Output: {metrics.get('output_tokens', 0)} tks | Latency: {metrics.get('total_time_ms', 0)}ms")

            print(f"\n{Style.GREEN}[CONTROL]{Style.RESET} {Style.BOLD}Hold 'F8' to talk.{Style.RESET}")

        except Exception as e:
            log.error(f"[DM ERROR] Critical failure during rendering: {str(e)}")
            print(f"\n{Style.RED}[SYSTEM ERROR] Could not render full mentor card.{Style.RESET}")

    @staticmethod
    def show_user_transcription(text: str) -> None:
        print(f"\n{Style.BOLD}{Style.RED}═══ USER ═══{Style.RESET}")
        print(f"{Style.BOLD}{text}{Style.RESET}")
        print(f"{Style.RED}════════════{Style.RESET}")
