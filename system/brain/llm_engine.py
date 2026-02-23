### --- IMPORTS --- ###
import logging as log
from typing import Self, TypedDict, cast

import httpx

###


class Message(TypedDict):
    role: str
    content: str


class OllamaResponse(TypedDict):
    model: str
    created_at: str
    message: Message
    done: bool
    ### METRICS ###
    total_duration: int
    prompt_eval_count: int
    eval_count: int
    eval_duration: int


class LLMEngine:
    __slots__ = ("endpoint", "model", "messages", "slp_payload")

    def __init__(self: Self) -> None:
        self.endpoint = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b-instruct-q6_K"
        self.messages: list[dict[str, str]] = []
        self.slp_payload: list[dict[str, str]] = [
            {
                "stage": "1_Identity",
                "role": "system",
                "content": (
                    "IDENTITY LAYER: You are the 'Elite Polymath English Mentor'. "
                    "Apply the 'Architectural Defense' dialectic. "
                    "Never be purely supportive; challenge assumptions rigorously. "
                    "RESPONSE RULE: Reply ONLY with 'ACK_ID'."
                ),
                "expected_ack": "ACK_ID",
            },
            {
                "stage": "2_Logic",
                "role": "system",
                "content": (
                    "LOGIC LAYER: Activate SLA frameworks (Krashen i+1, Swain Output, Vygotsky).Force complex sentence production and scaffold logically.RESPONSE RULE: Reply ONLY with 'ACK_LOGIC'."
                ),
                "expected_ack": "ACK_LOGIC",
            },
            {
                "stage": "3_Constraints",
                "role": "system",
                "content": (
                    "CONSTRAINTS LAYER: Enforce Bilingual Protocol. "
                    "Feedback/Grammar/IPA MUST be in PT-BR. Debates MUST be in EN. "
                    "Always follow the Mandatory Output Schema. "
                    "RESPONSE RULE: Reply ONLY with 'ACK_CONSTRAINTS'."
                ),
                "expected_ack": "ACK_CONSTRAINTS",
            },
        ]

    async def initialize_layered_mentor(self: Self) -> bool:
        async with httpx.AsyncClient() as client:
            for layer in self.slp_payload:
                log.info(f"[SYSTEM] Injecting {layer['stage']}...")
                self.messages.append({"role": layer["role"], "content": layer["content"]})
                payload = {"model": self.model, "messages": self.messages, "stream": False, "options": {"num_ctx": 5600}, "keep_alive": 0}

                try:
                    response = await client.post(self.endpoint, json=payload, timeout=90.0)
                    response.raise_for_status()
                    data = cast(OllamaResponse, response.json())
                    bot_message: str = data["message"]["content"]

                    if layer["expected_ack"] not in bot_message:
                        log.error(f"[ERROR] Fail the handshake in {layer['stage']}. Response: {bot_message}")
                        self.messages.clear()
                        return False

                    log.info(f"[SYSTEM] {layer['stage']} Validated with Sucess.")

                except Exception as e:
                    log.error(f"[ERROR] Exception during the injecting SLP: {str(e)} ")
                    self.messages.clear()
                    return False
        log.info("[SYSTEM] Elite Polymath English Mentor initialized.")
        return True

    async def think(self: Self, user_input: str) -> dict[str, str | int]:
        log.info(f"[LLM] Input bytes received: {len(user_input.encode())} bytes.")
        self.messages.append({"role": "user", "content": (f"{user_input}")})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "options": {"num_ctx": 5600},
            "keep_alive": 0,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, json=payload, timeout=90.0)
                if response.status_code == 200:
                    data = cast(OllamaResponse, response.json())
                    bot_message: str = data["message"]["content"]
                    self.messages.append({"role": "assistant", "content": bot_message})

                    metrics = {
                        "text": bot_message,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "output_tokens": data.get("eval_count", 0),
                        "total_time_ms": data.get("total_duration", 0) // 1_000_000,
                    }
                    return metrics
                else:
                    raise Exception(f"Server returned ::: Status Code: {response.status_code}")
            except Exception as e:
                log.error(f"[ERROR] ::: Type: {type(e).__name__} | Details: {str(e)}")
                return {
                    "text": "I'm sorry, my brain is cooling down.",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
