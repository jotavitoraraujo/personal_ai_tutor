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


class LLMEngine:
    __slots__ = ("endpoint", "model", "student_profile")

    def __init__(self: Self) -> None:
        self.endpoint = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b-instruct-q6_K"
        self.student_profile: list[dict[str, str]] = [
            {"role": "system", "content": ("TEST")}
        ]

    async def think(self: Self, user_input: str) -> str:
        log.info(f"[LLM] Input bytes received: {len(user_input.encode())} bytes.")
        self.student_profile.append({"role": "user", "content": (f"{user_input}")})

        payload = {
            "model": self.model,
            "messages": self.student_profile,
            "stream": False,
            "options": {"num_ctx": 5600},
            "keep_alive": 0,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, json=payload, timeout=30.0)
                if response.status_code == 200:
                    data = cast(OllamaResponse, response.json())
                    bot_message: str = data["message"]["content"]
                    self.student_profile.append(
                        {"role": "assistant", "content": bot_message}
                    )
                    return bot_message
                else:
                    raise Exception(
                        f"Server returned ::: Status Code: {response.status_code}"
                    )
            except Exception as e:
                log.error(f"[ERROR] ::: {e}")
                return "I'm sorry, my brain is cooling down."
