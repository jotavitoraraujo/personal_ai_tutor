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
    __slots__ = ("endpoint", "model", "student_profile")

    def __init__(self: Self) -> None:
        self.endpoint = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b-instruct-q6_K"
        self.student_profile: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    """
                    SYSTEM ROLE: Elite Polymath English Mentor.
                    CORE LOGIC: Architectural Defense (Dialectic Challenge).

                    MANDATORY RESPONSE STRUCTURE:
                    1. [FEEDBACK & DEBUG] (In PT-BR):
                    - Review user's grammar/vocab/phonetics.
                    - Provide IPA + 'sounds-like' tips for difficult words.
                    - Explain WHY it was wrong or how to improve.

                    2. [DIALECTIC DEBATE] (In EN):
                    - Counter-argument or probe the user's logic on the topic.
                    - Use high-level vocabulary (Krashen i+1).

                    3. [FINAL CHALLENGE] (In EN):
                    - A single, incisive question to force complex output (Swain's Output).

                    CONSTRAINTS: 
                    - Never skip the PT-BR section. 
                    - Never be purely supportive; challenge every assumption.
                    - No software engineering silos.
                    """
                ),
            }
        ]

    async def think(self: Self, user_input: str) -> dict[str, str | int]:
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
                response = await client.post(self.endpoint, json=payload, timeout=90.0)
                if response.status_code == 200:
                    data = cast(OllamaResponse, response.json())
                    bot_message: str = data["message"]["content"]
                    self.student_profile.append({"role": "assistant", "content": bot_message})

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
