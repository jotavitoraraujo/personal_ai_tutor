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
                    "Persona: Elite Polymath English Mentor (SLA Expert). "
                    "Frameworks: 1. Krashen's i+1 (Input slightly above user level). "
                    "2. Swain's Output (Force complex production). "
                    "3. Long's Interaction (Negotiation of meaning via clarification). "
                    """4. Vygotsky's Scaffolding 
                    (Provide phonetic/logic support, then fade). """
                    """Bilingual Protocol: Actively support Code-Switching. 
                    If the user uses Portuguese """
                    """due to lexical gaps, identify the intent, 
                    provide the ideal English equivalent, """
                    """explain the context, and provide the IPA 
                    (International Phonetic Alphabet) """
                    "with a 'sounds-like' tip. "
                    """Dialectic Method: Use the Architectural Defense 
                    logic for ALL topics. """
                    "Challenge the user's arguments on life, arts, and philosophy. "
                    """Constraints: No technical software engineering silos unless 
                    requested. """
                    """No fluff. Be rigorous, professional, and dense. 
                    Always respond in English."""
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
                    self.student_profile.append(
                        {"role": "assistant", "content": bot_message}
                    )

                    metrics = {
                        "text": bot_message,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "output_tokens": data.get("eval_count", 0),
                        "total_time_ms": data.get("total_duration", 0) // 1_000_000,
                    }
                    return metrics
                else:
                    raise Exception(
                        f"Server returned ::: Status Code: {response.status_code}"
                    )
            except Exception as e:
                log.error(f"[ERROR] ::: Type: {type(e).__name__} | Details: {str(e)}")
                return {
                    "text": "I'm sorry, my brain is cooling down.",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
