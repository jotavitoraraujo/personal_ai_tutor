### --- IMPORTS --- ###
import json
import logging as log
import re
from typing import Any, Self, TypedDict, cast

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
    __slots__ = ("endpoint", "model", "messages", "cycle_count", "system_contract")

    def __init__(self: Self) -> None:
        self.endpoint = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b-instruct-q6_K"
        self.messages: list[dict[str, str]] = []
        self.cycle_count = 0
        self.system_contract = (
            "SYSTEM CONTRACT — PERSONAL AI TUTOR\n"
            "PERSONALITY: Act as a sharp, witty, and supportive English Language Tutor. Avoid robotic or generic assistant language. Your tone should be conversational and challenging."
            "Role: English language tutor focused on grammar, vocabulary, fluency, and clarity.\n"
            "Operational Protocol:\n"
            "1. Cycles: 3-response collection cycles. State is externally defined.\n"
            "2. Phase Collection: Concisely reply to user without corrections or evaluation.\n"
            "3. Phase Evaluation: Provide a JSON block following this SCHEMA:\n"
            '{"conversation_response": "Your friendly and thought-provoking response", "global_score": 0-10, "vector": "G|V|F|C|SC", '
            '"natural_reconstruction": "Corrected user sentences only", '
            '"dominant_error_pattern": "Technical description", '
            '"micro_challenge": "Focused question"}\n'
            "Style: Dense, technical, no praise. Response Rule: Reply ONLY with 'ACK_CONTRACT' on first initialization."
        )

    async def initialize_layered_mentor(self: Self) -> bool:
        async with httpx.AsyncClient() as client:
            self.messages.append({"role": "system", "content": self.system_contract})
            self.messages.append({"role": "user", "content": "Establish the System Contract and respond."})
            payload = {"model": self.model, "messages": self.messages, "stream": False, "options": {"num_ctx": 5600}, "keep_alive": 0}

            try:
                response = await client.post(self.endpoint, json=payload, timeout=90.0)
                response.raise_for_status()
                data = cast(OllamaResponse, response.json())
                bot_message: str = data["message"]["content"]

                if "ACK_CONTRACT" not in bot_message:
                    log.error(f"[CRITICAL] Contract Reject. Response: {bot_message}")
                    self.messages.clear()
                    return False

                log.info("[SYSTEM] System Contract Validated and Prefilled.")
                self.messages.append({"role": "assistant", "content": bot_message})
                return True

            except Exception as e:
                log.error(f"[ERROR] Connection failure during bootstrapping: {str(e)} ")
                self.messages.clear()
                return False

    def _generate_fallback(self: Self, text: str) -> dict[str, str | int | list[None]]:
        return {"conversation_response": text, "global_score": 0, "vector": "N/A", "natural_reconstruction": "Reconstruction failed.", "dominant_error_pattern": "JSON Parse Error"}

    async def think(self: Self, user_input: str) -> dict[str, Any | int]:
        self.cycle_count += 1
        if self.cycle_count == 3:
            phase_prompt = "Phase: Evaluation. Generate the pedagogical JSON block now including your conversation_response."
        else:
            phase_prompt = f"Phase: Collection ({self.cycle_count}/3). Respond conversationally."
        self.messages.append({"role": "user", "content": f"{phase_prompt}\nUser: {user_input}"})

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
                    raw_content: str = data["message"]["content"]
                    if self.cycle_count == 3:
                        try:
                            json_match = re.search(r"\{.*}", raw_content, re.DOTALL)
                            if json_match:
                                parsed_data = cast(dict[str, Any], json.loads(json_match.group()))
                            else:
                                parsed_data = self._generate_fallback(raw_content)
                            self.cycle_count = 0
                        except (json.JSONDecodeError, AttributeError):
                            parsed_data = self._generate_fallback(raw_content)
                    else:
                        parsed_data = {"conversation_response": raw_content}

                    self.messages.append({"role": "assistant", "content": raw_content})

                    metrics = {
                        "structured_data": parsed_data,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "output_tokens": data.get("eval_count", 0),
                        "total_time_ms": data.get("total_duration", 0) // 1_000_000,
                    }
                    return metrics
                else:
                    raise Exception(f"Server returned ::: Status Code: {response.status_code}")
            except Exception as e:
                log.error(f"[ERROR] Type/Name Error: {type(e).__name__}")
                log.error(f"[ERROR] Cycle {self.cycle_count} failed: {str(e)}")
                return {
                    "structured_data": self._generate_fallback("I'm sorry, my brain is cooling down."),
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
