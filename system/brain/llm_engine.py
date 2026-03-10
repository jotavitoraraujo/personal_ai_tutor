### --- IMPORTS --- ###
import json
import logging as log
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
    __slots__ = ("endpoint", "model", "messages", "json_schema", "system_contract")

    def __init__(self: Self) -> None:
        self.endpoint = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b-instruct-q6_K"
        self.messages: list[dict[str, str]] = []
        self.json_schema = {"type": "object", "properties": {"speech": {"type": "string"}, "template": {"type": "string"}}, "required": ["speech", "template"]}
        self.system_contract = (
            "ROLE: Native English Tutor. USER: Brazilian beginner, analytical learner. "
            "You understand Portuguese perfectly but MUST generate text ONLY in English.\n"
            "PROTOCOL: When user uses Portuguese or makes errors:\n"
            "1. Isolate the target word.\n"
            "2. Teach its direct translation and ONE primary context.\n"
            "3. Issue a short translation challenge using the new word.\n"
            "OUTPUT RULES: Strictly JSON matching the schema.\n"
            "- 'template': ONLY the exact expected answer to the challenge. No theory, no extra words.\n"
            "- 'speech': Micro-lesson + challenge. Max 3 sentences. Direct, strict, no filler.\n"
            "CONSTRAINTS: NEVER speak Portuguese. NO praise. "
            "INIT: If user says 'Establish the System Contract...', output 'ACK_CONTRACT' inside the 'speech' key."
        )

    async def initialize_layered_mentor(self: Self) -> bool:
        if any(key["role"] == "system" for key in self.messages):
            return True

        async with httpx.AsyncClient() as client:
            self.messages.append({"role": "system", "content": self.system_contract})
            self.messages.append({"role": "user", "content": "Establish the System Contract and respond."})
            payload = {"model": self.model, "messages": self.messages, "stream": False, "format": self.json_schema, "options": {"num_ctx": 5600}, "keep_alive": 0}

            try:
                response = await client.post(self.endpoint, json=payload, timeout=90.0)
                response.raise_for_status()
                data = cast(OllamaResponse, response.json())
                raw_string: str = data["message"]["content"]
                json_load: dict[str, str] = json.loads(raw_string)

                if json_load["speech"] != "ACK_CONTRACT":
                    log.error(f"[CRITICAL] Contract Reject. Response: {raw_string}")
                    self.messages.clear()
                    return False

                log.info("[SYSTEM] System Contract Validated and Prefilled.")
                self.messages.append({"role": "assistant", "content": raw_string})
                return True

            except Exception as e:
                log.error(f"[ERROR] Connection failure during bootstrapping: {str(e)} ")
                self.messages.clear()
                return False

    def _generate_fallback(self: Self, raw_string: str) -> dict[str, str]:
        return {"speech": f"[WARNING] The model not returned none content to variable raw_string. Result: {raw_string}", "template": "void"}

    def _remove_last_input(self: Self, messages: list[dict[str, Any]]) -> None:
        del messages[-1]
        return

    async def think(self: Self, user_input: str) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "format": self.json_schema,
            "options": {"num_ctx": 5600},
            "keep_alive": 0,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, json=payload, timeout=90.0)
                if response.status_code == 200:
                    data = cast(OllamaResponse, response.json())
                    raw_string: str = data["message"]["content"]

                    try:
                        if raw_string:
                            json_schema_done = cast(dict[str, str], json.loads(raw_string))
                            self.messages.append({"role": "assistant", "content": raw_string})
                        else:
                            self._remove_last_input(self.messages)
                            json_schema_done = self._generate_fallback(raw_string)

                    except (json.JSONDecodeError, AttributeError):
                        self._remove_last_input(self.messages)
                        json_schema_done = self._generate_fallback(raw_string)

                    metrics = {
                        "structured_data": json_schema_done,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "output_tokens": data.get("eval_count", 0),
                        "total_time_ms": data.get("total_duration", 0) // 1_000_000,
                    }
                    return metrics

                else:
                    raise Exception(f"Server returned ::: Status Code: {response.status_code}")

            except Exception as e:
                log.error(f"[ERROR] Type/Name Error: {type(e).__name__}")
                self._remove_last_input(self.messages)
                return {
                    "structured_data": self._generate_fallback("RAW STRING EMPTY"),
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
