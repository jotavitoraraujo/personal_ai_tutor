### --- IMPORTS --- ###
import json
import logging as log
import time
from typing import Any, Self, TypedDict, cast

import httpx

from system.utils.config import Config

###


class Metrics(TypedDict):
    "metrics to control the context limit"

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Message(TypedDict):
    "dictionary that have the role and content of the response of model"

    role: str
    content: str
    refusal: str


class Responses(TypedDict):
    "a list of dictionarys"

    index: int
    message: Message
    logprobs: None
    finish_reason: str


class PayloadResponse(TypedDict):
    "structure of the dictionary returned of the openAI api"

    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: list[Responses]
    usage: Metrics


class LLMEngine:
    __slots__ = ("config", "endpoint", "model", "messages", "json_object", "system_contract")

    def __init__(self: Self, config: Config) -> None:
        self.config = config
        self.endpoint = config.endpoint
        self.model = config.model
        self.messages: list[dict[str, str]] = []
        self.json_object = {"type": "json_object"}
        self.system_contract = (
            """ROLE: Expert English Tutor. USER: João (Brazilian, intermediate reader, needs pattern automation and connected speech practice)
            "You understand Portuguese but MUST output strictly in English.\n"""
            "TONE: Conversational, natural, and engaging. FATAL RULE: Under no circumstances act like a formal dictionary or a textbook. Speak like a real human in a voice call.\n"
            "PROTOCOL:\n"
            "1. FEEDBACK: If João attempted a translation, briefly evaluate it. Correct sentence construction contextually.\n"
            "2. MICRO-LESSON: Teach a natural phrase pattern, contraction, or connected speech. Focus on chunks, not isolated words.\n"
            "3. CHALLENGE: End with a short translation challenge for him to practice the new pattern.\n"
            "OUTPUT RULES: Strictly JSON.\n"
            "- 'template': ONLY the exact expected English answer for your NEW challenge. Zero theory.\n"
            "- 'speech': Your feedback + micro-lesson + challenge. Max 3 sentences. Keep it punchy for TTS audio.\n"
            "INIT: If prompt is 'Establish the System Contract...', output 'ACK_CONTRACT' inside 'speech'."
        )

    async def initialize_layered_mentor(self: Self) -> bool:
        if any(key["role"] == "system" for key in self.messages):
            return True

        async with httpx.AsyncClient() as client:
            self.messages.append({"role": "system", "content": self.system_contract})
            self.messages.append({"role": "user", "content": "Establish the System Contract and respond."})
            payload = {"model": self.model, "messages": self.messages, "max_tokens": 500, "temperature": 0.7, "response_format": self.json_object}

            try:
                if not self.config.header:
                    response = await client.post(url=self.endpoint, json=payload, timeout=90.0)
                else:
                    response = await client.post(url=self.endpoint, json=payload, headers=self.config.header, timeout=90.0)

                response.raise_for_status()
                data = cast(PayloadResponse, response.json())
                raw_string: str = data["choices"][0]["message"]["content"]
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
        start_time = time.time()
        self.messages.append({"role": "user", "content": user_input})
        payload = {"model": self.model, "messages": self.messages, "max_tokens": 500, "temperature": 0.7, "response_format": self.json_object}

        async with httpx.AsyncClient() as client:
            try:
                if not self.config.header:
                    response = await client.post(url=self.endpoint, json=payload, timeout=90.0)
                else:
                    response = await client.post(url=self.endpoint, json=payload, headers=self.config.header, timeout=90.0)

                if response.status_code == 200:
                    data = cast(PayloadResponse, response.json())
                    raw_string: str = data["choices"][0]["message"]["content"]

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

                    end_time = time.time()
                    delta = end_time - start_time
                    latency = round(delta, 2)

                    metrics = {
                        "structured_data": json_schema_done,
                        "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                        "output_tokens": data["usage"].get("completion_tokens", 0),
                        "total_tokens": data["usage"].get("total_tokens", 0),
                        "latency": latency,
                    }
                    return metrics

                else:
                    raise Exception(f"Server returned ::: Status Code: {response.status_code}")

            except Exception as e:
                log.error(f"[ERROR] Type/Name Error: {type(e).__name__}")
                self._remove_last_input(self.messages)
                return {"structured_data": self._generate_fallback("RAW STRING EMPTY"), "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency": 0.0}
