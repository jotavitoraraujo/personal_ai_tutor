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
    __slots__ = ("endpoint", "model", "messages", "slp_payload")

    def __init__(self: Self) -> None:
        self.endpoint = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b-instruct-q6_K"
        self.messages: list[dict[str, str]] = []
        self.slp_payload: list[dict[str, str]] = [
            {
                "stage": "1_Identity",
                "role": "user",
                "content": (
                    "IDENTITY LAYER: You are a 'Pedagogical English Mentor' specialized in 'Conversational Immersion'. "
                    "Your goal is to provide a high-fidelity environment for language practice. "
                    "Maintain a supportive and natural dialogue while systematically identifying linguistic "
                    "improvement points to be presented in the structured feedback card. "
                    "Apply the 'Architectural Defense' method: prioritize conversational flow while "
                    "meticulously monitoring syntax and grammar patterns for pedagogical analysis. "
                    "Maintain high linguistic standards in an encouraging professional tone. "
                    "RESPONSE RULE: Reply ONLY with 'ACK_ID'."
                ),
                "expected_ack": "ACK_ID",
            },
            {
                "stage": "2_Logic",
                "role": "user",
                "content": (
                    "LOGIC LAYER: Activate SLA frameworks (Krashen i+1, Swain Output, Vygotsky).Force complex sentence production and scaffold logically.RESPONSE RULE: Reply ONLY with 'ACK_LOGIC'."
                ),
                "expected_ack": "ACK_LOGIC",
            },
            {
                "stage": "3_Constraints",
                "role": "user",
                "content": (
                    "CONSTRAINTS LAYER: Strict Bilingual Separation Protocol. "
                    "1. FIELD ISOLATION: The key 'conversation_response' MUST ALWAYS be in English. Under no circumstances should this field contain Portuguese. "
                    "Even if the user makes mistakes, you maintain the English immersion."
                    "2. FEEDBACK ISOLATION: The keys 'corrections', 'pedagogical_tip', and 'proficiency_assessment' MUST ALWAYS be in Portuguese (PT-BR). "
                    "This is the only place where Portuguese is permitted."
                    "3. LOGICAL AUDIT: If the user speaks English, do NOT treat it as 'broken Portuguese'. Your pedagogical audit must evaluate the user's English proficiency only. "
                    "4. HOLISTIC AUDIT: You must audit sentence structure, verb tenses, and quantifiers. "
                    "Errors like 'drink very coffee' or 'I build today' must be identified as high-priority structural failures."
                    "5. SEMANTIC FIDELITY: Your reconstruction must preserve the user's original meaning. "
                    "If the user talks about 'programming today', your reconstruction must be about 'programming today'. "
                    "Do not invent new topics or change the user's story."
                    "6. AUDITOR TONE: In the 'corrections' list, do not be conversational. Use strict, direct academic corrections only. Format: 'wrong word -> correct word'."
                    "RESPONSE RULE: To confirm these rigid boundaries, reply ONLY with 'ACK_CONSTRAINTS'."
                ),
                "expected_ack": "ACK_CONSTRAINTS",
            },
            {
                "stage": "4_Formatting",
                "role": "user",
                "content": (
                    "FORMATTING LAYER: You are a JSON generator. "
                    "Every response MUST be a valid JSON object. "
                    "SCHEMA: {"
                    '"conversation_response": "Your reply in English", '
                    "\"natural_reconstruction\": \"ONLY the corrected version of the user's sentence. Do not explain intent. Example: 'I drink very coffee' -> 'I drank a lot of coffee'\","
                    '"accuracy_score": 0-100, '
                    '"corrections": [{"original": "...", "improved": "...", "reason": "..."}], '
                    '"pedagogical_tip": "Explicação em PT-BR", '
                    '"proficiency_assessment": "A1|A2|B1..."'
                    "}. "
                    "STRICT RULE: No conversational text outside JSON. No markdown blocks. "
                    "RESPONSE RULE: Reply ONLY with 'ACK_FORMATTING'."
                ),
                "expected_ack": "ACK_FORMATTING",
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
                    self.messages.append({"role": "assistant", "content": bot_message})

                except Exception as e:
                    log.error(f"[ERROR] Exception during the injecting SLP: {str(e)} ")
                    self.messages.clear()
                    return False
        log.info("[SYSTEM] Elite Polymath English Mentor initialized.")
        return True

    def _generate_fallback(self: Self, text: str) -> dict[str, str | int | list[None]]:
        return {"conversation_response": text, "accuracy_score": 0, "corrections": [], "pedagogical_tip": "Erro no processamento pedagógico.", "proficiency_assessment": "N/A"}

    async def think(self: Self, user_input: str) -> dict[str, Any | int]:
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
                    raw_content: str = data["message"]["content"]
                    try:
                        json_match = re.search(r"\{.*}", raw_content, re.DOTALL)
                        if json_match:
                            parsed_data = cast(dict[str, Any], json.loads(json_match.group()))
                        else:
                            parsed_data = self._generate_fallback(raw_content)
                    except (json.JSONDecodeError, AttributeError):
                        parsed_data = self._generate_fallback(raw_content)

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
                log.error(f"[ERROR] ::: Type: {type(e).__name__} | Details: {str(e)}")
                return {
                    "structured_data": self._generate_fallback("I'm sorry, my brain is cooling down."),
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                }
