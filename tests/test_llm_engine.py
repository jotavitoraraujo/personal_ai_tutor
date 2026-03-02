import asyncio
import json
from typing import Any, Self
from unittest.mock import AsyncMock, patch

import httpx

from system.brain.llm_engine import LLMEngine


class MockResponse:
    def __init__(
        self: Self,
        json_data: dict[str, Any],
        status_code: int,
    ) -> None:
        self._json_data: dict[str, Any] = json_data
        self.status_code: int = status_code

    def json(self: Self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self: Self) -> None:
        if self.status_code != 200:
            raise httpx.HTTPStatusError("Mocked Error", request=AsyncMock(), response=AsyncMock())


def test_think_structured_success() -> None:
    engine: LLMEngine = LLMEngine()
    engine.cycle_count = 2
    structured_content: dict[str, Any] = {
        "conversation_response": "Hello! Brazil is great.",
        "global_score": 9,
        "vector": "G9|V8|F9|C10|SC7",
        "natural_reconstruction": "I am from Brazil.",
        "dominant_error_pattern": "None",
        "micro_challenge": "What city?",
    }

    mock_data: dict[str, Any] = {
        "message": {"role": "assistant", "content": json.dumps(structured_content)},
        "prompt_eval_count": 100,
        "eval_count": 50,
        "total_duration": 2000000000,
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: dict[str, Any] = asyncio.run(engine.think("I is from Brazil"))

        assert result["structured_data"]["global_score"] == 9
        assert result["structured_data"]["conversation_response"] == "Hello! Brazil is great."


def test_think_with_markdown_pollution() -> None:
    engine: LLMEngine = LLMEngine()
    engine.cycle_count = 2
    polluted_content: str = 'Sure! ```json\n{"conversation_response": "Valid", "global_score": 10\n```'

    mock_data: dict[str, Any] = {"message": {"content": polluted_content}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: dict[str, Any] = asyncio.run(engine.think("Cleanup test"))

        assert result["structured_data"]["global_score"] == 0


def test_think_catastrophic_fallback() -> None:
    engine: LLMEngine = LLMEngine()
    mock_data: dict[str, Any] = {"message": {"content": "Invalid raw string without JSON structure."}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: dict[str, Any] = asyncio.run(engine.think("Fallback test"))

        assert "Invalid raw string" in result["structured_data"]["conversation_response"]


def test_initialize_layered_mentor_success() -> None:
    engine: LLMEngine = LLMEngine()
    mock_data: dict[str, Any] = {
        "message": {"content": "ACK_CONTRACT"},
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: bool = asyncio.run(engine.initialize_layered_mentor())

        assert result is True
        assert len(engine.messages) == 2


def test_initialize_layered_mentor_failure() -> None:
    engine: LLMEngine = LLMEngine()
    mock_data: dict[str, Any] = {
        "message": {"content": "I am just a normal AI."},
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: bool = asyncio.run(engine.initialize_layered_mentor())

        assert result is False
        assert len(engine.messages) == 0


def test_think_timeout_handling() -> None:
    engine: LLMEngine = LLMEngine()

    with patch("httpx.AsyncClient.post", side_effect=httpx.ReadTimeout("Timeout")):
        result: dict[str, Any] = asyncio.run(engine.think("Timeout test"))

        assert result["structured_data"]["global_score"] == 0
        assert result["prompt_tokens"] == 0
