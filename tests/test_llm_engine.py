import asyncio
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


def test_initialize_layered_mentor_success() -> None:
    engine = LLMEngine()
    mock_data: dict[str, Any] = {
        "message": {"content": '{"speech": "ACK_CONTRACT", "template": "void"}'},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: bool = asyncio.run(engine.initialize_layered_mentor())

        assert result is True
        assert len(engine.messages) == 3


def test_initialize_layered_mentor_idempotency() -> None:
    engine = LLMEngine()
    mock_data: dict[str, Any] = {
        "message": {"content": '{"speech": "ACK_CONTRACT", "template": "void"}'},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)

        asyncio.run(engine.initialize_layered_mentor())
        asyncio.run(engine.initialize_layered_mentor())

        system_messages = [msg for msg in engine.messages if msg["role"] == "system"]

        assert len(system_messages) == 1, f"[WARNING]! Was found {len(system_messages)} copies of the contract."
        assert len(engine.messages) == 3


def test_initialize_layered_mentor_hallucination() -> None:
    engine = LLMEngine()
    mock_data: dict[str, Any] = {
        "message": {"content": "I am just a normal AI and I say ACK_CONTRACT."},
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: bool = asyncio.run(engine.initialize_layered_mentor())

        assert result is False
        assert len(engine.messages) == 0


def test_think_structured_success() -> None:
    engine = LLMEngine()
    engine.messages = [{"role": "system", "content": "..."}]

    raw_response = '{"speech": "The word is soft. Can you use it?", "template": "That is very soft."}'
    mock_data: dict[str, Any] = {
        "message": {"content": raw_response},
        "prompt_eval_count": 50,
        "eval_count": 20,
        "total_duration": 1500000000,  # 1.5 seconds
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)

        result = asyncio.run(engine.think("Aquele tecido é macio."))

        assert "structured_data" in result
        assert result["structured_data"]["speech"] == "The word is soft. Can you use it?"
        assert result["structured_data"]["template"] == "That is very soft."

        assert engine.messages[-1]["role"] == "assistant"
        assert engine.messages[-1]["content"] == raw_response


def test_think_hallucination_rollback() -> None:
    engine = LLMEngine()
    engine.messages = [{"role": "system", "content": "..."}]
    initial_message_count = len(engine.messages)

    mock_data: dict[str, Any] = {
        "message": {"content": '{"speech": "Here is your answer...'},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)

        result = asyncio.run(engine.think("This will break."))

        assert result["structured_data"]["template"] == "void"
        assert "[WARNING]" in result["structured_data"]["speech"]
        assert len(engine.messages) == initial_message_count


def test_think_network_timeout_fallback() -> None:
    engine = LLMEngine()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection timeout")

        result = asyncio.run(engine.think("Hello?"))

        assert "structured_data" in result
        assert "speech" in result["structured_data"]
        assert "template" in result["structured_data"]
        assert result["structured_data"]["template"] == "void"
        assert result["output_tokens"] == 0
