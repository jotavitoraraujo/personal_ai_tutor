import asyncio
from typing import Any, Self
from unittest.mock import AsyncMock, Mock, patch

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


def get_mock_config() -> Mock:
    mock_config = Mock()
    mock_config.endpoint = "http://fake-endpoint"
    mock_config.model = "fake-model"
    mock_config.header = {}
    return mock_config


def test_initialize_layered_mentor_success() -> None:
    engine = LLMEngine(get_mock_config())
    mock_data: dict[str, Any] = {"choices": [{"message": {"content": '{"speech": "ACK_CONTRACT", "template": "void"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: bool = asyncio.run(engine.initialize_layered_mentor())

        assert result is True
        assert len(engine.messages) == 3


def test_initialize_layered_mentor_idempotency() -> None:
    engine = LLMEngine(get_mock_config())
    engine.messages = [{"role": "system", "content": "..."}]

    mock_data: dict[str, Any] = {"choices": [{"message": {"content": '{"speech": "ACK_CONTRACT", "template": "void"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result: bool = asyncio.run(engine.initialize_layered_mentor())

        assert result is True


def test_think_hallucination_rollback() -> None:
    engine = LLMEngine(get_mock_config())
    engine.messages = [{"role": "system", "content": "..."}]
    initial_message_count = len(engine.messages)

    mock_data: dict[str, Any] = {"choices": [{"message": {"content": '{"speech": "Here is your answer...'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)

        result = asyncio.run(engine.think("This will break."))

        assert result["structured_data"]["template"] == "void"
        assert "[WARNING]" in result["structured_data"]["speech"]
        assert len(engine.messages) == initial_message_count


def test_think_network_timeout_fallback() -> None:
    engine = LLMEngine(get_mock_config())

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection timeout")

        result = asyncio.run(engine.think("Hello?"))

        assert "structured_data" in result
        assert "speech" in result["structured_data"]
        assert "template" in result["structured_data"]
