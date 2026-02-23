### --- IMPORTS --- ###
import asyncio
from typing import Self
from unittest.mock import AsyncMock, patch

import httpx

from system.brain.llm_engine import LLMEngine

###


class MockResponse:
    def __init__(
        self: Self,
        json_data: dict[str, str | dict[str, str] | bool | int],
        status_code: int,
    ) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def json(self: Self) -> dict[str, str | dict[str, str] | bool | int]:
        return self._json_data

    def raise_for_status(self: Self) -> None:
        if self.status_code != 200:
            raise httpx.HTTPStatusError("Mocked Error", request=None, response=self)  # type: ignore


def test_think_success() -> None:
    engine = LLMEngine()
    mock_data = {
        "model": "llama3.1:8b-instruct-q6_K",
        "message": {"role": "assistant", "content": "Keep practicing, Joao!"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 20,
        "total_duration": 1000000000,
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result = asyncio.run(engine.think("Hi"))

        assert result["text"] == "Keep practicing, Joao!"
        assert result["prompt_tokens"] == 10
        assert len(engine.messages) == 2


def test_think_malformed_json() -> None:
    engine = LLMEngine()
    mock_data = {"status": "error_but_200_ok"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)  # type: ignore
        result = asyncio.run(engine.think("Testing error handling."))

        assert result["text"] == "I'm sorry, my brain is cooling down."
        assert len(engine.messages) == 1


def test_think_server_error() -> None:
    engine = LLMEngine()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse({}, 500)
        result = asyncio.run(engine.think("Are you there?"))

        assert result["text"] == "I'm sorry, my brain is cooling down."
        assert len(engine.messages) == 1


def test_think_timeout() -> None:
    engine = LLMEngine()

    with patch("httpx.AsyncClient.post", side_effect=httpx.ReadTimeout("Timeout")):
        result = asyncio.run(engine.think("This might take too long..."))

        assert result["text"] == "I'm sorry, my brain is cooling down."
        assert len(engine.messages) == 1


def test_initialize_layered_mentor_success() -> None:
    engine = LLMEngine()
    mock_data = {
        "message": {"content": "ACK_ID ACK_LOGIC ACK_CONSTRAINTS"},
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)  # type: ignore
        result = asyncio.run(engine.initialize_layered_mentor())

        assert result is True
        assert len(engine.messages) == 3


def test_initialize_layered_mentor_failure() -> None:
    engine = LLMEngine()
    mock_data = {
        "message": {"content": "Hello! I am a friendly AI."},
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)  # type: ignore
        result = asyncio.run(engine.initialize_layered_mentor())

        assert result is False
        assert len(engine.messages) == 0
