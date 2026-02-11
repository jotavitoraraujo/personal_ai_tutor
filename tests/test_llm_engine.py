### --- IMPORTS --- ###
import asyncio
from typing import Self
from unittest.mock import AsyncMock, patch

import httpx

from system.brain.llm_engine import LLMEngine

###


class MockResponse:
    def __init__(
        self: Self, json_data: dict[str, str | dict[str, str] | bool], status_code: int
    ) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def json(self: Self) -> dict[str, str | dict[str, str] | bool]:
        return self._json_data


def test_think_success() -> None:
    engine = LLMEngine()
    mock_data = {
        "model": "llama3.1:8b-instruct-q6_K",
        "message": {"role": "assistant", "content": "Keep practicing, Joao!"},
        "done": True,
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)
        result = asyncio.run(engine.think("Hi, I am installing a package."))

        assert result == "Keep practicing, Joao!"
        assert len(engine.student_profile) == 3


def test_think_malformed_json() -> None:
    engine = LLMEngine()
    mock_data = {"status": "error_but_200_ok"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse(mock_data, 200)  # type: ignore
        result = asyncio.run(engine.think("Testing error handling."))

        assert result == "I'm sorry, my brain is cooling down."
        assert len(engine.student_profile) == 2


def test_think_server_error() -> None:
    engine = LLMEngine()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MockResponse({}, 500)
        result = asyncio.run(engine.think("Are you there?"))

        assert result == "I'm sorry, my brain is cooling down."
        assert len(engine.student_profile) == 2


def test_think_timeout() -> None:
    engine = LLMEngine()

    with patch("httpx.AsyncClient.post", side_effect=httpx.ReadTimeout("Timeout")):
        result = asyncio.run(engine.think("This might take too long..."))

        assert result == "I'm sorry, my brain is cooling down."
        assert len(engine.student_profile) == 2
