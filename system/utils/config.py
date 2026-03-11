### --- IMPORTS --- ###
import os
from typing import Self

import dotenv


class Config:
    def __init__(self: Self) -> None:
        dotenv.load_dotenv()
        self.environment = os.getenv("ENVIRONMENT")
        self.API_KEY = os.getenv("GROQ_API_KEY")
        self.model: str
        self.endpoint: str
        self.header: dict[str, str] | None = {}
        self._set_endpoint()

    def _set_endpoint(self: Self) -> None:
        if self.environment == "development":
            self.endpoint = "http://localhost:11434/v1/chat/completions"
            self.model = "llama3.1:8b-instruct-q6_K"

        if self.environment == "production":
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "llama-3.1-8b-instant"
            self.header = {"Authorization": f"Bearer {self.API_KEY}"}
