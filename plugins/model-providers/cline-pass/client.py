"""Normalize Cline's buffered completion envelope without consuming SSE streams."""

from typing import Any

from openai import OpenAI
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion


class ClinePassCompletions(Completions):
    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = super().create(*args, **kwargs)
        if kwargs.get("stream"):
            return response
        # Cline wraps buffered text and tool calls in {success, data}; the SDK preserves data as an extra field.
        data = getattr(response, "data", None)
        if isinstance(data, dict) and "choices" in data:
            return ChatCompletion.model_validate(data)
        return response


class ClinePassClient(OpenAI):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chat.completions = ClinePassCompletions(self)
