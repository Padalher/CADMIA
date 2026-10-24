"""Provider — minimal LLM provider interface (provider-agnostic, optional).

LLM output is text-only and never feeds the numeric pipeline: parsing,
descriptors and the database never depend on a cloud service.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class LLMError(RuntimeError):
    """Raised when an LLM provider is misconfigured or a request fails."""


@dataclass
class LLMResponse:
    text:          str
    model:         str
    provider:      str
    input_tokens:  Optional[int] = None
    output_tokens: Optional[int] = None


class Provider(ABC):
    """A chat-completion backend. Implementations: Anthropic, OpenAI-compatible, Gemini."""

    name: str = ""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a single-turn prompt and return the model's text response."""
