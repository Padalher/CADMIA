"""Provider registry — build a Provider from an LLMConfig."""
from __future__ import annotations

from typing import Optional

import os

from cadmia.llm.config import LLMConfig, load_config
from cadmia.llm.provider import LLMError, Provider

AVAILABLE = ("anthropic", "openai", "gemini")


def get_provider(config: Optional[LLMConfig] = None) -> Provider:
    config = config or load_config()
    common = {"model": config.resolved_model(), "max_tokens": config.max_tokens,
              "timeout_s": config.timeout_s}
    if config.provider == "anthropic":
        from cadmia.llm.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=config.api_key(), **common)
    if config.provider == "openai":
        from cadmia.llm.openai_compat import OpenAICompatProvider
        # Local/compatible servers (custom base_url) usually need no API key.
        if config.base_url:
            key = os.environ.get(config.api_key_env(), "").strip()
        else:
            key = config.api_key()
        return OpenAICompatProvider(api_key=key, base_url=config.base_url, **common)
    if config.provider == "gemini":
        from cadmia.llm.gemini import GeminiProvider
        return GeminiProvider(api_key=config.api_key(), **common)
    raise LLMError(
        f"Unknown LLM provider '{config.provider}' (available: {', '.join(AVAILABLE)})."
    )
