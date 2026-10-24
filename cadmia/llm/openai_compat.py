"""OpenAI-compatible provider — Chat Completions API over stdlib HTTP.

Covers OpenAI itself and any server that speaks the same protocol (Ollama,
llama.cpp, vLLM, LM Studio, Mistral, …) by pointing ``base_url`` at it, e.g.
``base_url: http://localhost:11434/v1`` for a local Ollama. Local servers
usually ignore the Authorization header, so no API key is required when a
custom ``base_url`` is configured.
"""
from __future__ import annotations

from typing import Optional

from cadmia.llm._http import post_json
from cadmia.llm.provider import LLMError, LLMResponse, Provider

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, *, model: str, api_key: str = "", base_url: Optional[str] = None,
                 max_tokens: int = 2048, timeout_s: int = 120):
        if not model:
            raise LLMError("OpenAI-compatible provider requires a model name.")
        self.model = model
        self._api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def build_payload(self, prompt: str, *, system: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # max_tokens is the widely-supported field across compatible servers;
        # OpenAI also accepts it as an alias of max_completion_tokens.
        return {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> LLMResponse:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = post_json(
            f"{self.base_url}/chat/completions",
            headers,
            self.build_payload(prompt, system=system, max_tokens=max_tokens),
            timeout_s=self.timeout_s, provider="OpenAI-compatible",
        )
        choices = data.get("choices") or []
        text = (choices[0].get("message", {}).get("content") or "") if choices else ""
        if not text:
            reason = choices[0].get("finish_reason") if choices else "no choices"
            raise LLMError(f"Empty response from OpenAI-compatible API (finish_reason: {reason}).")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
