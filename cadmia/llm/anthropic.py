"""Anthropic provider — Messages API over stdlib HTTP (no SDK dependency)."""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from cadmia.llm._http import post_json
from cadmia.llm.provider import LLMError, LLMResponse, Provider

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, *, model: str, api_key: str, max_tokens: int = 2048,
                 timeout_s: int = 120):
        if not model:
            raise LLMError("Anthropic provider requires a model name.")
        self.model = model
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def build_request(self, prompt: str, *, system: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> urllib.request.Request:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        return urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> LLMResponse:
        request = self.build_request(prompt, system=system, max_tokens=max_tokens)
        data = post_json(
            API_URL, dict(request.header_items()), json.loads(request.data),
            timeout_s=self.timeout_s, provider="Anthropic",
        )

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        if not text:
            raise LLMError(f"Empty response from Anthropic (stop_reason: {data.get('stop_reason')}).")
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
