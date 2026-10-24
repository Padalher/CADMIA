"""Google Gemini provider — generateContent API over stdlib HTTP."""
from __future__ import annotations

from typing import Optional

from cadmia.llm._http import post_json
from cadmia.llm.provider import LLMError, LLMResponse, Provider

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, *, model: str, api_key: str, max_tokens: int = 2048,
                 timeout_s: int = 120):
        if not model:
            raise LLMError("Gemini provider requires a model name.")
        self.model = model
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def build_payload(self, prompt: str, *, system: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> dict:
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens or self.max_tokens},
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        return payload

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> LLMResponse:
        data = post_json(
            f"{BASE_URL}/{self.model}:generateContent",
            {"x-goog-api-key": self._api_key},
            self.build_payload(prompt, system=system, max_tokens=max_tokens),
            timeout_s=self.timeout_s, provider="Gemini",
        )
        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content", {}).get("parts", [])) if candidates else []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            reason = candidates[0].get("finishReason") if candidates else "no candidates"
            raise LLMError(f"Empty response from Gemini (finishReason: {reason}).")
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            model=data.get("modelVersion", self.model),
            provider=self.name,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
        )
