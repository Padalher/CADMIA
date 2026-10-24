"""LLM configuration — ~/.cadmia/llm.yaml + API key from environment.

The YAML file holds only non-secret settings (provider, model, base_url, max_tokens).
API keys are read exclusively from environment variables and are never written to
disk or to the database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cadmia.llm.provider import LLMError

DEFAULT_CONFIG_PATH = Path.home() / ".cadmia" / "llm.yaml"

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-5-mini",
    "gemini":    "gemini-2.5-flash",
}

API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}


@dataclass
class LLMConfig:
    provider:   str = "anthropic"
    model:      Optional[str] = None     # provider default when None
    base_url:   Optional[str] = None     # for OpenAI-compatible/local endpoints
    max_tokens: int = 2048
    timeout_s:  int = 120

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    def api_key_env(self) -> str:
        return API_KEY_ENV.get(self.provider, f"{self.provider.upper()}_API_KEY")

    def api_key(self) -> str:
        env = self.api_key_env()
        key = os.environ.get(env, "").strip()
        if not key:
            raise LLMError(
                f"API key not found: set the {env} environment variable "
                f"(LLM features are optional; everything else works without it)."
            )
        return key


def load_config(path: "str | Path | None" = None) -> LLMConfig:
    """Load ~/.cadmia/llm.yaml (or *path*); missing file returns defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return LLMConfig()
    import yaml
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise LLMError(f"Invalid LLM config (expected a mapping): {cfg_path}")
    known = {"provider", "model", "base_url", "max_tokens", "timeout_s"}
    # Flag secret-bearing keys with a clear message (only outside the whitelist, so
    # legitimate keys like max_tokens are never caught by the "token" marker).
    _secret_markers = ("key", "token", "secret", "password", "passwd")
    if any(k not in known and any(m in k.lower() for m in _secret_markers) for k in data):
        raise LLMError(f"API keys/secrets must not be stored in {cfg_path}; use environment variables.")
    unknown = set(data) - known
    if unknown:
        raise LLMError(f"Unknown keys in {cfg_path}: {', '.join(sorted(unknown))}")
    return LLMConfig(**data)
