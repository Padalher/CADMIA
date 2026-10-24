"""cadmia.llm — optional, provider-agnostic LLM features (text-only, never numeric).

Configured via ~/.cadmia/llm.yaml + an API key environment variable; without them,
the rest of CADMIA works unchanged.
"""
from cadmia.llm.provider import LLMError, LLMResponse, Provider
from cadmia.llm.config import LLMConfig, load_config, DEFAULT_CONFIG_PATH
from cadmia.llm.registry import get_provider
from cadmia.llm.narrator import NarrationResult, narrate_report
from cadmia.llm.diagnostician import SuggestionResult, suggest_fixes
from cadmia.llm.sql import SQLQueryResult, ask_database, validate_readonly_sql

__all__ = [
    "LLMError", "LLMResponse", "Provider",
    "LLMConfig", "load_config", "DEFAULT_CONFIG_PATH",
    "get_provider", "NarrationResult", "narrate_report",
    "SuggestionResult", "suggest_fixes",
    "SQLQueryResult", "ask_database", "validate_readonly_sql",
]
