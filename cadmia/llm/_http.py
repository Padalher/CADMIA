"""Shared stdlib HTTP helper for LLM providers (no SDK dependencies)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict

from cadmia.llm.provider import LLMError


def post_json(url: str, headers: Dict[str, str], payload: dict, *,
              timeout_s: int, provider: str) -> dict:
    """POST *payload* as JSON and return the parsed JSON response.

    HTTP and network errors are normalized to LLMError with the provider name
    and the API's own error message when available.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = json.loads(exc.read().decode("utf-8"))
            err = body.get("error", body)
            if isinstance(err, dict):
                detail = err.get("message", "")
        except Exception:
            pass
        raise LLMError(f"{provider} API error {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach the {provider} API: {exc.reason}") from exc
