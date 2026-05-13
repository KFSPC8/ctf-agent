"""Simple Ollama HTTP client helper."""

from __future__ import annotations

import httpx
import json
from typing import Any


def generate(ollama_url: str, model: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> dict[str, Any]:
    """Call Ollama local generate endpoint and return parsed JSON response.

    Try several common endpoint paths and payload shapes to be tolerant of
    different Ollama releases or wrappers. Returns the parsed JSON response
    from the first successful call.
    """
    base = ollama_url.rstrip("/")
    paths = ["/api/generate", "/v1/generate", "/api/v1/generate", "/generate"]
    headers = {"Content-Type": "application/json"}

    payloads = [
        {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"model": model, "input": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"model": model, "prompt": prompt},
        {"model": model, "input": prompt},
    ]

    last_err: Exception | None = None
    for p in paths:
        url = base + p
        for body in payloads:
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, headers=headers, content=json.dumps(body))
                    resp.raise_for_status()
                    # parse JSON if possible
                    try:
                        return resp.json()
                    except Exception:
                        # return raw text wrapped
                        return {"output": resp.text}
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Ollama generate request failed (tried multiple endpoints): {last_err}")
