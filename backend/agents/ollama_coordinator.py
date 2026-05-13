"""Minimal Ollama-based coordinator.

This coordinator is intentionally lightweight: it sends the coordinator prompt
to a local Ollama model and does not implement the Claude SDK MCP toolset or
Codex JSON-RPC dynamic tools. It supports the same run_event_loop interface
used by other coordinators (turn_fn receives text prompts and should drive
the system), so we adapt the simple HTTP response as the coordinator's reply.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.ollama import generate as ollama_generate
from backend.agents.coordinator_loop import build_deps, run_event_loop
from backend.config import Settings

logger = logging.getLogger(__name__)


async def run_ollama_coordinator(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    coordinator_model: str | None = None,
    msg_port: int = 0,
) -> dict[str, Any]:
    ctfd, cost_tracker, deps = build_deps(settings, model_specs, challenges_root, no_submit)
    deps.msg_port = msg_port

    resolved_model = coordinator_model or settings.ollama_default_model

    async def turn_fn(msg: str) -> None:
        logger.debug(f"Ollama coordinator query: {msg[:200]}")
        # Call Ollama synchronously via helper
        try:
            resp = ollama_generate(settings.ollama_url, resolved_model, msg)
        except Exception as e:
            logger.error(f"Ollama coordinator call failed: {e}")
            return

        # Try to extract textual content robustly
        text = ""
        if isinstance(resp, dict):
            # Ollama outputs may vary by version; check common keys
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("message") or first.get("content", "")
                else:
                    text = str(first)
            else:
                # fallback to stringifying
                text = resp.get("output") or resp.get("content") or str(resp)
        else:
            text = str(resp)

        logger.info("Ollama coordinator turn done")

    return await run_event_loop(deps, ctfd, cost_tracker, turn_fn)
