"""Ollama solver — simple prompt/response solver adapter.

This solver is a lightweight adapter that sends the system+user prompt to the
local Ollama model and interprets the returned text. It does not implement
the Codex JSON-RPC dynamic tool protocol. Instead, tools are executed locally
by the Python process when the model's response contains special markers
in the form of JSON blocks starting with ```json and containing a dict with
type: "tool_call" or type: "flag_found". This is intentionally minimal but
lets a local Ollama model participate in solver swarms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from backend.ollama import generate as ollama_generate
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.loop_detect import LoopDetector
from backend.models import model_id_from_spec, supports_vision
from backend.prompts import ChallengeMeta, build_prompt, list_distfiles
from backend.sandbox import DockerSandbox
from backend.solver_base import CANCELLED, ERROR, FLAG_FOUND, GAVE_UP, SolverResult
from backend.tracing import SolverTracer

logger = logging.getLogger(__name__)


class OllamaSolver:
    def __init__(
        self,
        model_spec: str,
        challenge_dir: str,
        meta: ChallengeMeta,
        ctfd: CTFdClient,
        cost_tracker: CostTracker,
        settings: object,
        cancel_event: asyncio.Event | None = None,
        no_submit: bool = False,
        submit_fn=None,
        message_bus=None,
        notify_coordinator=None,
    ) -> None:
        self.model_spec = model_spec
        self.model_id = model_id_from_spec(model_spec)
        self.challenge_dir = challenge_dir
        self.meta = meta
        self.message_bus = message_bus
        self.notify_coordinator = notify_coordinator
        self.ctfd = ctfd
        self.cost_tracker = cost_tracker
        self.settings = settings
        self.cancel_event = cancel_event or asyncio.Event()
        self.no_submit = no_submit
        self.submit_fn = submit_fn

        self.sandbox = DockerSandbox(
            image=getattr(settings, "sandbox_image", "ctf-sandbox"),
            challenge_dir=challenge_dir,
            memory_limit=getattr(settings, "container_memory_limit", "4g"),
        )
        self.use_vision = supports_vision(model_spec)
        self.loop_detector = LoopDetector()
        self.tracer = SolverTracer(meta.name, self.model_id)
        self.agent_name = f"{meta.name}/{self.model_id}"

        self._step_count = 0
        self._flag = None
        self._confirmed = False
        self._findings = ""

    async def start(self) -> None:
        await self.sandbox.start()
        arch_result = await self.sandbox.exec("uname -m", timeout_s=10)
        container_arch = arch_result.stdout.strip() or "unknown"

        distfile_names = list_distfiles(self.challenge_dir)
        self.system_prompt = build_prompt(self.meta, distfile_names, container_arch=container_arch)
        self.tracer.event("start", challenge=self.meta.name, model=self.model_id)
        logger.info(f"[{self.agent_name}] Ollama solver started")

    async def run_until_done_or_gave_up(self) -> SolverResult:
        t0 = time.monotonic()
        try:
            prompt_text = "Solve this CTF challenge."
            # Combine system prompt + user prompt
            full_prompt = f"{self.system_prompt}\n\n{prompt_text}"

            # Call Ollama
            resp = ollama_generate(self.settings.ollama_url, self.model_id, full_prompt)

            text = ""
            if isinstance(resp, dict):
                # Try common fields
                if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                    first = resp["choices"][0]
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("message") or first.get("content", "")
                    else:
                        text = str(first)
                else:
                    text = resp.get("output") or resp.get("content") or str(resp)
            else:
                text = str(resp)

            self._step_count += 1
            self.tracer.model_response(text[:500], self._step_count, input_tokens=0, output_tokens=0)

            # Look for JSON block markers for structured output
            flag = None
            try:
                # naive: find first ```json ... ``` block
                if "```json" in text:
                    start = text.index("```json") + len("```json")
                    end = text.index("```", start)
                    block = text[start:end].strip()
                    parsed = json.loads(block)
                    if isinstance(parsed, dict) and parsed.get("type") == "flag_found":
                        flag = parsed.get("flag")
            except Exception:
                flag = None

            if flag:
                self._flag = flag
                self._confirmed = True if self.no_submit else False
                self._findings = f"Flag found: {flag}"

            duration = time.monotonic() - t0
            self.tracer.event("turn_complete", duration=round(duration, 1), steps=self._step_count)

            if self._confirmed and self._flag:
                return self._result(FLAG_FOUND)
            return self._result(GAVE_UP)

        except Exception as e:
            logger.error(f"[{self.agent_name}] Error: {e}", exc_info=True)
            self._findings = f"Error: {e}"
            self.tracer.event("error", error=str(e))
            return self._result(ERROR)

    def bump(self, insights: str) -> None:
        # Simple bump: prepend insights to next run (not fully implemented)
        self.system_prompt = f"{insights}\n\n{self.system_prompt}"
        self.loop_detector.reset()
        self.tracer.event("bump", insights=insights[:500])

    def _result(self, status: str) -> SolverResult:
        self.tracer.event("finish", status=status, flag=self._flag, confirmed=self._confirmed)
        return SolverResult(
            flag=self._flag,
            status=status,
            findings_summary=self._findings[:2000],
            step_count=self._step_count,
            cost_usd=0.0,
            log_path=self.tracer.path,
        )

    async def stop(self) -> None:
        self.tracer.event("stop", step_count=self._step_count)
        self.tracer.close()
        if self.sandbox:
            await self.sandbox.stop()
