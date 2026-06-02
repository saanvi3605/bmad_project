"""
agents_impl/a2a_client.py
────────────────────────────────────────────────────────────────────────────────
A2A SimpleGenerator agent — choreography-based routing for simple prompts.

Agent definition : agents/a2a_client.md
Skill definition : skills/a2a_request.yaml

Instead of calling project_final_final over HTTP, this agent writes the prompt
to a shared event bus file (a2a_control.yaml) and polls until project_final_final's
own orchestrator picks it up, runs the pipeline, and writes the result back.

This is the same choreography pattern used by the BMAD pipeline itself:
  - control.yaml      ← BMAD's event bus  (orchestrator_agent.py watches it)
  - a2a_control.yaml  ← A2A event bus     (a2a_orchestrator_agent.py watches it)

Prerequisites
─────────────
project_final_final's orchestrator must be running before triggering a simple prompt:
    cd project_final_final
    python agents_impl/a2a_orchestrator_agent.py

Environment variables (optional overrides)
──────────────────────────────────────────
    A2A_CONTROL_FILE   path to the shared event bus YAML
                       default: a2a_control.yaml (in project_bmad root)
    A2A_TIMEOUT        max seconds to wait for result (default: 300)
    A2A_POLL_INTERVAL  seconds between status checks (default: 5)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock, Timeout

# ── Configuration ──────────────────────────────────────────────────────────────

_DEFAULT_CONTROL = Path(__file__).parent.parent / "a2a_control.yaml"

A2A_CONTROL_FILE  = Path(os.getenv("A2A_CONTROL_FILE", str(_DEFAULT_CONTROL)))
A2A_LOCK_FILE     = A2A_CONTROL_FILE.with_suffix(".yaml.lock")
A2A_TIMEOUT       = int(os.getenv("A2A_TIMEOUT", "300"))
A2A_POLL_INTERVAL = int(os.getenv("A2A_POLL_INTERVAL", "5"))
LOCK_TIMEOUT      = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read() -> dict:
    try:
        data = yaml.safe_load(A2A_CONTROL_FILE.read_text(encoding="utf-8")) or {}
        return data
    except Exception:
        return {"request": {"status": "idle", "prompt": "", "requested_at": None},
                "result": {"code": "", "technical_design": "", "functional_spec": "",
                           "error": "", "completed_at": None}}


def _write(data: dict) -> None:
    try:
        with FileLock(str(A2A_LOCK_FILE), timeout=LOCK_TIMEOUT):
            A2A_CONTROL_FILE.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
    except Timeout:
        print("  [A2A] ⚠  Could not acquire event bus lock — retrying...")


def _write_request(prompt: str) -> None:
    """Write a generation request to the event bus."""
    data = _read()
    data["request"] = {
        "status": "pending",
        "prompt": prompt,
        "requested_at": _now(),
    }
    data["result"] = {
        "code": "",
        "technical_design": "",
        "functional_spec": "",
        "error": "",
        "completed_at": None,
    }
    _write(data)


def _reset_bus() -> None:
    """Reset the event bus to idle after reading the result."""
    data = _read()
    data["request"]["status"] = "idle"
    _write(data)


# ── Polling ───────────────────────────────────────────────────────────────────

def _poll_for_result(timeout: int) -> dict:
    """
    Poll a2a_control.yaml until status is 'complete' or 'failed'.
    Raises RuntimeError on timeout or pipeline failure.
    """
    deadline = time.time() + timeout
    waited = 0

    while time.time() < deadline:
        data   = _read()
        status = data.get("request", {}).get("status", "idle")

        if status == "complete":
            result = data.get("result", {})
            _reset_bus()
            return result

        if status == "failed":
            error = data.get("result", {}).get("error", "Unknown error")
            _reset_bus()
            raise RuntimeError(f"[A2A] project_final_final pipeline failed: {error}")

        print(f"  [A2A] Waiting for result… ({waited}s elapsed, status={status})")
        time.sleep(A2A_POLL_INTERVAL)
        waited += A2A_POLL_INTERVAL

    raise RuntimeError(
        f"[A2A] Timed out after {timeout}s waiting for project_final_final.\n"
        "Make sure agents_impl/a2a_orchestrator_agent.py is running in project_final_final."
    )


# ── Agent function ────────────────────────────────────────────────────────────

def simple_generator_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    A2A SimpleGenerator agent — routes simple prompts via choreography.

    Reads:  state["user_request"] (raw prompt, before PromptRefiner)
    Writes: state["code"], state["technical_design"], state["functional_spec"],
            state["a2a_used"]
    """
    prompt = (state.get("user_request", "")).strip()
    score  = state.get("complexity_score", "?")

    print(f"\n  [A2A] Score {score}/10 — routing via choreography event bus")
    print(f"  [A2A] Event bus : {A2A_CONTROL_FILE}")
    print(f"  [A2A] Prompt    : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Check the event bus isn't already busy
    current = _read()
    current_status = current.get("request", {}).get("status", "idle")
    if current_status in ("pending", "running"):
        raise RuntimeError(
            f"[A2A] Event bus is busy (status={current_status}). "
            "Another A2A request is already in progress."
        )

    # Write request to event bus
    _write_request(prompt)
    print(f"  [A2A] Request written to event bus — waiting for orchestrator to pick it up...")

    # Poll until project_final_final writes the result back
    result = _poll_for_result(timeout=A2A_TIMEOUT)

    code = result.get("code", "")
    print(f"  [A2A] Code received ({len(code)} chars) — handing off to Executor.")

    state["code"]             = code
    state["technical_design"] = result.get("technical_design", "")
    state["functional_spec"]  = result.get("functional_spec", "")
    state["a2a_used"]         = True

    return state
