"""
agents_impl/a2a_client.py
────────────────────────────────────────────────────────────────────────────────
A2A client agent — calls project_final_final's HTTP service to generate code
for simple prompts (complexity score ≤ simple_threshold in config/models.yaml).

Called by graph.py when ComplexityScorer scores the prompt ≤ threshold.
Injects code, technical_design, and functional_spec into state so the rest of
the BMAD pipeline (Validator → Tester → Reviewer → Executor → ...) runs normally
on the returned code.

Prerequisites
─────────────
The project_final_final service must be running before triggering a simple prompt:
    cd project_final_final
    python service.py          # starts on http://localhost:8001

Environment variables (optional overrides)
──────────────────────────────────────────
    A2A_SERVICE_URL   default: http://localhost:8001
    A2A_TIMEOUT       default: 300  (seconds — pipeline takes ~2-3 min)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from typing import Any

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
A2A_SERVICE_URL = os.getenv("A2A_SERVICE_URL", "http://localhost:8001")
A2A_TIMEOUT     = int(os.getenv("A2A_TIMEOUT", "300"))   # 5 minutes


def simple_generator_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    A2A agent node — routes simple prompts to project_final_final via HTTP.

    Reads:  state["refined_prompt"] (or state["user_request"] as fallback)
    Writes: state["code"], state["technical_design"], state["functional_spec"],
            state["a2a_used"]
    """
    prompt = (state.get("refined_prompt") or state.get("user_request", "")).strip()

    score  = state.get("complexity_score", "?")
    print(f"\n  [A2A] Score {score}/10 — routing to project_final_final service")
    print(f"  [A2A] Service : {A2A_SERVICE_URL}/generate")
    print(f"  [A2A] Prompt  : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  [A2A] Waiting for response (pipeline takes ~2-3 minutes)...")

    try:
        response = requests.post(
            f"{A2A_SERVICE_URL}/generate",
            json={"prompt": prompt},
            timeout=A2A_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"\n[A2A] Cannot reach service at {A2A_SERVICE_URL}.\n"
            "Start it first:\n"
            "  cd project_final_final\n"
            "  python service.py\n"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"\n[A2A] Service timed out after {A2A_TIMEOUT}s.\n"
            "The pipeline is taking longer than expected. "
            "Try increasing A2A_TIMEOUT."
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"\n[A2A] Service returned HTTP {exc.response.status_code}:\n"
            f"{exc.response.text}"
        )

    code = data.get("code", "")
    print(f"  [A2A] Code received ({len(code)} chars) — handing off to Validator.")

    state["code"]             = code
    state["technical_design"] = data.get("technical_design", "")
    state["functional_spec"]  = data.get("functional_spec", "")
    state["a2a_used"]         = True

    return state
