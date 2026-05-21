"""
agents_impl/complexity_scorer_agent.py
────────────────────────────────────────────────────────────────────────────────
Scores the refined prompt 1–10 for complexity and optionally overrides the
heavy model with the light model for the rest of the pipeline run.

Position in graph: PromptRefiner → ComplexityScorer → Planner → ...

Model selection logic
  score ≤ simple_threshold (config)  →  complexity_model_override = simple_model
  score >  simple_threshold          →  complexity_model_override = None (use heavy)

run_agent() in agent_runner.py checks state["complexity_model_override"] before
every heavy-model call and routes to the override LLM when set.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from typing import Any

from core.agent_runner import run_agent
from core.llm_factory import get_config

_SCORER_PROMPT = """\
You are a software complexity classifier. Rate the following Streamlit app \
specification on a scale from 1 to 10.

Complexity scale:
  1-2  Trivial    — single feature, no persistence (e.g. show current time)
  3-4  Simple     — 1-2 features, basic CRUD on one entity (e.g. todo list)
  5-6  Moderate   — 3-5 features, multi-table SQLite, charts, filters
  7-8  Complex    — many features, complex data relationships, multiple views
  9-10 Very complex — multi-entity model, advanced charts, file I/O, external APIs

App specification:
{prompt}

Respond ONLY in this exact format (no other text):
SCORE: <integer 1-10>
REASON: <one sentence explaining the score>"""


def complexity_scorer_agent(state: dict[str, Any]) -> dict[str, Any]:
    prompt_text = state.get("refined_prompt") or state.get("user_request", "")

    raw = run_agent(
        prompt=_SCORER_PROMPT.format(prompt=prompt_text),
        agent_name="complexity_scorer",
        prompt_key="complexity_scorer_v1",
        state=state,
        light=True,  # classification is cheap — always use light model
    )

    score, reason = _parse_score(raw)
    state["complexity_score"] = score
    state["complexity_reason"] = reason

    cfg = get_config()
    scorer_cfg = cfg.get("complexity_scorer", {})
    threshold = int(scorer_cfg.get("simple_threshold", 4))
    simple_model = str(scorer_cfg.get("simple_model", "llama-3.1-8b-instant"))

    if score <= threshold:
        state["complexity_model_override"] = simple_model
        print(
            f"\n  [ComplexityScorer] Score {score}/10 — SIMPLE. "
            f"Overriding heavy model with: {simple_model}"
        )
    else:
        state["complexity_model_override"] = None
        print(
            f"\n  [ComplexityScorer] Score {score}/10 — COMPLEX. "
            f"Using configured heavy model."
        )

    return state


def _parse_score(raw: str) -> tuple[int, str]:
    """Extract (score, reason) from LLM output. Defaults to (5, raw) on parse failure."""
    score = 5
    reason = raw.strip()

    score_match = re.search(r"SCORE:\s*(\d+)", raw, re.IGNORECASE)
    if score_match:
        score = max(1, min(10, int(score_match.group(1))))

    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    return score, reason
