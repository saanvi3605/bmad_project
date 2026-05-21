"""
agents_impl/prompt_refiner_agent.py
────────────────────────────────────────────────────────────────────────────────
PromptRefiner agent — first node in the pipeline.

Rewrites the raw user prompt into a structured, detailed product brief before
it reaches the Planner.  Handles vague inputs ("make me an app") by expanding
them into a sensible, concrete specification.  Detailed inputs are enriched
with any obvious missing requirements without changing intent.

Output is stored in state["refined_prompt"].  The Planner reads this field
(falling back to user_request if absent for backward compatibility).
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent

_AGENT_NAME = "prompt_refiner"
_PROMPT_KEY = "prompt_refiner_v1"

_PROMPT_TEMPLATE = """\
You are a senior product analyst. A user wants to build a Streamlit application backed by SQLite.

Raw user request:
{raw}

TASK: Rewrite the request into a clear, unambiguous product brief that eliminates guesswork for the development team.

RULES:
- If the request is vague or minimal (e.g. "make me an app", "build something cool"), invent a \
sensible, complete application with realistic features appropriate to any implied domain
- If the request is already detailed, add obvious missing requirements (e.g. delete/edit actions, \
validation rules, status fields) without changing the user's intent
- Never add: authentication/login, external APIs, payment processing, email — unless explicitly requested
- All features must be achievable in a single Streamlit + SQLite file
- Use concrete, real-world names — never "Item 1", "Field A", "Category X"

OUTPUT — write ONLY the product brief below, no preamble, no explanation:

## Application: <a specific descriptive name>

### Purpose
<Who uses this app and what problem it solves — 2 sentences maximum>

### Features
<Numbered list of 5-8 specific, concrete features a user can perform>
1.
2.
3.

### Data to Track
<Each data entity with its key fields — be specific: name every field and its type/range>
- <Entity>: <field1 (type/range)>, <field2 (type/range)>, ...

### User Workflows
<Step-by-step description of the 2-3 most important end-to-end user actions>
1.
2.

### Business Rules
<Hard constraints: validation ranges, required fields, calculated values, status transitions>
-
-

Max 300 words. Every field, range, and rule must be concrete enough for a developer to implement \
immediately without asking follow-up questions.\
"""


def prompt_refiner_agent(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("user_request", "").strip()

    content = run_agent(
        prompt=_PROMPT_TEMPLATE.format(raw=raw),
        agent_name=_AGENT_NAME,
        prompt_key=_PROMPT_KEY,
        state=state,
        light=True,  # Rewriting/expansion — 8b is sufficient, saves 70b quota
    )

    state["refined_prompt"] = content.strip()
    print(f"\n  [PromptRefiner] Done — refined to {len(state['refined_prompt'])} chars")
    return state
