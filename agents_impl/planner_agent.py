"""
agents_impl/planner_agent.py
────────────────────────────────────────────────────────────────────────────────
Planner agent — converts the user requirement into a functional specification.

Changes from original:
  - Removed module-level ChatGroq instantiation.
  - LLM call delegated to core.agent_runner.run_agent().
  - All prompt text preserved verbatim.
  - State mutations preserved verbatim.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent

_AGENT_NAME = "planner"
_PROMPT_KEY = "planner_v1"


def planner_agent(state: dict[str, Any]) -> dict[str, Any]:
    # Use the PromptRefiner's output when available; raw user_request otherwise.
    effective_request = state.get("refined_prompt") or state["user_request"]

    prompt = f"""You are a software business analyst following the BMAD methodology.
Convert the user request into a functional specification using EXACTLY this template — every section heading must appear verbatim.

User Request:
{effective_request}

OUTPUT TEMPLATE (reproduce every heading exactly):

## Functional Specification

### 1. Project Overview
[2-3 sentences describing what the app does and who it is for]

### 2. Core Functional Requirements
[Numbered list of 4-8 specific, testable requirements]
1.
2.
...

### 3. User Workflows
[Bullet list: each bullet is one end-to-end user action, e.g. "User fills form → system saves → table refreshes"]

### 4. Data Requirements
[What data must be stored, key entities and their important fields]

### 5. Business Rules & Constraints
[Numbered list of hard rules: validation limits, calculations, status transitions, etc.]
1.
2.
...

### 6. Out of Scope
[Bullet list of things explicitly NOT included — auth, external APIs, etc. unless requested]

RULES:
- Do NOT add authentication or login unless the user explicitly asked for it
- Do NOT add features beyond what the user requested
- Maximum 350 words total
- Be specific — name real fields, real ranges, real categories from the user's request
"""
    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key=_PROMPT_KEY,
        state=state,
        light=True,  # Functional spec — 8b handles this well, saves 70b quota
    )
    state["functional_spec"] = content
    state["review_attempts"] = 0
    return state
