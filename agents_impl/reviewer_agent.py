"""
agents_impl/reviewer_agent.py
────────────────────────────────────────────────────────────────────────────────
Reviewer agent — LLM-based quality gate with a high bar for Streamlit apps.

should_retry() routing function is preserved exactly.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent
from core.llm_factory import get_config

_AGENT_NAME = "reviewer"


def reviewer_agent(state: dict[str, Any]) -> dict[str, Any]:
    if not get_config().get("reviewer", {}).get("enabled", True):
        print("\n  [Reviewer] Disabled in config — auto-approving.")
        state["review_approved"] = True
        state["review_feedback"] = "Reviewer disabled."
        state["review_attempts"] = state.get("review_attempts", 0) + 1
        return state

    prompt = f"""You are a strict senior code reviewer for Streamlit applications.
Review the generated code below and produce a review report using EXACTLY this template.

Generated Code:
{state['code']}

OUTPUT TEMPLATE (reproduce every heading exactly):

## Code Review Report

### 1. Review Summary
APPROVED: YES or NO
[One sentence overall verdict]

### 2. Criteria Checklist
| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | SYNTAX: Valid executable Python, no SyntaxErrors | PASS/FAIL | |
| 2 | FRAMEWORK: Streamlit only — no FastAPI, Flask, uvicorn | PASS/FAIL | |
| 3 | DATABASE: sqlite3 stdlib only — no SQLAlchemy, no ORM | PASS/FAIL | |
| 4 | PAGE CONFIG: st.set_page_config() is the very first Streamlit call | PASS/FAIL | |
| 5 | FORMS: All data entry uses st.form() + st.form_submit_button() | PASS/FAIL | |
| 6 | VALIDATION: Every form validates inputs and shows st.error() on bad data | PASS/FAIL | |
| 7 | CACHE: DB reads have @st.cache_data(ttl=5); st.cache_data.clear() after every write | PASS/FAIL | |
| 8 | COMPLETENESS: No stub functions, no TODOs, no placeholder comments | PASS/FAIL | |
| 9 | SINGLE FILE: One self-contained .py file | PASS/FAIL | |

### 3. Issues Found
[If APPROVED: YES — write "None". If NO — list each issue:]
- Criterion <N> FAIL: <specific line number or location> — <exact fix required>

### 4. Recommended Fixes
[If APPROVED: YES — write "None". If NO — bullet list of exact code changes needed]

RULES:
- The first line of your response after the template MUST be "APPROVED: YES" or "APPROVED: NO"
- Be specific: cite line numbers or function names, not vague descriptions
"""

    state["review_attempts"] = state.get("review_attempts", 0) + 1
    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key="reviewer_v1",
        state=state,
        light=False,  # Reviewer reads full code — needs 70b comprehension depth to catch subtle bugs
    )

    upper = content.upper()
    state["review_approved"] = "APPROVED: YES" in upper

    # Extract issues from the templated report — fall back to full content
    issues_marker = "### 3. Issues Found"
    issues_start = content.find(issues_marker)
    if issues_start != -1:
        state["review_feedback"] = content[issues_start:].strip()
    else:
        state["review_feedback"] = content.strip()

    status = "APPROVED" if state["review_approved"] else "REJECTED"
    print(f"\n  [Reviewer] {status} (attempt {state['review_attempts']})")
    if not state["review_approved"]:
        print(f"  Feedback: {state['review_feedback'][:200]}...")

    return state


def should_retry(state: dict[str, Any]) -> str:
    if state.get("review_approved"):
        return "approved"
    attempts = state.get("review_attempts", 0)
    if attempts >= 2:
        print("\n  [Reviewer] Max retry attempts reached — proceeding to Executor.")
        return "max_retries"
    return "retry"
