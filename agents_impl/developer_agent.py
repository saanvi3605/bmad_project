"""
agents_impl/developer_agent.py
────────────────────────────────────────────────────────────────────────────────
Developer agent — generates production-quality Streamlit apps.

The prompt now includes concrete working code patterns so the LLM cannot
produce vague/incomplete implementations. Dual feedback-mode detection
is preserved exactly.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent
from core.sanitizer import sanitize_code

_AGENT_NAME = "developer"

PROMPT_TEMPLATE = """You are a senior Python developer. Generate a COMPLETE, fully runnable Streamlit app as a SINGLE Python file.

Technical Design:
{technical_design}

{feedback_section}

RULES (all mandatory):
- Return ONLY raw Python code — no markdown, no backticks, no explanations
- Streamlit ONLY — no FastAPI, Flask, uvicorn
- SQLite via sqlite3 stdlib ONLY — no SQLAlchemy, no ORM
- Single file — no fake section headers like "# models.py"
- st.set_page_config() must be the very first Streamlit call
- init_db() called once at module level; use CREATE TABLE IF NOT EXISTS
- DB reads: @st.cache_data(ttl=5); DB writes: commit then st.cache_data.clear() + st.rerun()
- All data entry via st.form() + st.form_submit_button() — never bare st.button()
- Validate every form input; show st.error() on bad data
- st.metric() for KPIs; st.dataframe(df, use_container_width=True, hide_index=True) for tables
- Sidebar navigation with emoji labels; plotly.express for all charts
- Every section in the technical design must be fully implemented — no TODOs, no stubs
- Never use generic placeholder labels like "Subject 1", "Field 1", "Item 1" — always use descriptive real-world labels from the technical design (e.g. "Mathematics", "Science", "English" for a grade manager; "Amount", "Category", "Description" for an expense tracker)
- For integer inputs (marks, counts, quantities, IDs) always use format="%d" and step=1 — NEVER format="%.2f"
- Only use format="%.2f" for currency or decimal inputs like prices, rates, and ratios
- SQLite table names must use snake_case — never spaces (e.g. menu_items not "Menu Items", order_details not "Order Details")
"""

UI_RULES = ""


def developer_agent(state: dict[str, Any]) -> dict[str, Any]:
    # ── Dual feedback-mode detection — preserved exactly ──────────────────
    feedback_section = ""
    if state.get("validation_error") and not state.get("validation_passed"):
        feedback_section = (
            "⚠️  VALIDATION FAILED — your previous code had this error:\n"
            + state["validation_error"]
            + "\n\nFix this specific error. Keep everything else intact."
        )
        prompt_key = "developer_validation_fix_v1"
    elif state.get("review_feedback") and not state.get("review_approved"):
        feedback_section = (
            "⚠️  CODE REVIEW FAILED — the reviewer rejected your code:\n"
            + state["review_feedback"]
            + "\n\nFix ALL issues listed above. Do not remove any working functionality."
        )
        prompt_key = "developer_review_fix_v1"
    else:
        prompt_key = "developer_clean_v1"

    prompt = PROMPT_TEMPLATE.format(
        technical_design=state["technical_design"],
        feedback_section=feedback_section,
    ) + UI_RULES

    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key=prompt_key,
        state=state,
    )
    state["code"] = sanitize_code(content)
    return state
