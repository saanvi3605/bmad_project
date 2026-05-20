"""
agents_impl/architect_agent.py
────────────────────────────────────────────────────────────────────────────────
Architect agent — produces detailed technical designs for Streamlit apps.
The more specific the design, the better the Developer's output.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent

_AGENT_NAME = "architect"


def architect_agent(state: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""You are a software architect. Produce a technical design for a Streamlit app using EXACTLY this template — every section heading must appear verbatim.

Functional Specification:
{state['functional_spec']}

OUTPUT TEMPLATE (reproduce every heading exactly):

## Technical Design — Solution Blueprint

### 1. Tech Stack
- UI: Streamlit (st.form, st.tabs, st.columns, st.metric, st.dataframe, st.expander)
- Database: SQLite via sqlite3 stdlib
- Charts: plotly.express | Data: pandas | Other: python-dotenv, datetime

### 2. Database Schema
[Every table with columns, types, constraints — be exact]
Table: <name>
  - id: INTEGER PRIMARY KEY AUTOINCREMENT
  - <col>: <TYPE> [NOT NULL / DEFAULT ...]
  ...

### 3. Sidebar Navigation
[Every page with exact emoji label and one-line purpose]
- "emoji Page Name" — purpose

### 4. Forms & Input Fields
[Every form — key, fields, validation, submit label]
Form: "<form_key>"
  - <Field>: <widget_type>, <options/range/default>, <validation rule>
  Submit: "<button label>"

### 5. Business Logic & Calculations
[Every formula, lookup table, rule, or status transition — be precise]
- <Rule>: <exact formula or condition>

### 6. Results Display
[What appears after each form submission]
- st.metric(): <label> = <value expression>
- st.dataframe(): columns [<col1>, <col2>, ...]
- Chart: <type>, x=<col>, y=<col>, title="<title>"
- st.success/error: <condition>

### 7. Analytics Dashboard
[Every chart and KPI on the analytics page]
- <Chart type>: x=<col>, y=<col>, title="<title>"
- st.metric(): <label>

RULES:
- Streamlit ONLY — no FastAPI, no Flask | SQLite + sqlite3 ONLY — no ORM
- No auth unless explicitly in the spec
- Be SPECIFIC — name real fields, real formulas, real labels from the spec
- Maximum 350 words
"""

    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key="architect_v1",
        state=state,
    )
    state["technical_design"] = content
    return state
