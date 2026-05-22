"""
agents_impl/readme_agent.py
────────────────────────────────────────────────────────────────────────────────
ReadmeWriter agent — final pipeline step.

Generates a production-quality README.md for the generated Streamlit app using
the artifacts already in state:

  functional_spec    — feature list and user stories  (from Planner)
  technical_design   — DB schema, component layout     (from Architect)
  test_file          — path to the generated pytest file (from TestWriter)
  execution_result   — success/crash message from Executor

Output:
  • Writes README.md next to the generated app (configurable via
    config/models.yaml executor.readme_output_file).
  • Copies it into the session archive directory if archiving is enabled.
  • Stores the path in state["readme_file"].

Uses the light (8b) model — README generation is summarisation of decisions
that have already been made; it does not require the deep comprehension of
the 70b model.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from core.agent_runner import run_agent
from core.llm_factory import get_executor_config, get_session_config

_AGENT_NAME = "readme_writer"
_PROMPT_KEY = "readme_writer_v1"

_PROMPT_TEMPLATE = """\
You are a technical writer. Write a complete, professional README.md for the \
Streamlit application described below.

────────────────────────────────────────────────────────────────
FUNCTIONAL SPECIFICATION
────────────────────────────────────────────────────────────────
{functional_spec}

────────────────────────────────────────────────────────────────
TECHNICAL DESIGN
────────────────────────────────────────────────────────────────
{technical_design}

────────────────────────────────────────────────────────────────
EXECUTION STATUS
────────────────────────────────────────────────────────────────
{execution_result}

────────────────────────────────────────────────────────────────
TEST FILE
────────────────────────────────────────────────────────────────
{test_file}

────────────────────────────────────────────────────────────────
REQUIRED README SECTIONS (include all of them, in this order)
────────────────────────────────────────────────────────────────
1. # <App Name>
   One-sentence tagline.

2. ## Overview
   2–4 sentences describing what the app does and who it is for.

3. ## Features
   Bullet list of every feature from the functional specification.

4. ## Tech Stack
   | Layer        | Technology              |
   |-------------|-------------------------|
   | UI           | Streamlit               |
   | Database     | SQLite (stdlib sqlite3) |
   | Language     | Python 3.10+            |
   | Charts       | Plotly Express          |
   | Tests        | pytest                  |

5. ## Setup
   Step-by-step instructions:
   a. Prerequisites (Python 3.10+, pip)
   b. Clone / download the project
   c. Install dependencies:
      ```bash
      pip install streamlit pandas plotly python-dotenv
      ```
   d. Create a `.env` file (mention DB_PATH if relevant):
      ```
      DB_PATH=app.db
      ```

6. ## Running the App
   ```bash
   streamlit run generated_app.py
   ```
   Mention the default browser URL (http://localhost:8501).

7. ## Running Tests
   ```bash
   pip install pytest
   python -m pytest {test_file} -v
   ```
   1–2 sentences describing what the tests cover.

8. ## Database Schema
   For every SQLite table in the technical design, show:
   ```sql
   CREATE TABLE table_name (
       column  TYPE  CONSTRAINTS,
       ...
   );
   ```

9. ## Project Structure
   ```
   generated_app.py          # Main Streamlit application
   {test_file}               # Pytest test suite
   README.md                 # This file
   app.db                    # SQLite database (auto-created on first run)
   ```

10. ## License
    MIT

────────────────────────────────────────────────────────────────
RULES
────────────────────────────────────────────────────────────────
- Return ONLY raw Markdown — no code fences around the whole document
- Use real names from the spec (never placeholders like "Feature 1")
- Keep each section concise but complete
- Code blocks must use triple backticks with a language hint
"""


def readme_agent(state: dict[str, Any]) -> dict[str, Any]:
    # ── Build prompt from state ────────────────────────────────────────────
    test_file_path = state.get("test_file") or "outputs/test_generated_app.py"
    # Show just the filename in the README (cleaner than an absolute path)
    test_file_name = os.path.basename(test_file_path)

    prompt = _PROMPT_TEMPLATE.format(
        functional_spec=state.get("functional_spec") or "(not available)",
        technical_design=state.get("technical_design") or "(not available)",
        execution_result=state.get("execution_result") or "(not available)",
        test_file=test_file_name,
    )

    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key=_PROMPT_KEY,
        state=state,
        light=True,
    )

    # Strip any accidental outer markdown fences the LLM may have added
    readme_text = content.strip()
    if readme_text.startswith("```"):
        readme_text = readme_text.lstrip("`").lstrip("markdown").lstrip("\n")
    if readme_text.endswith("```"):
        readme_text = readme_text[: readme_text.rfind("```")].rstrip()

    # ── Resolve output path from config ───────────────────────────────────
    exec_cfg = get_executor_config()
    readme_output_path: str = exec_cfg.get("readme_output_file", "outputs/README.md")

    parent = os.path.dirname(readme_output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(readme_output_path, "w", encoding="utf-8") as f:
        f.write(readme_text)

    # ── Optional session archiving ─────────────────────────────────────────
    session_cfg = get_session_config()
    if session_cfg.get("archive_outputs", False):
        output_dir = state.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(readme_output_path, os.path.join(output_dir, "README.md"))

    state["readme_file"] = readme_output_path
    print(f"\n  [ReadmeWriter] README saved to {readme_output_path}")
    return state
