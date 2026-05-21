"""
agents_impl/test_writer_agent.py
────────────────────────────────────────────────────────────────────────────────
TestWriter agent — generates a pytest file for the generated Streamlit application.

Generated apps are Streamlit apps (single Python file, SQLite via sqlite3).
They have NO ASGI 'app' object, so the original starlette.testclient approach
was broken.  Tests instead use a sqlite3.Connection fixture that:
  1. Calls init_db() from the generated app to set up the schema.
  2. Yields a sqlite3.Connection for direct DB-level assertions.
  3. Cleans up the test DB file on teardown.

Changes from original:
  - FIXTURE_BLOCK replaced: starlette/TestClient removed, DB-level fixture added.
  - Prompt updated to instruct LLM to write def test_xxx(db) functions using
    sqlite3 inserts/queries instead of HTTP client calls.
  - All post-processing regex operations preserved (async stripping still useful
    as a safety net in case the LLM produces async tests).
  - Output path read from config instead of hardcoded "test_generated_app.py".
  - Optional session archiving of the test file.
  - State mutations preserved verbatim.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

from core.agent_runner import run_agent
from core.llm_factory import get_executor_config, get_session_config

_AGENT_NAME = "test_writer"
_PROMPT_KEY = "test_writer_v1"

# ---------------------------------------------------------------------------
# FIXTURE_BLOCK — preserved verbatim from original test_writer_agent.py
# ---------------------------------------------------------------------------

FIXTURE_BLOCK = (
    "import pytest\n"
    "import os\n"
    "import sys\n"
    "import sqlite3\n"
    "import gc\n"
    "\n"
    "# Ensure generated_app.py (in the same directory as this test file) is importable\n"
    "# regardless of which directory pytest is invoked from.\n"
    "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
    "from generated_app import init_db  # noqa: E402\n"
    "\n\n"
    "_TEST_DB = \"test_app.db\"\n"
    "\n\n"
    "@pytest.fixture\n"
    "def db():\n"
    "    \"\"\"Provides a sqlite3.Connection after a fresh init_db() call.\"\"\"\n"
    "    os.environ[\"DB_PATH\"] = _TEST_DB\n"
    "    # Remove any leftover DB from a previous test run\n"
    "    for _f in (_TEST_DB, \"app.db\"):\n"
    "        try:\n"
    "            os.remove(_f)\n"
    "        except FileNotFoundError:\n"
    "            pass\n"
    "    init_db()\n"
    "    conn = sqlite3.connect(_TEST_DB)\n"
    "    conn.row_factory = sqlite3.Row\n"
    "    yield conn\n"
    "    conn.close()\n"
    "    gc.collect()\n"
    "    for _f in (_TEST_DB, \"app.db\"):\n"
    "        try:\n"
    "            os.remove(_f)\n"
    "        except (PermissionError, FileNotFoundError):\n"
    "            pass\n"
    "\n\n"
)


# ---------------------------------------------------------------------------
# Post-processing helpers — each function is a direct extraction of one block
# from the original agent so they can be unit-tested individually.
# ---------------------------------------------------------------------------


def _strip_async_patterns(test_code: str) -> str:
    """Remove all async test patterns — preserved verbatim from original."""
    test_code = re.sub(r'@pytest\.mark\.anyio\s*\n', '', test_code)
    test_code = re.sub(r'@pytest\.mark\.asyncio\s*\n', '', test_code)
    test_code = test_code.replace("async def test_", "def test_")
    test_code = re.sub(r'await client\.', 'client.', test_code)
    test_code = re.sub(r'await ac\.', 'client.', test_code)
    test_code = test_code.replace("import anyio\n", "")
    test_code = test_code.replace("import asyncio\n", "")
    test_code = re.sub(r'async with.*?AsyncClient.*?as.*?:\s*\n', '', test_code)
    return test_code


def _fix_count_assertions(test_code: str) -> str:
    """Normalise exact count assertions — preserved verbatim from original."""
    def fix_count_assert(match: re.Match) -> str:
        n = int(match.group(1))
        return 'assert response.json() == []' if n == 0 else 'assert len(response.json()) >= 1'
    return re.sub(r'assert len\(response\.json\(\)\) == (\d+)', fix_count_assert, test_code)


def _extract_test_functions(test_code: str) -> str:
    """Strip everything above the first test_ function — preserved verbatim."""
    first_test = re.search(r'\ndef test_', test_code)
    if first_test:
        tests_only = test_code[first_test.start():]
    else:
        tests_only = test_code
    # Dedent any leftover indentation on test functions
    tests_only = re.sub(r'\n    def test_', '\ndef test_', tests_only)
    return tests_only


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------


def test_writer_agent(state: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "You are a senior QA engineer writing pytest tests for a Streamlit + SQLite app.\n\n"
        "Test Case Descriptions:\n"
        + (state.get("test_cases") or "")
        + "\n\nRULES:\n"
        "1. Write ONLY def test_xxx(db): functions — nothing else\n"
        "2. Do NOT write imports or fixtures — they are injected automatically\n"
        "3. 'db' is a sqlite3.Connection with row_factory=sqlite3.Row; the schema is already created\n"
        "4. Insert test data:  db.execute('INSERT INTO table(col) VALUES (?)', (value,)); db.commit()\n"
        "5. Query data:        rows = db.execute('SELECT * FROM table').fetchall()\n"
        "6. Assert presence:   assert len(rows) >= 1   (never use == N for counts)\n"
        "7. Assert empty:      assert rows == []\n"
        "8. Sync only — no async def, no await\n"
        "9. Return ONLY test functions, no markdown\n"
    )

    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key=_PROMPT_KEY,
        state=state,
        light=True,
    )

    test_code = content.replace("```python", "").replace("```", "").strip()

    # ── Post-processing — order preserved verbatim from original ──────────
    test_code = _strip_async_patterns(test_code)
    test_code = _fix_count_assertions(test_code)
    tests_only = _extract_test_functions(test_code)

    # ── Assemble final file — preserved verbatim from original ─────────────
    final_code = FIXTURE_BLOCK + tests_only.strip() + "\n"

    # ── Resolve output path from config ───────────────────────────────────
    exec_cfg = get_executor_config()
    test_output_path: str = exec_cfg.get("test_output_file", "test_generated_app.py")

    parent = os.path.dirname(test_output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(test_output_path, "w", encoding="utf-8") as f:
        f.write(final_code)

    # ── Optional session archiving ─────────────────────────────────────────
    session_cfg = get_session_config()
    if session_cfg.get("archive_outputs", False):
        session_id = state.get("session_id")
        output_dir = state.get("output_dir")
        if session_id and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(test_output_path, os.path.join(output_dir, "test_generated_app.py"))

    state["test_file"] = test_output_path
    print(f"\n  [TestWriter] DONE - pytest file saved to {test_output_path}")
    print("  [TestWriter] Run with: python -m pytest " + test_output_path + " -v")
    return state
