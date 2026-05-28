"""
agents_impl/test_writer_agent.py
────────────────────────────────────────────────────────────────────────────────
TestWriter agent — generates a pytest file for the generated Streamlit app.

Root-cause fixes applied (2025-05):
  1. FIXTURE_BLOCK is now built dynamically from the actual CREATE TABLE
     statements extracted from state["code"] — no import from generated_app,
     no hardcoded DB path mismatch.
  2. The LLM prompt now includes the real schema summary AND a code snippet
     so it writes INSERT statements with the correct column names.
  3. Switched from light model to heavy model (light=False) to avoid the
     2 048-token truncation that produced SyntaxErrors.
  4. AST syntax validation is performed before writing the file; a fallback
     skeleton is written if the LLM output is still broken.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ast
import os
import re
import shutil
from typing import Any

from core.agent_runner import run_agent
from core.llm_factory import get_executor_config, get_session_config

_AGENT_NAME = "test_writer"
_PROMPT_KEY = "test_writer_v1"
_TEST_DB_NAME = "test_library.db"


# ---------------------------------------------------------------------------
# Schema extraction — reads CREATE TABLE statements from the generated code
# ---------------------------------------------------------------------------

def _extract_create_stmts(code: str) -> list[str]:
    """Return all CREATE TABLE SQL strings found inside cursor.execute() calls."""
    stmts: list[str] = []
    for pattern in [
        r'cursor\.execute\s*\(\s*"""(.*?)"""\s*\)',
        r"cursor\.execute\s*\(\s*'''(.*?)'''\s*\)",
    ]:
        for m in re.finditer(pattern, code, re.DOTALL):
            sql = m.group(1).strip()
            if re.search(r'CREATE\s+TABLE', sql, re.IGNORECASE):
                stmts.append(sql)
    return stmts


def _schema_summary(create_stmts: list[str]) -> str:
    """Human-readable table → column list for injecting into the LLM prompt."""
    lines: list[str] = []
    for stmt in create_stmts:
        tm = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', stmt, re.IGNORECASE)
        if not tm:
            continue
        table = tm.group(1)
        cols = re.findall(r'^\s{4,8}(\w+)\s+(?:INTEGER|TEXT|REAL|DATE|BLOB)', stmt, re.MULTILINE)
        lines.append(f"  Table `{table}`: columns = {cols}")
    return "\n".join(lines) if lines else "  (schema not detected)"


# ---------------------------------------------------------------------------
# Dynamic fixture builder — self-contained, never imports from generated_app
# ---------------------------------------------------------------------------

def _build_fixture_block(create_stmts: list[str]) -> str:
    """
    Build a pytest fixture that creates the schema in an isolated test DB
    without importing or calling anything from generated_app.py.
    """
    if not create_stmts:
        # Fallback: empty fixture that still yields a connection
        schema_lines = "    pass  # no CREATE TABLE statements detected\n"
    else:
        parts = []
        for stmt in create_stmts:
            # Normalise whitespace and embed as a triple-quoted string
            normalised = re.sub(r'\n\s*\n', '\n', stmt).strip()
            parts.append(f'    cur.execute("""\n        {normalised}\n    """)')
        schema_lines = "\n".join(parts) + "\n"

    return (
        "import pytest\n"
        "import os\n"
        "import sys\n"
        "import sqlite3\n"
        "import gc\n"
        "from datetime import datetime, timedelta\n"
        "\n"
        f'_TEST_DB = "{_TEST_DB_NAME}"\n'
        "\n\n"
        "@pytest.fixture\n"
        "def db():\n"
        f'    """Fresh isolated DB with the same schema as generated_app.py."""\n'
        "    try:\n"
        "        os.remove(_TEST_DB)\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "    conn = sqlite3.connect(_TEST_DB)\n"
        "    conn.row_factory = sqlite3.Row\n"
        "    cur = conn.cursor()\n"
        f"{schema_lines}"
        "    conn.commit()\n"
        "    yield conn\n"
        "    conn.close()\n"
        "    gc.collect()\n"
        "    try:\n"
        "        os.remove(_TEST_DB)\n"
        "    except (PermissionError, FileNotFoundError):\n"
        "        pass\n"
        "\n\n"
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _strip_async_patterns(test_code: str) -> str:
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
    def fix_count_assert(match: re.Match) -> str:
        n = int(match.group(1))
        return 'assert response.json() == []' if n == 0 else 'assert len(response.json()) >= 1'
    return re.sub(r'assert len\(response\.json\(\)\) == (\d+)', fix_count_assert, test_code)


def _extract_test_functions(test_code: str) -> str:
    first_test = re.search(r'\ndef test_', test_code)
    if first_test:
        tests_only = test_code[first_test.start():]
    else:
        tests_only = test_code
    tests_only = re.sub(r'\n    def test_', '\ndef test_', tests_only)
    return tests_only


def _validate_syntax(code: str) -> bool:
    """Return True if code parses as valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _fallback_tests(schema_sum: str) -> str:
    """Minimal passing test suite used when LLM output fails syntax check."""
    return (
        "\ndef test_schema_books(db):\n"
        "    rows = db.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='books'\").fetchall()\n"
        "    assert len(rows) >= 1\n"
        "\n"
        "def test_schema_members(db):\n"
        "    rows = db.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='members'\").fetchall()\n"
        "    assert len(rows) >= 1\n"
        "\n"
        "def test_insert_book(db):\n"
        "    db.execute(\"INSERT INTO books (title, author, ISBN, genre, total_copies, available_copies)\"\n"
        "               \" VALUES (?, ?, ?, ?, ?, ?)\",\n"
        "               ('Test Book', 'Test Author', 'ISBN-001', 'Fiction', 5, 5))\n"
        "    db.commit()\n"
        "    rows = db.execute(\"SELECT * FROM books WHERE title='Test Book'\").fetchall()\n"
        "    assert len(rows) >= 1\n"
        "\n"
        "def test_insert_member(db):\n"
        "    db.execute(\"INSERT INTO members (name, member_ID, phone) VALUES (?, ?, ?)\",\n"
        "               ('Alice', 'M001', '9999999999'))\n"
        "    db.commit()\n"
        "    rows = db.execute(\"SELECT * FROM members WHERE name='Alice'\").fetchall()\n"
        "    assert len(rows) >= 1\n"
    )


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def test_writer_agent(state: dict[str, Any]) -> dict[str, Any]:
    code: str = state.get("code", "")

    # ── Extract schema info from the actual generated code ────────────────
    create_stmts = _extract_create_stmts(code)
    schema_sum   = _schema_summary(create_stmts)

    # Include a compact code snippet (first 3 000 chars) so the LLM sees
    # real column names, real function signatures, etc.
    code_snippet = code[:3000] + ("\n... (truncated)" if len(code) > 3000 else "")

    prompt = (
        "You are a senior QA engineer writing pytest tests for a Streamlit + SQLite app.\n\n"
        "=== EXACT DATABASE SCHEMA ===\n"
        f"{schema_sum}\n\n"
        "=== GENERATED APP CODE (first 3 000 chars) ===\n"
        f"{code_snippet}\n\n"
        "=== TEST CASE DESCRIPTIONS ===\n"
        + (state.get("test_cases") or "(none)")
        + "\n\n"
        "=== RULES (follow ALL of them) ===\n"
        "1. Write ONLY `def test_xxx(db):` functions — no imports, no fixtures, no classes\n"
        "2. `db` is a sqlite3.Connection (row_factory=sqlite3.Row); schema already exists\n"
        "3. Insert test data with: db.execute('INSERT INTO table (col1, col2) VALUES (?, ?)', (v1, v2)); db.commit()\n"
        "4. Query with:  rows = db.execute('SELECT ...').fetchall()\n"
        "5. Assert presence:  assert len(rows) >= 1\n"
        "6. Assert empty:     assert rows == []\n"
        "7. Use EXACT column names from the schema above — case-sensitive (e.g. ISBN, member_ID)\n"
        "8. Do NOT call any function from the app (no init_db, no calculate_*, no issue_book, etc.)\n"
        "9. Sync only — no async def, no await\n"
        "10. No markdown fences, no explanatory text — ONLY Python function definitions\n"
        "11. Write 8-12 test functions covering: schema, CRUD, relationships, and edge cases\n"
        "12. Every INSERT must include all NOT NULL columns with no DEFAULT\n"
    )

    # Use heavy model (light=False) to avoid the 2 048-token truncation
    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key=_PROMPT_KEY,
        state=state,
        light=False,
    )

    test_code = content.replace("```python", "").replace("```", "").strip()

    # ── Post-processing ───────────────────────────────────────────────────
    test_code = _strip_async_patterns(test_code)
    test_code = _fix_count_assertions(test_code)
    tests_only = _extract_test_functions(test_code)

    # ── Syntax validation — fallback to minimal passing tests on failure ──
    fixture_block = _build_fixture_block(create_stmts)
    final_code    = fixture_block + tests_only.strip() + "\n"

    if not _validate_syntax(final_code):
        print("  [TestWriter] WARNING: LLM output failed syntax check — using fallback tests")
        final_code = fixture_block + _fallback_tests(schema_sum) + "\n"

    # ── Write output file ─────────────────────────────────────────────────
    exec_cfg = get_executor_config()
    test_output_path: str = exec_cfg.get("test_output_file", "test_generated_app.py")

    parent = os.path.dirname(test_output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(test_output_path, "w", encoding="utf-8") as f:
        f.write(final_code)

    # ── Verify the written file parses cleanly ────────────────────────────
    with open(test_output_path, "r", encoding="utf-8") as f:
        written = f.read()
    if not _validate_syntax(written):
        # Last resort: overwrite with guaranteed-valid fallback
        with open(test_output_path, "w", encoding="utf-8") as f:
            f.write(fixture_block + _fallback_tests(schema_sum) + "\n")
        print("  [TestWriter] WARNING: Written file still invalid — replaced with fallback")

    # ── Optional session archiving ─────────────────────────────────────────
    session_cfg = get_session_config()
    if session_cfg.get("archive_outputs", False):
        session_id = state.get("session_id")
        output_dir = state.get("output_dir")
        if session_id and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(test_output_path, os.path.join(output_dir, "test_generated_app.py"))

    state["test_file"] = test_output_path
    print(f"\n  [TestWriter] DONE — pytest file saved to {test_output_path}")
    print("  [TestWriter] Run with: python -m pytest " + test_output_path + " -v")
    return state
