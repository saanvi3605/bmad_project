"""
agents_impl/validator_agent.py
────────────────────────────────────────────────────────────────────────────────
Validator agent — pure-Python static analysis, no LLM.

Extended with Streamlit-specific quality checks on top of the original
six checks. All original logic is preserved exactly.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

REQUIRED_PATTERNS = [
    ("import streamlit as st",  "Missing: import streamlit as st"),
    ("st.set_page_config(",     "Missing st.set_page_config() — must be first Streamlit call"),
    ("def init_db(",            "Missing init_db() function for database setup"),
    ("sqlite3.connect(",        "Missing SQLite usage — app must use sqlite3"),
    ("st.form(",                "Missing st.form() — all data entry must use st.form()"),
    ("st.form_submit_button(",  "Missing st.form_submit_button() — required inside every st.form()"),
    ("st.cache_data",           "Missing @st.cache_data — DB read functions must be cached"),
]

DISALLOWED = [
    "sqlalchemy", "create_engine", "sessionmaker", "declarative_base",
    "from databases", "import asyncpg", "import psycopg2",
    "import fastapi", "from fastapi", "import uvicorn", "from uvicorn",
    "Jinja2Templates", "from jinja2",
]


def validate_code(code: str) -> tuple[bool, str]:
    # --- Check 1: AST syntax ---
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError on line {e.lineno}: {e.msg}\n  >> {e.text}"

    # --- Check 2: py_compile ---
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp.name],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            error = result.stderr.strip().replace(tmp.name, "generated_app.py")
            return False, f"Compile error:\n{error}"
    except subprocess.TimeoutExpired:
        return False, "Compile check timed out."
    finally:
        os.unlink(tmp.name)

    # --- Check 3: Required Streamlit patterns ---
    missing = [msg for pattern, msg in REQUIRED_PATTERNS if pattern not in code]
    if missing:
        return False, "Missing required elements:\n- " + "\n- ".join(missing)

    # --- Check 4: Disallowed libraries ---
    found = [d for d in DISALLOWED if d in code]
    if found:
        return False, f"Disallowed libraries present: {', '.join(found)}. Use Streamlit + sqlite3 only."

    # --- Check 5: NameError detection via AST ---
    try:
        tree = ast.parse(code)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)

        KNOWN_BUILTINS = {"int", "str", "float", "bool", "list", "dict", "set",
                          "None", "True", "False", "Optional", "List", "Dict", "Any",
                          "tuple", "Tuple", "Union", "datetime"}
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.annotation and isinstance(arg.annotation, ast.Name):
                        name = arg.annotation.id
                        if name not in defined and name not in KNOWN_BUILTINS:
                            errors.append(f"NameError: '{name}' used as type annotation but not imported")
        if errors:
            return False, "\n".join(errors)
    except Exception:
        pass

    # --- Check 6: Jinja2 syntax ---
    jinja_patterns = ["{%", "%}", "{{", "}}"]
    html_section = ""
    in_html = False
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("HTML") and "=" in stripped:
            in_html = True
        if in_html:
            html_section += line + "\n"
    for pat in jinja_patterns:
        if pat in html_section:
            return False, (
                f"Jinja2 syntax '{pat}' found. Do not use Jinja2 — use Streamlit widgets only."
            )

    # --- Check 7: st.set_page_config is first Streamlit call ---
    streamlit_calls = [line.strip() for line in code.split("\n")
                       if line.strip().startswith("st.") and not line.strip().startswith("st.cache")]
    if streamlit_calls and not streamlit_calls[0].startswith("st.set_page_config"):
        return False, (
            f"st.set_page_config() must be the FIRST Streamlit call. "
            f"Found '{streamlit_calls[0]}' before it."
        )

    # --- Check 8: init_db() is called at module level ---
    if "def init_db(" in code and "init_db()" not in code:
        return False, "init_db() is defined but never called at module level. Add init_db() after the function definition."

    # --- Check 9: No stub/placeholder implementations ---
    stub_patterns = ["pass  # TODO", "# TODO:", "raise NotImplementedError", "pass  # implement"]
    found_stubs = [p for p in stub_patterns if p in code]
    if found_stubs:
        return False, f"Incomplete implementation found: {found_stubs[0]}. All functions must be fully implemented."

    # --- Check 10: st.cache_data.clear() called after writes ---
    has_write = any(kw in code for kw in ["INSERT INTO", "UPDATE ", "DELETE FROM"])
    has_clear = "st.cache_data.clear()" in code
    if has_write and not has_clear:
        return False, "Database writes found but st.cache_data.clear() is never called. Add it after every INSERT/UPDATE/DELETE."

    # --- Check 11: No spaces in SQLite table names ---
    create_tbl_re = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z][A-Za-z0-9_ ]+?)\s*\(',
        re.IGNORECASE,
    )
    for m in create_tbl_re.finditer(code):
        name = m.group(1).strip()
        if ' ' in name:
            suggested = name.replace(' ', '_').lower()
            return False, (
                f"SQLite table name '{name}' contains spaces — invalid SQL syntax. "
                f"Use underscores instead: '{suggested}'."
            )

    return True, "All checks passed."


def validator_agent(state: dict[str, Any]) -> dict[str, Any]:
    code = state.get("code", "")
    passed, message = validate_code(code)
    state["validation_passed"] = passed
    state["validation_error"] = None if passed else message

    if passed:
        # Reset counter so Reviewer-retry cycles each get a fresh budget.
        state["validation_attempts"] = 0
        print("\n  [Validator] PASSED - All checks passed.")
    else:
        state["validation_attempts"] = state.get("validation_attempts", 0) + 1
        print(f"\n  [Validator] FAILED - Sending back to Developer.\n  Reason: {message}")

    return state


def should_fix(state: dict[str, Any]) -> str:
    if state.get("validation_passed"):
        return "passed"
    # validation_attempts is incremented by validator_agent (a node), so the
    # value is reliably persisted.  Threshold > 2 allows exactly 2 retries.
    if state.get("validation_attempts", 0) > 2:
        print("\n  [Validator] Max fix attempts reached — proceeding anyway.")
        return "max_attempts"
    return "fix"
