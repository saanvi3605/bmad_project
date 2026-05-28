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

    # --- Check 12: st.form_submit_button must be at the top level of with st.form() ---
    # If it is nested inside an `if` block within the form, Streamlit warns
    # "This form has no submit button" whenever that condition is False.
    def _has_submit_call(node: ast.AST) -> bool:
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "form_submit_button"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "st"
            ):
                return True
        return False

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            is_form_ctx = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "form"
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "st"
                for item in node.items
            )
            if not is_form_ctx:
                continue
            # Walk the direct children of the with-block
            top_level_submit = any(
                not isinstance(stmt, ast.If) and _has_submit_call(stmt)
                for stmt in node.body
            )
            if_nested_submit = any(
                isinstance(stmt, ast.If) and _has_submit_call(stmt)
                for stmt in node.body
            )
            if if_nested_submit and not top_level_submit:
                return False, (
                    "st.form_submit_button() is only reachable inside an 'if' block within "
                    "st.form(). When that condition is False, Streamlit shows 'This form has "
                    "no submit button'. Fix: call st.form_submit_button() unconditionally at "
                    "the top level of the with st.form(): block, then check the returned bool "
                    "outside the form."
                )
    except Exception:
        pass  # AST errors already caught in Check 1

    return True, "All checks passed."


def validate_rag_code(code: str) -> tuple[bool, str]:
    """
    Validation checks for FastAPI RAG mode (main.py only).

    Replaces the Streamlit-specific checks with RAG-specific equivalents.
    """
    # Check 1: AST syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError on line {e.lineno}: {e.msg}\n  >> {e.text}"

    # Check 2: py_compile
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp.name],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            error = result.stderr.strip().replace(tmp.name, "main.py")
            return False, f"Compile error:\n{error}"
    except subprocess.TimeoutExpired:
        return False, "Compile check timed out."
    finally:
        os.unlink(tmp.name)

    # Check 3: FastAPI app present
    if "from fastapi import" not in code and "import fastapi" not in code:
        return False, "Missing FastAPI import — RAG service must use FastAPI."
    if "FastAPI()" not in code and "FastAPI(" not in code:
        return False, "Missing FastAPI() app instantiation."

    # Check 4: Required endpoints
    missing_endpoints = []
    for ep in ["/health", "/ingest", "/query", "/metrics"]:
        if ep not in code:
            missing_endpoints.append(ep)
    if missing_endpoints:
        return False, f"Missing required endpoints: {missing_endpoints}"

    # Check 5: ChromaDB usage
    if "chromadb" not in code:
        return False, "Missing ChromaDB — RAG service must use chromadb."

    # Check 6: Langfuse tracing
    if "langfuse" not in code.lower():
        return False, "Missing Langfuse — RAG service must include tracing."

    # Check 7: No Streamlit imports
    if "import streamlit" in code or "from streamlit" in code:
        return False, "Streamlit import found in RAG service — use FastAPI only."

    # Check 8: Pydantic models
    if "BaseModel" not in code:
        return False, "Missing Pydantic BaseModel — define request/response schemas."

    # Check 9: CORS middleware
    if "CORSMiddleware" not in code:
        return False, "Missing CORSMiddleware — RAG service must enable CORS."

    # Check 10: No stub implementations
    stub_patterns = ["pass  # TODO", "# TODO:", "raise NotImplementedError"]
    found_stubs = [p for p in stub_patterns if p in code]
    if found_stubs:
        return False, f"Incomplete implementation: {found_stubs[0]}. All functions must be fully implemented."

    return True, "All RAG checks passed."


def validator_agent(state: dict[str, Any]) -> dict[str, Any]:
    code = state.get("code", "")
    project_mode = state.get("project_mode", "streamlit_crud")

    if project_mode == "fastapi_rag":
        passed, message = validate_rag_code(code)
    else:
        passed, message = validate_code(code)

    state["validation_passed"] = passed
    state["validation_error"] = None if passed else message

    if passed:
        state["validation_attempts"] = 0
        print(f"\n  [Validator] PASSED ({project_mode}) - All checks passed.")
    else:
        state["validation_attempts"] = state.get("validation_attempts", 0) + 1
        print(f"\n  [Validator] FAILED ({project_mode}) - Sending back to Developer.\n  Reason: {message}")

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
