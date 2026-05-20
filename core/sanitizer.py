"""
core/sanitizer.py
────────────────────────────────────────────────────────────────────────────────
sanitize_code() for Streamlit-based generated apps.

Strips disallowed libraries (FastAPI, uvicorn, ORMs), enforces required
Streamlit imports, injects init_db() if missing, and removes uvicorn.run calls.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRIP_PREFIXES = (
    # Web frameworks we do NOT want in Streamlit apps
    "import uvicorn", "from uvicorn",
    "import fastapi", "from fastapi",
    "from starlette",
    "import flask", "from flask",
    "import django", "from django",
    # ORMs / non-SQLite databases
    "import sqlalchemy", "from sqlalchemy",
    "import databases", "from databases",
    "import asyncpg", "from asyncpg",
    "import psycopg2", "from psycopg2",
    "import pymysql", "from pymysql",
    "import motor", "from motor",
    "import tortoise", "from tortoise",
    # Template engines
    "import jinja2", "from jinja2",
    "from fastapi.templating",
    "from starlette.templating",
    # Async file I/O
    "import aiofiles", "from aiofiles",
)

REQUIRED_IMPORTS = [
    "import os",
    "import sqlite3",
    "import streamlit as st",
    "import pandas as pd",
    "from datetime import datetime",
    "from dotenv import load_dotenv",
    "load_dotenv()",
]

MINIMAL_INIT_DB = '''

def init_db():
    conn = sqlite3.connect(os.environ.get("DB_PATH", "app.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

init_db()
'''

STREAMLIT_MAIN_GUARD = '''

if __name__ == "__main__":
    pass  # Run with: streamlit run generated_app.py
'''


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def sanitize_code(code: str) -> str:
    # Strip markdown fences
    code = code.replace("```python", "").replace("```", "").strip()

    # Remove SQLAlchemy patterns
    code = re.sub(r"engine\s*=\s*create_engine\([^)]*\)\n?", "", code)
    code = re.sub(r"SessionLocal\s*=\s*sessionmaker\([^)]*\)\n?", "", code)
    code = re.sub(r"Base\s*=\s*declarative_base\(\)\n?", "", code)
    code = re.sub(r"Base\.metadata\.create_all\([^)]*\)\n?", "init_db()\n", code)
    code = re.sub(r",?\s*db:\s*Session\s*=\s*Depends\(get_db\)", "", code)
    code = re.sub(r"def get_db\(\).*?(?=\ndef |\n@)", "", code, flags=re.DOTALL)

    # Remove uvicorn.run calls entirely — not used in Streamlit apps
    code = re.sub(r"uvicorn\.run\(.*?\)", "", code, flags=re.DOTALL)

    # Strip disallowed imports line by line
    lines = code.split("\n")
    cleaned = [l for l in lines if not l.strip().startswith(STRIP_PREFIXES)]

    # Strip raw HTML tags outside strings
    quote_count = 0
    safe = []
    for line in cleaned:
        quote_count += line.count('"""') + line.count("'''")
        in_string = (quote_count % 2) == 1
        s = line.strip()
        if not in_string and s.startswith("<") and ">" in s:
            continue
        safe.append(line)

    # Prepend required imports
    result = "\n".join(REQUIRED_IMPORTS) + "\n\n" + "\n".join(safe).lstrip()

    # Inject init_db if missing
    if "def init_db" not in result:
        result = result.replace("load_dotenv()\n", "load_dotenv()\n" + MINIMAL_INIT_DB, 1)

    # Strip Jinja2 syntax
    result = re.sub(r"\{%.*?%\}", "", result)
    result = re.sub(r"\{\{.*?\}\}", "", result)

    # Fix SQLite table names that contain spaces — replaces spaces with
    # underscores in every SQL keyword context (FROM, INTO, UPDATE, JOIN…).
    # Python display strings like st.header("Menu Items") are untouched.
    result = _fix_table_names_with_spaces(result)

    # Fix format="%.2f" on integer number_inputs.
    # Matches st.number_input(...) calls where value= is a plain integer AND
    # format="%.2f" is present — swaps to format="%d" and step=1.
    result = _fix_integer_number_inputs(result)

    # Ensure __main__ guard exists
    if "__name__" not in result:
        result += STREAMLIT_MAIN_GUARD

    return result


def _fix_table_names_with_spaces(code: str) -> str:
    """Replace spaces with underscores in SQLite table names.

    Only rewrites occurrences that follow SQL keywords (TABLE, FROM, INTO,
    UPDATE, JOIN, REFERENCES) so Python display strings like
    st.header("Menu Items") are never touched.
    """
    create_re = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z][A-Za-z0-9_ ]+?)\s*\(',
        re.IGNORECASE,
    )
    name_map: dict[str, str] = {}
    for m in create_re.finditer(code):
        raw = m.group(1).strip()
        if ' ' in raw:
            name_map[raw] = raw.replace(' ', '_').lower()

    if not name_map:
        return code

    kw_prefix = (
        r'(?:TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        r'|DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?'
        r'|FROM\s+'
        r'|INTO\s+'
        r'|UPDATE\s+'
        r'|JOIN\s+'
        r'|REFERENCES\s+)'
    )
    for old, new in name_map.items():
        code = re.sub(
            rf'(?i)({kw_prefix}){re.escape(old)}\b',
            lambda m, _new=new: m.group(1) + _new,
            code,
        )

    return code


def _fix_integer_number_inputs(code: str) -> str:
    """Replace format="%.2f" with format="%d" and float step with step=1
    on st.number_input calls whose value= argument is an integer literal."""

    def _patch_call(m: re.Match) -> str:
        call = m.group(0)
        # Only patch if value= is a bare integer (no decimal point)
        if not re.search(r'\bvalue\s*=\s*\d+\b(?!\s*\.)', call):
            return call
        call = re.sub(r'format\s*=\s*["\']%\.2f["\']', 'format="%d"', call)
        # Replace float steps like step=0.01, step=0.1 with step=1
        call = re.sub(r'\bstep\s*=\s*0\.\d+', 'step=1', call)
        return call

    # Match st.number_input( ... ) across multiple lines, non-greedy
    return re.sub(
        r'st\.number_input\((?:[^()]*|\([^()]*\))*\)',
        _patch_call,
        code,
        flags=re.DOTALL,
    )
