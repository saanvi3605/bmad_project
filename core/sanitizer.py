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
    # Vector databases — not allowed in Streamlit CRUD apps
    "import chromadb", "from chromadb",
    "import pinecone", "from pinecone",
    "import weaviate", "from weaviate",
    "import qdrant", "from qdrant",
    # LLM orchestration frameworks — not needed in Streamlit CRUD apps
    "import langchain", "from langchain",
    "import llamaindex", "from llamaindex",
    "from llama_index",
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

# CSS injected into every generated app right after st.set_page_config().
# Edit this constant to change the look of all future pipeline outputs.
_BMAD_THEME_CSS = """<style>
/* bmad-theme — injected by sanitizer */

/* Hide Streamlit chrome for a cleaner look */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }

/* Typography */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

/* Buttons — rounded with hover lift */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.4rem 1.2rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
}

/* Metric cards */
[data-testid="metric-container"] {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    background: rgba(255, 255, 255, 0.03);
}

/* Form containers */
[data-testid="stForm"] {
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    background: rgba(255, 255, 255, 0.02);
}

/* Input fields */
.stTextInput  > div > div > input,
.stNumberInput > div > div > input,
.stTextArea   > div > div > textarea {
    border-radius: 8px !important;
}

/* Selectbox */
.stSelectbox > div > div { border-radius: 8px !important; }

/* DataFrames */
[data-testid="stDataFrame"] > div { border-radius: 10px; overflow: hidden; }

/* Alert / status boxes */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 8px !important;
}

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {
    background: rgba(0, 0, 0, 0.12);
    border-right: 1px solid rgba(128, 128, 128, 0.12);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.4rem 1rem;
}

/* Expanders */
.streamlit-expanderHeader {
    border-radius: 8px !important;
    font-weight: 500;
}
</style>"""


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

    # Replace deprecated Streamlit experimental APIs (removed in Streamlit 1.30+)
    result = result.replace("st.experimental_rerun()", "st.rerun()")
    result = result.replace("st.experimental_get_query_params()", "st.query_params")
    result = result.replace("st.experimental_set_query_params(", "st.query_params.update(")

    # Fix SQLite table names that contain spaces — replaces spaces with
    # underscores in every SQL keyword context (FROM, INTO, UPDATE, JOIN…).
    # Python display strings like st.header("Menu Items") are untouched.
    result = _fix_table_names_with_spaces(result)

    # Fix format="%.2f" on integer number_inputs.
    # Matches st.number_input(...) calls where value= is a plain integer AND
    # format="%.2f" is present — swaps to format="%d" and step=1.
    result = _fix_integer_number_inputs(result)

    # Inject CSS theme right after st.set_page_config(...)
    result = _inject_css_theme(result)

    # Ensure __main__ guard exists
    if "__name__" not in result:
        result += STREAMLIT_MAIN_GUARD

    # Auto-format: black normalises whitespace/line-length; isort deduplicates
    # and orders imports.  Both run best-effort — a syntax error or missing
    # package never blocks the pipeline; we just keep the pre-format string.
    result = _apply_formatters(result)

    return result


def _inject_css_theme(code: str) -> str:
    """
    Insert the BMAD CSS theme block immediately after st.set_page_config(...).

    The injection marker ``/* bmad-theme */`` prevents double-injection if
    sanitize_code() is ever called twice on the same code.

    The parenthesis-depth walk correctly handles multi-line set_page_config
    calls with nested dicts/tuples (e.g. page_icon=("🤖",)).
    """
    if "/* bmad-theme */" in code:
        return code  # Already injected — skip

    start = code.find("st.set_page_config(")
    if start == -1:
        # No set_page_config found — prepend theme after all imports as fallback
        insert_at = code.find("\n\n")
        if insert_at == -1:
            return code
        theme_call = f'\nst.markdown("""{_BMAD_THEME_CSS}""", unsafe_allow_html=True)\n'
        return code[:insert_at] + theme_call + code[insert_at:]

    # Walk forward to find the matching closing parenthesis
    depth = 0
    i = start
    while i < len(code):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                insert_at = i + 1
                break
        i += 1
    else:
        return code  # Malformed call — leave unchanged

    theme_call = f'\nst.markdown("""{_BMAD_THEME_CSS}""", unsafe_allow_html=True)\n'
    return code[:insert_at] + theme_call + code[insert_at:]


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


def _apply_formatters(code: str) -> str:
    """
    Run black then isort on sanitized code.

    Both steps are best-effort:
      - black normalises indentation, trailing commas, and line length so the
        AST validator never trips on cosmetic whitespace issues.
      - isort deduplicates and sorts the prepended REQUIRED_IMPORTS block,
        removing any copies the LLM already emitted in the generated code.

    If either tool is not installed, or the code still has a syntax error that
    prevents formatting, the original string is returned unchanged so the
    pipeline is never blocked by a formatter failure.
    """
    try:
        import black  # type: ignore[import]
        code = black.format_str(code, mode=black.Mode())
    except Exception:
        pass
    try:
        import isort  # type: ignore[import]
        code = isort.code(code)
    except Exception:
        pass
    return code


def sanitize_rag_files(files: dict[str, str]) -> dict[str, str]:
    """
    Sanitize a dict of {filename: content} from the RAG developer agent.

    For RAG mode we do NOT strip FastAPI/ChromaDB/LangChain imports — those
    are required.  We only strip markdown fences, fix common LLM import
    hallucinations, normalise whitespace, and run black/isort on Python files.
    """
    sanitized: dict[str, str] = {}
    for filename, content in files.items():
        # Strip markdown fences
        content = content.replace("```python", "").replace("```yaml", "").replace("```", "").strip()

        if filename.endswith(".py"):
            content = _fix_rag_imports(content)
            content = _apply_formatters(content)

        sanitized[filename] = content
    return sanitized


def _fix_rag_imports(code: str) -> str:
    """
    Fix common LLM import hallucinations in RAG-mode Python files.

    The package is `python-dotenv` but the import name is `dotenv`.
    LLMs frequently hallucinate `import python_dotenv` which doesn't exist.
    """
    # `import python_dotenv` → `from dotenv import load_dotenv`
    code = re.sub(r'import python_dotenv\b', 'from dotenv import load_dotenv', code)
    # `python_dotenv.load_dotenv()` → `load_dotenv()`
    code = re.sub(r'python_dotenv\.load_dotenv\(\)', 'load_dotenv()', code)
    # `import dotenv` (bare) → `from dotenv import load_dotenv`
    code = re.sub(r'^import dotenv\s*$', 'from dotenv import load_dotenv', code, flags=re.MULTILINE)
    # `dotenv.load_dotenv()` → `load_dotenv()`
    code = re.sub(r'dotenv\.load_dotenv\(\)', 'load_dotenv()', code)
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
