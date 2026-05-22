"""
tests/test_validator.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for agents_impl.validator_agent.validate_code() and should_fix().

VALID_CODE is a minimal Streamlit + SQLite app that passes every check.
Tests that need a broken variant mutate VALID_CODE locally; the constant
itself is never modified.
────────────────────────────────────────────────────────────────────────────────
"""

import pytest
from agents_impl.validator_agent import validate_code, should_fix, validator_agent

# ---------------------------------------------------------------------------
# Minimal Streamlit app that must pass all 11 validator checks
# ---------------------------------------------------------------------------
VALID_CODE = '''\
import os
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Test App", layout="wide")


def init_db():
    conn = sqlite3.connect(os.environ.get("DB_PATH", "app.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS items "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()


init_db()


@st.cache_data(ttl=5)
def load_items():
    conn = sqlite3.connect(os.environ.get("DB_PATH", "app.db"))
    rows = conn.execute("SELECT * FROM items").fetchall()
    conn.close()
    return rows


with st.form("add_item"):
    name = st.text_input("Item name")
    submitted = st.form_submit_button("Add")
    if submitted and name:
        conn = sqlite3.connect(os.environ.get("DB_PATH", "app.db"))
        conn.execute(
            "INSERT INTO items (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        st.cache_data.clear()
        st.rerun()

items = load_items()
if items:
    df = pd.DataFrame(items, columns=["id", "name", "created_at"])
    st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    pass
'''


class TestValidateCodePassing:
    def test_valid_code_passes(self):
        passed, msg = validate_code(VALID_CODE)
        assert passed is True, f"Expected pass but got: {msg}"
        assert msg == "All checks passed."


class TestCheck1SyntaxParse:
    def test_syntax_error_detected(self):
        bad_code = "def broken(\n    x\n"
        passed, msg = validate_code(bad_code)
        assert passed is False
        assert "SyntaxError" in msg

    def test_valid_syntax_passes(self):
        passed, _ = validate_code(VALID_CODE)
        assert passed is True


class TestCheck3RequiredPatterns:
    """Each test removes one required Streamlit pattern and checks the error."""

    def test_missing_streamlit_import(self):
        code = VALID_CODE.replace("import streamlit as st\n", "")
        passed, msg = validate_code(code)
        assert passed is False
        assert "streamlit" in msg.lower()

    def test_missing_set_page_config(self):
        code = VALID_CODE.replace("st.set_page_config(", "# st.set_page_config(")
        passed, msg = validate_code(code)
        assert passed is False
        assert "set_page_config" in msg

    def test_missing_init_db(self):
        # Rename the function so the pattern `def init_db(` is absent
        code = VALID_CODE.replace("def init_db(", "def setup_db(")
        passed, msg = validate_code(code)
        assert passed is False
        assert "init_db" in msg

    def test_missing_form(self):
        # Replace st.form( with st.container( so the pattern is absent
        code = VALID_CODE.replace("st.form(", "st.container(")
        passed, msg = validate_code(code)
        assert passed is False
        assert "form" in msg.lower()


class TestCheck4DisallowedLibraries:
    def test_sqlalchemy_detected(self):
        code = VALID_CODE + "\nimport sqlalchemy\n"
        passed, msg = validate_code(code)
        assert passed is False
        assert "sqlalchemy" in msg.lower()

    def test_create_engine_detected(self):
        code = VALID_CODE + "\nengine = create_engine('sqlite:///app.db')\n"
        passed, msg = validate_code(code)
        assert passed is False
        assert "create_engine" in msg


class TestCheck6Jinja2Detection:
    """Jinja2 syntax inside an HTML= line must be detected."""

    def test_jinja2_block_tag_detected(self):
        # Append a line that looks like an HTML template variable
        code = VALID_CODE + '\nHTML = "<html>{% for item in items %}</html>"\n'
        passed, msg = validate_code(code)
        assert passed is False
        assert "Jinja2" in msg

    def test_jinja2_variable_tag_detected(self):
        code = VALID_CODE + '\nHTML = "<html>{{ item.name }}</html>"\n'
        passed, msg = validate_code(code)
        assert passed is False
        assert "Jinja2" in msg


class TestShouldFix:
    """
    should_fix() is a pure routing function — it reads validation_attempts but
    does NOT mutate it.  Incrementing is the responsibility of validator_agent().
    """

    def test_passed_returns_passed(self):
        state = {"validation_passed": True, "validation_attempts": 0}
        result = should_fix(state)
        assert result == "passed"
        assert state["validation_attempts"] == 0  # should_fix never mutates

    def test_first_failure_returns_fix(self):
        # validator_agent increments before should_fix is called; simulate attempts=1
        state = {"validation_passed": False, "validation_attempts": 1}
        result = should_fix(state)
        assert result == "fix"

    def test_second_failure_returns_fix(self):
        state = {"validation_passed": False, "validation_attempts": 2}
        result = should_fix(state)
        assert result == "fix"

    def test_max_attempts_returns_max_attempts(self):
        # Threshold is > 2; needs attempts >= 3 to trigger max_attempts
        state = {"validation_passed": False, "validation_attempts": 3}
        result = should_fix(state)
        assert result == "max_attempts"


class TestValidatorAgent:
    def test_agent_sets_passed_on_valid_code(self):
        state = {
            "code": VALID_CODE,
            "agent_log": [],
            "session_id": "test",
            "langfuse_handler": None,
            "validation_attempts": 0,
        }
        result = validator_agent(state)
        assert result["validation_passed"] is True
        assert result["validation_error"] is None

    def test_agent_sets_failed_on_bad_code(self):
        state = {
            "code": "def bad(:\n    pass",
            "agent_log": [],
            "session_id": "test",
            "langfuse_handler": None,
            "validation_attempts": 0,
        }
        result = validator_agent(state)
        assert result["validation_passed"] is False
        assert result["validation_error"] is not None

    def test_agent_increments_attempts_on_failure(self):
        state = {
            "code": "not valid python ::::",
            "agent_log": [],
            "session_id": "test",
            "langfuse_handler": None,
            "validation_attempts": 1,
        }
        result = validator_agent(state)
        assert result["validation_passed"] is False
        assert result["validation_attempts"] == 2
