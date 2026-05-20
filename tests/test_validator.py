"""
tests/test_validator.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for agents_impl.validator_agent.validate_code() and should_fix().

Tests cover all six validation stages and the routing function.
────────────────────────────────────────────────────────────────────────────────
"""

import pytest
from agents_impl.validator_agent import validate_code, should_fix, validator_agent

# Minimal valid code that passes all six checks
VALID_CODE = '''\
import os
import sqlite3
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

def init_db():
    conn = sqlite3.connect("app.db")
    conn.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

init_db()

HTML = "<html><body><h1>App</h1></body></html>"

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML

if __name__ == "__main__":
    uvicorn.run("generated_app:app", host="0.0.0.0", port=8000, reload=True)
'''


class TestValidateCodePassing:
    def test_valid_code_passes(self):
        passed, msg = validate_code(VALID_CODE)
        assert passed is True
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
    def test_missing_get_route(self):
        code = VALID_CODE.replace('@app.get("/"', '@app.get("/other"')
        passed, msg = validate_code(code)
        assert passed is False
        assert "Missing GET / route" in msg

    def test_missing_html_response(self):
        code = VALID_CODE.replace("HTMLResponse", "JSONResponse")
        passed, msg = validate_code(code)
        assert passed is False
        assert "Missing HTMLResponse" in msg

    def test_missing_uvicorn_run(self):
        code = VALID_CODE.replace("uvicorn.run(", "# uvicorn.run(")
        passed, msg = validate_code(code)
        assert passed is False
        assert "Missing uvicorn.run" in msg

    def test_missing_main_guard(self):
        code = VALID_CODE.replace("if __name__", "if True")
        passed, msg = validate_code(code)
        assert passed is False
        assert "Missing if __name__" in msg


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
    def test_jinja2_block_tag_detected(self):
        code = VALID_CODE.replace(
            'HTML = "<html><body><h1>App</h1></body></html>"',
            'HTML = "<html>{% for item in items %}</html>"',
        )
        passed, msg = validate_code(code)
        assert passed is False
        assert "Jinja2" in msg

    def test_jinja2_variable_tag_detected(self):
        code = VALID_CODE.replace(
            'HTML = "<html><body><h1>App</h1></body></html>"',
            'HTML = "<html>{{ item.name }}</html>"',
        )
        passed, msg = validate_code(code)
        assert passed is False
        assert "Jinja2" in msg


class TestShouldFix:
    def test_passed_returns_passed(self):
        state = {"validation_passed": True, "validation_attempts": 0}
        result = should_fix(state)
        assert result == "passed"
        assert state["validation_attempts"] == 0  # not incremented on pass

    def test_first_failure_returns_fix(self):
        state = {"validation_passed": False, "validation_attempts": 0}
        result = should_fix(state)
        assert result == "fix"
        assert state["validation_attempts"] == 1

    def test_second_failure_returns_fix(self):
        state = {"validation_passed": False, "validation_attempts": 1}
        result = should_fix(state)
        assert result == "fix"
        assert state["validation_attempts"] == 2

    def test_max_attempts_returns_max_attempts(self):
        state = {"validation_passed": False, "validation_attempts": 2}
        result = should_fix(state)
        assert result == "max_attempts"


class TestValidatorAgent:
    def test_agent_sets_passed_on_valid_code(self):
        state = {"code": VALID_CODE, "agent_log": [], "session_id": "test", "langfuse_handler": None}
        result = validator_agent(state)
        assert result["validation_passed"] is True
        assert result["validation_error"] is None

    def test_agent_sets_failed_on_bad_code(self):
        state = {"code": "def bad(:\n    pass", "agent_log": [], "session_id": "test", "langfuse_handler": None}
        result = validator_agent(state)
        assert result["validation_passed"] is False
        assert result["validation_error"] is not None
