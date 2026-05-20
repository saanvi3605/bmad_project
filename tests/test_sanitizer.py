"""
tests/test_sanitizer.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for core.sanitizer.sanitize_code().

Tests cover all transformation paths:
  - Markdown fence stripping
  - SQLAlchemy pattern removal
  - Disallowed import stripping
  - Raw HTML stripping
  - REQUIRED_IMPORTS injection
  - init_db injection
  - Jinja2 syntax stripping
  - uvicorn.run normalization
────────────────────────────────────────────────────────────────────────────────
"""

import pytest
from core.sanitizer import sanitize_code, REQUIRED_IMPORTS, STRIP_PREFIXES


class TestMarkdownFenceStripping:
    def test_strips_python_fence(self):
        code = "```python\nimport os\n```"
        result = sanitize_code(code)
        assert "```python" not in result
        assert "```" not in result

    def test_strips_bare_fence(self):
        code = "```\nimport os\n```"
        result = sanitize_code(code)
        assert "```" not in result


class TestRequiredImportsInjected:
    def test_all_required_imports_present(self):
        result = sanitize_code("x = 1")
        for imp in REQUIRED_IMPORTS:
            assert imp in result, f"Required import missing: {imp}"

    def test_no_duplicate_imports(self):
        code = "import os\nimport sqlite3\nx = 1"
        result = sanitize_code(code)
        # import os should appear exactly once (from REQUIRED_IMPORTS)
        assert result.count("import os\n") == 1


class TestSQLAlchemyRemoval:
    def test_engine_removed(self):
        code = "engine = create_engine('sqlite:///app.db')\nx = 1"
        result = sanitize_code(code)
        assert "create_engine" not in result

    def test_sessionmaker_removed(self):
        code = "SessionLocal = sessionmaker(bind=engine)\nx = 1"
        result = sanitize_code(code)
        assert "sessionmaker" not in result

    def test_declarative_base_removed(self):
        code = "Base = declarative_base()\nx = 1"
        result = sanitize_code(code)
        assert "declarative_base" not in result

    def test_base_metadata_replaced_with_init_db(self):
        code = "Base.metadata.create_all(engine)\nx = 1"
        result = sanitize_code(code)
        assert "Base.metadata.create_all" not in result
        assert "init_db()" in result

    def test_db_depends_removed(self):
        code = "async def route(db: Session = Depends(get_db)):\n    pass"
        result = sanitize_code(code)
        assert "db: Session" not in result

    def test_get_db_function_removed(self):
        code = (
            "def get_db():\n"
            "    db = SessionLocal()\n"
            "    try:\n"
            "        yield db\n"
            "    finally:\n"
            "        db.close()\n\n"
            "@app.get('/')\n"
            "def root():\n"
            "    return {}\n"
        )
        result = sanitize_code(code)
        assert "def get_db" not in result


class TestDisallowedImportStripping:
    def test_sqlalchemy_import_stripped(self):
        code = "import sqlalchemy\nx = 1"
        result = sanitize_code(code)
        assert "import sqlalchemy" not in result

    def test_asyncpg_stripped(self):
        code = "import asyncpg\nx = 1"
        result = sanitize_code(code)
        assert "import asyncpg" not in result

    def test_psycopg2_stripped(self):
        code = "import psycopg2\nx = 1"
        result = sanitize_code(code)
        assert "import psycopg2" not in result

    def test_jinja2_import_stripped(self):
        code = "import jinja2\nx = 1"
        result = sanitize_code(code)
        assert "import jinja2" not in result

    def test_fastapi_templating_stripped(self):
        code = "from fastapi.templating import Jinja2Templates\nx = 1"
        result = sanitize_code(code)
        assert "Jinja2Templates" not in result


class TestRawHTMLStripping:
    def test_raw_html_outside_string_stripped(self):
        code = "<html><body></body></html>\nx = 1"
        result = sanitize_code(code)
        assert "<html>" not in result

    def test_html_inside_string_preserved(self):
        code = '"""<html><body></body></html>"""\nx = 1'
        result = sanitize_code(code)
        # Content inside triple-quoted strings should not be stripped
        assert "<html>" in result


class TestInitDbInjection:
    def test_init_db_injected_when_missing(self):
        code = "app = FastAPI()\n"
        result = sanitize_code(code)
        assert "def init_db" in result

    def test_init_db_not_duplicated_when_present(self):
        code = "def init_db():\n    pass\n\ninit_db()\n"
        result = sanitize_code(code)
        assert result.count("def init_db") == 1


class TestJinja2Stripping:
    def test_jinja_block_tag_stripped(self):
        code = "HTML = '{% for item in items %}{{ item }}{% endfor %}'\n"
        result = sanitize_code(code)
        assert "{%" not in result
        assert "{{" not in result

    def test_jinja_variable_tag_stripped(self):
        code = "HTML = '{{ user.name }}'\n"
        result = sanitize_code(code)
        assert "{{" not in result


class TestUvicornRunNormalization:
    def test_uvicorn_run_normalized(self):
        code = 'uvicorn.run(app, host="127.0.0.1", port=5000)\n'
        result = sanitize_code(code)
        assert 'uvicorn.run("generated_app:app", host="0.0.0.0", port=8000, reload=True)' in result

    def test_uvicorn_run_multiline_normalized(self):
        code = (
            "uvicorn.run(\n"
            "    app,\n"
            "    host='0.0.0.0',\n"
            "    port=8080\n"
            ")\n"
        )
        result = sanitize_code(code)
        assert 'uvicorn.run("generated_app:app", host="0.0.0.0", port=8000, reload=True)' in result
