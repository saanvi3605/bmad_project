"""
tests/test_test_writer.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for agents_impl.test_writer_agent post-processing helper functions.

These helpers are extracted as named functions in the BMAD migration which
makes them individually unit-testable.
────────────────────────────────────────────────────────────────────────────────
"""

import pytest
from agents_impl.test_writer_agent import (
    _strip_async_patterns,
    _fix_count_assertions,
    _extract_test_functions,
    FIXTURE_BLOCK,
)


class TestStripAsyncPatterns:
    def test_removes_anyio_decorator(self):
        code = "@pytest.mark.anyio\ndef test_foo(client):\n    pass\n"
        result = _strip_async_patterns(code)
        assert "@pytest.mark.anyio" not in result

    def test_removes_asyncio_decorator(self):
        code = "@pytest.mark.asyncio\nasync def test_foo(client):\n    pass\n"
        result = _strip_async_patterns(code)
        assert "@pytest.mark.asyncio" not in result

    def test_converts_async_def_to_def(self):
        code = "async def test_create(client):\n    pass\n"
        result = _strip_async_patterns(code)
        assert "async def test_" not in result
        assert "def test_create(client):" in result

    def test_removes_await_client(self):
        code = "    response = await client.get('/items')\n"
        result = _strip_async_patterns(code)
        assert "await client." not in result
        assert "response = client.get('/items')" in result

    def test_removes_await_ac(self):
        code = "    response = await ac.post('/items', json={})\n"
        result = _strip_async_patterns(code)
        assert "await ac." not in result
        assert "client.post" in result

    def test_removes_import_anyio(self):
        code = "import anyio\n\ndef test_foo():\n    pass\n"
        result = _strip_async_patterns(code)
        assert "import anyio" not in result

    def test_removes_import_asyncio(self):
        code = "import asyncio\n\ndef test_foo():\n    pass\n"
        result = _strip_async_patterns(code)
        assert "import asyncio" not in result


class TestFixCountAssertions:
    def test_nonzero_count_becomes_gte_1(self):
        code = "    assert len(response.json()) == 3\n"
        result = _fix_count_assertions(code)
        assert "assert len(response.json()) >= 1" in result

    def test_zero_count_becomes_empty_list(self):
        code = "    assert len(response.json()) == 0\n"
        result = _fix_count_assertions(code)
        assert "assert response.json() == []" in result

    def test_one_count_becomes_gte_1(self):
        code = "    assert len(response.json()) == 1\n"
        result = _fix_count_assertions(code)
        assert "assert len(response.json()) >= 1" in result

    def test_no_match_unchanged(self):
        code = "    assert response.status_code == 200\n"
        result = _fix_count_assertions(code)
        assert result == code


class TestExtractTestFunctions:
    def test_strips_imports_above_first_test(self):
        code = (
            "import pytest\n"
            "from something import other\n\n"
            "def test_first(client):\n"
            "    pass\n\n"
            "def test_second(client):\n"
            "    pass\n"
        )
        result = _extract_test_functions(code)
        assert "import pytest" not in result
        assert "def test_first(client):" in result
        assert "def test_second(client):" in result

    def test_no_test_function_returns_original(self):
        code = "def helper():\n    pass\n"
        result = _extract_test_functions(code)
        assert result == code

    def test_dedents_indented_test_functions(self):
        code = "class TestSuite:\n    def test_foo(client):\n        pass\n"
        result = _extract_test_functions(code)
        # After dedent, test functions should not be indented
        assert "\ndef test_" in result or "def test_" in result


class TestFixtureBlock:
    def test_fixture_block_contains_required_imports(self):
        assert "import pytest" in FIXTURE_BLOCK
        assert "import os" in FIXTURE_BLOCK
        assert "import sqlite3" in FIXTURE_BLOCK
        assert "import gc" in FIXTURE_BLOCK

    def test_fixture_block_imports_from_generated_app(self):
        assert "from generated_app import app, init_db" in FIXTURE_BLOCK

    def test_fixture_block_uses_test_client(self):
        assert "TestClient" in FIXTURE_BLOCK

    def test_fixture_block_cleans_up_db(self):
        assert "test_app.db" in FIXTURE_BLOCK
        assert "os.remove" in FIXTURE_BLOCK
