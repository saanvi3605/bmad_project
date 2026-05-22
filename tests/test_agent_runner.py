"""
tests/test_agent_runner.py
────────────────────────────────────────────────────────────────────────────────
Tests for core.agent_runner — lazy initialization, extract_tokens,
monkeypatch compatibility.
────────────────────────────────────────────────────────────────────────────────
"""

import warnings
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_llm():
    """Reset the lazy singleton between tests."""
    import core.agent_runner as ar
    original = ar._llm
    ar._llm = None
    yield
    ar._llm = original


class TestLazyInitialization:
    def test_import_does_not_call_build_llm_without_key(self):
        """Importing agent_runner must not raise EnvironmentError even without GROQ_API_KEY."""
        import os
        original = os.environ.pop("GROQ_API_KEY", None)
        try:
            # Re-importing should not crash
            import core.agent_runner  # noqa: F401
        except EnvironmentError:
            pytest.fail("agent_runner raised EnvironmentError at import time")
        finally:
            if original:
                os.environ["GROQ_API_KEY"] = original

    def test_llm_none_before_first_call(self):
        import core.agent_runner as ar
        # _llm should be None (reset by fixture)
        assert ar._llm is None

    def test_get_llm_builds_on_first_call(self):
        mock = MagicMock()
        # Patch the name as it exists in agent_runner's local namespace
        with patch("core.agent_runner.build_llm", return_value=mock):
            import core.agent_runner as ar
            result = ar._get_llm()
            assert result is mock
            assert ar._llm is mock

    def test_get_llm_reuses_singleton(self):
        mock = MagicMock()
        with patch("core.agent_runner.build_llm", return_value=mock) as mock_build:
            import core.agent_runner as ar
            ar._get_llm()
            ar._get_llm()
            # build_llm should be called only once — second call reuses singleton
            mock_build.assert_called_once()

    def test_monkeypatch_llm_before_call(self):
        """Tests can assign ar._llm = Mock() to prevent real API calls."""
        import core.agent_runner as ar
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content="mocked",
            usage_metadata={"input_tokens": 1, "output_tokens": 1},
        )
        ar._llm = fake_llm
        assert ar._get_llm() is fake_llm


class TestExtractTokens:
    def test_usage_metadata_path(self):
        from core.agent_runner import extract_tokens
        resp = MagicMock()
        resp.usage_metadata = {"input_tokens": 100, "output_tokens": 200}
        inp, out = extract_tokens(resp)
        assert inp == 100
        assert out == 200

    def test_response_metadata_fallback(self):
        from core.agent_runner import extract_tokens
        resp = MagicMock()
        resp.usage_metadata = None
        resp.response_metadata = {"token_usage": {"prompt_tokens": 50, "completion_tokens": 75}}
        inp, out = extract_tokens(resp)
        assert inp == 50
        assert out == 75

    def test_neither_path_returns_zeros_and_warns(self):
        from core.agent_runner import extract_tokens
        resp = MagicMock()
        resp.usage_metadata = None
        # Set to None so the response_metadata fallback also fails → warning emitted
        resp.response_metadata = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inp, out = extract_tokens(resp)
        assert inp == 0
        assert out == 0
        assert len(w) == 1
        assert "token" in str(w[0].message).lower()


class TestRunAgent:
    def test_run_agent_returns_content(self):
        import core.agent_runner as ar

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="generated code here",
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
        )
        ar._llm = mock_llm

        state = {
            "langfuse_handler": None,
            "agent_log": [],
            "session_id": "test-session",
        }
        result = ar.run_agent("test prompt", "developer", "developer_clean_v1", state)
        assert result == "generated code here"

    def test_run_agent_appends_to_agent_log(self):
        import core.agent_runner as ar

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="code",
            usage_metadata={"input_tokens": 5, "output_tokens": 15},
        )
        ar._llm = mock_llm

        state = {
            "langfuse_handler": None,
            "agent_log": [],
            "session_id": "test-session",
        }
        ar.run_agent("prompt", "planner", "planner_v1", state)
        assert len(state["agent_log"]) == 1
        rec = state["agent_log"][0]
        assert rec.agent_name == "planner"
        assert rec.input_tokens == 5
        assert rec.output_tokens == 15

    def test_run_agent_attaches_langfuse_handler(self):
        import core.agent_runner as ar

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="response",
            usage_metadata={"input_tokens": 1, "output_tokens": 1},
        )
        ar._llm = mock_llm

        fake_handler = MagicMock()
        state = {
            "langfuse_handler": fake_handler,
            "agent_log": [],
            "session_id": "test-session",
        }
        ar.run_agent("prompt", "architect", "architect_v1", state)
        call_kwargs = mock_llm.invoke.call_args
        callbacks = call_kwargs[1]["config"]["callbacks"]
        assert fake_handler in callbacks


class TestBuildInitialState:
    def test_all_fields_present(self):
        from core.agent_runner import build_initial_state
        state = build_initial_state("build me a todo app", "session-123", None)

        expected_fields = [
            "user_request", "functional_spec", "technical_design", "code",
            "test_cases", "review_feedback", "review_approved", "review_attempts",
            "validation_passed", "validation_error", "validation_attempts",
            "execution_result", "execution_error", "langfuse_handler", "test_file",
            "session_id", "pipeline_status", "agent_log", "output_dir",
        ]
        for field in expected_fields:
            assert field in state, f"Missing field: {field}"

    def test_initial_values(self):
        from core.agent_runner import build_initial_state
        state = build_initial_state("test request", "sess-abc", None)
        assert state["user_request"] == "test request"
        assert state["session_id"] == "sess-abc"
        assert state["pipeline_status"] == "running"
        assert state["review_attempts"] == 0
        assert state["validation_attempts"] == 0
        assert state["agent_log"] == []
        assert state["code"] is None
