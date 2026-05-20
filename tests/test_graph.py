"""
tests/test_graph.py
────────────────────────────────────────────────────────────────────────────────
Tests for orchestration.graph — graph import, compilation, node inventory,
and routing function imports.

LLM is monkeypatched so these tests run without GROQ_API_KEY.
────────────────────────────────────────────────────────────────────────────────
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_llm():
    """Prevent any real LLM instantiation during graph import tests."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="mock response",
        usage_metadata={"input_tokens": 5, "output_tokens": 10},
    )
    with patch("core.llm_factory.build_llm", return_value=mock):
        import core.agent_runner as ar
        ar._llm = mock
        yield mock


class TestGraphImport:
    def test_graph_module_imports(self):
        from orchestration import graph
        assert graph is not None

    def test_app_is_compiled_graph(self):
        from orchestration.graph import app
        # CompiledGraph has an invoke method
        assert hasattr(app, "invoke")

    def test_workflow_is_state_graph(self):
        from orchestration.graph import workflow
        assert workflow is not None


class TestNodeInventory:
    def test_all_eight_nodes_registered(self):
        from orchestration.graph import workflow
        # LangGraph StateGraph stores nodes in graph.nodes
        node_names = set(workflow.nodes.keys())
        expected = {"Planner", "Architect", "Developer", "Validator",
                    "Tester", "Reviewer", "Executor", "TestWriter"}
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )


class TestRoutingFunctions:
    def test_should_fix_importable(self):
        from agents_impl.validator_agent import should_fix
        assert callable(should_fix)

    def test_should_retry_importable(self):
        from agents_impl.reviewer_agent import should_retry
        assert callable(should_retry)

    def test_should_fix_all_return_values(self):
        from agents_impl.validator_agent import should_fix
        # passed
        assert should_fix({"validation_passed": True, "validation_attempts": 0}) == "passed"
        # fix
        state = {"validation_passed": False, "validation_attempts": 0}
        assert should_fix(state) == "fix"
        # max_attempts
        assert should_fix({"validation_passed": False, "validation_attempts": 2}) == "max_attempts"

    def test_should_retry_all_return_values(self):
        from agents_impl.reviewer_agent import should_retry
        assert should_retry({"review_approved": True, "review_attempts": 1}) == "approved"
        assert should_retry({"review_approved": False, "review_attempts": 1}) == "retry"
        assert should_retry({"review_approved": False, "review_attempts": 2}) == "max_retries"


class TestWorkflowShim:
    def test_workflow_shim_imports_app(self):
        from workflow import app
        assert hasattr(app, "invoke")
