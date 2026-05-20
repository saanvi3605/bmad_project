"""
workflow.py — backward-compatibility re-export shim.

Any code that previously did ``from workflow import app`` continues to
work unchanged.  The compiled graph now lives in orchestration/graph.py.
"""

from orchestration.graph import app  # noqa: F401

__all__ = ["app"]
