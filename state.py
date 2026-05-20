"""
state.py — backward-compatibility re-export shim.

Any code that previously did ``from state import AgentState`` or
``from state import BMADState`` continues to work unchanged.
The canonical definition is in core/state.py.
"""

from core.state import BMADState, AgentState  # noqa: F401

__all__ = ["BMADState", "AgentState"]
