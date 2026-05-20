"""
tests/conftest.py — shared pytest configuration.

Adds the project root to sys.path so all module imports resolve correctly
regardless of where pytest is invoked from.
"""

import sys
import pathlib

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
