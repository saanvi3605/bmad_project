"""
core/multi_file_parser.py
────────────────────────────────────────────────────────────────────────────────
Parses multi-file output from the RAG developer agent.

The developer agent uses this delimiter format:

    ### FILE: main.py ###
    <python source>

    ### FILE: requirements.txt ###
    <pip requirements>

    ### FILE: docker-compose.yml ###
    <yaml content>

parse_multi_file() splits on those delimiters and returns a dict:
    {"main.py": "...", "requirements.txt": "...", "docker-compose.yml": "..."}

If the output contains no delimiters, it's treated as a single main.py file.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re

# Matches lines like:  ### FILE: main.py ###
_DELIMITER = re.compile(r"^#{2,}\s*FILE:\s*(.+?)\s*#{2,}\s*$", re.MULTILINE | re.IGNORECASE)

# Strip markdown code-fence wrappers from individual file content
_FENCE = re.compile(r"^```[a-zA-Z]*\n?|^```\s*$", re.MULTILINE)


def parse_multi_file(content: str) -> dict[str, str]:
    """
    Extract filename → content mappings from a delimited developer output.

    Parameters
    ----------
    content : str
        Raw string from the LLM (may contain markdown fences, prose preamble, etc.)

    Returns
    -------
    dict[str, str]
        At minimum {"main.py": "<python source>"}.
        May also contain "requirements.txt", "docker-compose.yml", ".env.example", etc.
    """
    parts = _DELIMITER.split(content)

    # parts = [preamble, filename1, body1, filename2, body2, ...]
    if len(parts) < 3:
        # No delimiters found — treat whole output as main.py
        return {"main.py": _clean(content)}

    files: dict[str, str] = {}
    # Skip preamble (parts[0]); iterate pairs (filename, body)
    for i in range(1, len(parts) - 1, 2):
        filename = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        files[filename] = _clean(body)

    # Guarantee main.py always exists
    if "main.py" not in files and files:
        # Use whichever .py file comes first as the "main" file
        py_files = [k for k in files if k.endswith(".py")]
        if py_files:
            files["main.py"] = files.pop(py_files[0])
        else:
            # Fallback: first file becomes main.py
            first_key = next(iter(files))
            files["main.py"] = files.pop(first_key)

    return files


def _clean(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    text = _FENCE.sub("", text)
    return text.strip()
