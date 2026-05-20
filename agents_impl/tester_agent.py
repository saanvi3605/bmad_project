"""
agents_impl/tester_agent.py
────────────────────────────────────────────────────────────────────────────────
Tester agent — generates structured test cases from the technical design and
the generated code.

Changes from original:
  - Removed module-level ChatGroq instantiation.
  - LLM call delegated to core.agent_runner.run_agent().
  - Prompt text preserved verbatim (f-string converted to .format()).
  - State mutations preserved verbatim.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent

_AGENT_NAME = "tester"
_PROMPT_KEY = "tester_v1"


def tester_agent(state: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""You are a QA engineer following the BMAD methodology. Produce a test strategy using EXACTLY this template — every section heading must appear verbatim.

Technical Design:
{state['technical_design']}

OUTPUT TEMPLATE (reproduce every heading exactly):

## Test Strategy

### 1. Scope
[1-2 sentences: what is being tested and what is out of scope]

### 2. Unit Tests
[Test individual functions in isolation — DB helpers, calculations, validators]
| ID | Function Under Test | Input | Expected Output |
|----|--------------------|----|-----------------|
| UT-01 | | | |
[3-4 rows minimum]

### 3. Integration Tests
[Test end-to-end flows: form submit → DB write → UI refresh]
| ID | Flow | Steps | Expected Result |
|----|------|-------|-----------------|
| IT-01 | | | |
[3-4 rows minimum]

### 4. System Tests
[Test the full app as a black box: navigation, page loads, full user journeys]
| ID | Scenario | User Actions | Expected Behaviour |
|----|----------|-------------|-------------------|
| ST-01 | | | |
[2-3 rows minimum]

### 5. Mock Tests
[Tests using mocked dependencies: mock DB connection, mock datetime, mock cache]
| ID | What is Mocked | Test Scenario | Expected Behaviour |
|----|---------------|--------------|-------------------|
| MT-01 | | | |
[2-3 rows minimum]

### 6. Edge Cases
[Boundary values, empty inputs, invalid data, concurrent writes]
| ID | Input Condition | Expected Behaviour |
|----|----------------|-------------------|
| EC-01 | | |
[3-4 rows minimum]

### 7. Test Data Requirements
[Specific sample values needed: record counts, field values, DB seed data]

RULES:
- Be specific to the actual technical design above — use real field names, real table names, real form keys
- Every test ID must be unique
"""
    content = run_agent(
        prompt=prompt,
        agent_name=_AGENT_NAME,
        prompt_key=_PROMPT_KEY,
        state=state,
        light=True,
    )
    state["test_cases"] = content
    return state
