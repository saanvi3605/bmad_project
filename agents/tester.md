# Agent: Tester

**Role:** QA Engineer (Test Case Designer)  
**Module:** `agents_impl/tester_agent.py`  
**LLM:** Yes (via `core.agent_runner.run_agent`)  
**Skills:** `test_generation` Stage 1 (see `skills/test_generation.yaml`)

---

## Responsibility

Generates structured human-readable test case specifications (TC-XX format) from the technical design and the generated code. These specifications are consumed by the TestWriter agent to produce a runnable pytest file.

---

## Inputs

| Field | Description |
|---|---|
| `technical_design` | From Architect |
| `code` | From Developer (sanitized) |

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `test_cases` | str | Structured TC-XX test case specs (max 10 cases) |

---

## Output Format

```
- ID: TC-01
- Type: unit / integration / edge
- Description: what is being tested
- Input: example input
- Expected Output: expected result
```

---

## Test Types

- **Unit:** Individual function or route behaviour
- **Integration:** End-to-end request/response flow  
- **Edge:** Empty inputs, invalid data, missing fields

---

## Routing

`Tester → Reviewer` (unconditional edge)
