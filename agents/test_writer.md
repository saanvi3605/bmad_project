# Agent: TestWriter

**Role:** Senior QA Engineer (Test File Generator)  
**Module:** `agents_impl/test_writer_agent.py`  
**LLM:** Yes (via `core.agent_runner.run_agent`)  
**Skills:** `test_generation` Stage 2 (see `skills/test_generation.yaml`)

---

## Responsibility

Generates a runnable pytest file by combining a hardcoded fixture preamble (`FIXTURE_BLOCK`) with LLM-generated test functions derived from the Tester's specifications and the actual generated code.

---

## FIXTURE_BLOCK (hardcoded — never modified by LLM)

```python
import pytest
import os
import sqlite3
import gc
from starlette.testclient import TestClient
from generated_app import app, init_db

@pytest.fixture
def client():
    os.environ["DB_PATH"] = "test_app.db"
    init_db()
    with TestClient(app) as c:
        yield c
    gc.collect()
    for _db in ("test_app.db", "app.db"):
        try:
            os.remove(_db)
        except (PermissionError, FileNotFoundError):
            pass
```

---

## Post-processing Transforms (applied to LLM output in order)

1. `_strip_async_patterns()` — removes `@pytest.mark.anyio/asyncio`, `async def test_`, `await client.`, `await ac.`, `import anyio/asyncio`, `async with AsyncClient` blocks
2. `_fix_count_assertions()` — `assert len(response.json()) == N` → `>= 1` (except `== 0` → `== []`)
3. `_extract_test_functions()` — strips everything above first `def test_`, dedents

---

## Outputs

| Field | Description |
|---|---|
| `test_file` | Path to the written pytest file (from `executor.test_output_file` config) |

---

## Run Command

```bash
python -m pytest outputs/test_generated_app.py -v
```

---

## Routing

`TestWriter → END` (terminal node)
