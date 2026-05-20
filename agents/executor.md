# Agent: Executor

**Role:** Application Launch Verifier  
**Module:** `agents_impl/executor_agent.py`  
**LLM:** No — subprocess only  
**Skills:** `code_execution` (see `skills/code_execution.yaml`)

---

## Responsibility

Writes the generated Python application to disk at the configured output path, launches it as a subprocess, waits for the startup period, then polls the process to determine whether the application started successfully or crashed.

---

## Mechanism

```python
process = subprocess.Popen(["python", output_path], stdout=PIPE, stderr=PIPE, text=True)
time.sleep(startup_wait_seconds)   # default: 8s (config-driven)
if process.poll() is None:
    # Process still alive → success
    process.terminate()
    state["execution_result"] = "App started successfully and is running."
else:
    # Process exited → failure
    stdout, stderr = process.communicate()
    state["execution_result"] = stdout
    state["execution_error"] = stderr
```

---

## Configuration (from `config/models.yaml`)

| Key | Default | Description |
|---|---|---|
| `executor.output_file` | `outputs/generated_app.py` | Path to write the generated app |
| `executor.startup_wait_seconds` | `8` | Seconds to wait before polling |

---

## Session Archiving

When `session.archive_outputs: true` in `models.yaml`, the executor copies the generated app into `outputs/sessions/{session_id}/generated_app.py` and sets `state["output_dir"]`.

---

## Outputs

| Field | Description |
|---|---|
| `execution_result` | Success message or stdout |
| `execution_error` | Empty string on success; stderr on failure |
| `output_dir` | Absolute path to per-session archive (if archiving enabled) |

---

## Routing

`Executor → TestWriter` (unconditional edge)
