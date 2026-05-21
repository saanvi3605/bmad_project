"""
main.py — CLI entrypoint for the BMAD pipeline.

Usage:
    python main.py "Build a task tracker with add/delete/list"

Or run interactively:
    python main.py
"""

from __future__ import annotations

import sys

import yaml
import pathlib


def _load_pipeline_rules() -> str:
    """Load the pipeline constraints suffix from config/pipeline_rules.yaml."""
    try:
        cfg_path = pathlib.Path(__file__).parent / "config" / "pipeline_rules.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return data.get("prompt_suffix", "")
    except Exception:
        return ""


def run(user_request: str) -> None:
    from core.guardrails import check_prompt, format_rejection
    from core.observability import create_session, get_pipeline_summary, flush
    from core.agent_runner import build_initial_state, reset_llm_singletons
    from orchestration.graph import app

    # ── Guardrail check — before spending any tokens ───────────────────────
    is_safe, reason = check_prompt(user_request.strip())
    if not is_safe:
        print("\n" + "=" * 70)
        print("  REQUEST BLOCKED")
        print("=" * 70)
        print(format_rejection(reason).replace("**", "").replace("_", ""))
        return

    reset_llm_singletons()

    print("\n" + "=" * 70)
    print("  BMAD AI ORCHESTRATION PIPELINE")
    print("=" * 70)
    print(f"\nRequest: {user_request[:120]}{'...' if len(user_request) > 120 else ''}\n")

    rules_suffix = _load_pipeline_rules()
    full_request = user_request + rules_suffix

    session_id, langfuse_handler = create_session("BMAD CLI Run")
    print(f"Session ID: {session_id}\n")

    initial_state = build_initial_state(
        user_request=full_request,
        session_id=session_id,
        langfuse_handler=langfuse_handler,
    )

    print("Starting pipeline...\n")
    invoke_cfg = (
        {"callbacks": [langfuse_handler]}
        if langfuse_handler is not None and hasattr(langfuse_handler, "on_llm_start")
        else {}
    )
    final_state = app.invoke(initial_state, invoke_cfg)

    # Flush Langfuse
    try:
        flush(langfuse_handler)
    except Exception:
        pass

    # Print summary
    summary = get_pipeline_summary(final_state)
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nAgents executed: {' → '.join(summary.agents_called)}")
    print(f"Total tokens:    {summary.total_tokens:,}")
    print(f"Estimated cost:  ${summary.total_cost_usd:.4f}")
    print(f"Total latency:   {summary.total_latency_ms/1000:.1f}s")

    exec_result = final_state.get("execution_result", "")
    exec_error = final_state.get("execution_error", "")

    if exec_result:
        print(f"\nExecution: {exec_result}")
    if exec_error:
        print(f"Execution error: {exec_error}")

    test_file = final_state.get("test_file", "")
    if test_file:
        print(f"\nTest file: {test_file}")
        print(f"Run tests: python -m pytest {test_file} -v")

    output_dir = final_state.get("output_dir", "")
    if output_dir:
        print(f"Session archive: {output_dir}")

    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        request = input("Enter your application request: ").strip()
        if not request:
            print("No request provided. Exiting.")
            sys.exit(1)

    run(request)
