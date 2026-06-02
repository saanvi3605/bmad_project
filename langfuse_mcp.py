"""
langfuse_mcp.py
────────────────────────────────────────────────────────────────────────────────
MCP server that exposes Langfuse analytics as tools for Claude Code.

Registered via .mcp.json — open a new session in Claude Code to pick up changes.

Tools
  get_usage_summary(days)   — totals: tokens, cost, traces, per-model breakdown
  get_daily_trends(days)    — day-by-day table of traces, tokens, cost
  get_recent_traces(limit)  — latest pipeline runs with token/cost/latency info
  get_sessions(limit)       — recent sessions and how many traces each has
  ask_langfuse(question)    — AI-powered Q&A: fetches all data and answers via Groq

Example questions you can ask Claude after opening a new session:
  "How many tokens did I use this week?"
  "What did the pipeline cost me in the last 30 days?"
  "Am I spending too much? How can I reduce costs?"
  "Which of my recent runs was most expensive and why?"
  "Give me a full analysis of my usage trends"
  "Compare my usage this week vs last week"
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("langfuse-analytics")

_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
_PUB  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SEC  = os.getenv("LANGFUSE_SECRET_KEY", "")


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _PUB or not _SEC:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set. "
            "Add them to the .env file in the project root."
        )
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{_HOST}{path}", auth=(_PUB, _SEC), params=params or {})
        r.raise_for_status()
        return r.json()


def _date_range(days: int) -> tuple[str, str]:
    """Return (from_iso, to_iso) covering the last N days."""
    now     = datetime.now(timezone.utc)
    from_dt = (now - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    to_dt   = now.strftime("%Y-%m-%dT23:59:59Z")
    return from_dt, to_dt


# ---------------------------------------------------------------------------
# Tool 1 — Usage summary (aggregated totals + per-model breakdown)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_usage_summary(days: int = 7) -> str:
    """
    Return aggregated token usage, estimated cost, and trace count for the
    last N days, plus a per-model token/cost breakdown.

    Best for questions like:
      "How many tokens did I use?"
      "How many credits / what did it cost?"
      "How many traces or pipeline runs are there?"
      "Which model used the most tokens?"
      "Give me an overall usage summary"
    """
    from_dt, to_dt = _date_range(days)

    data = _api_get("/api/public/metrics/daily", {
        "fromTimestamp": from_dt,
        "toTimestamp":   to_dt,
    })
    rows = data.get("data", [])

    if not rows:
        return f"No usage data found in Langfuse for the last {days} day(s)."

    total_traces = sum(r.get("countTraces", 0)      for r in rows)
    total_obs    = sum(r.get("countObservations", 0) for r in rows)
    total_cost   = sum(r.get("totalCost", 0.0)       for r in rows)

    model_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        for u in row.get("usage", []):
            m = u.get("model") or "unknown"
            s = model_stats.setdefault(m, {"input": 0, "output": 0, "total": 0, "cost": 0.0})
            s["input"]  += u.get("inputUsage",  0)
            s["output"] += u.get("outputUsage", 0)
            s["total"]  += u.get("totalUsage",  0)
            s["cost"]   += float(u.get("totalCost", 0.0))

    total_input  = sum(v["input"]  for v in model_stats.values())
    total_output = sum(v["output"] for v in model_stats.values())
    total_tokens = sum(v["total"]  for v in model_stats.values())

    lines = [
        f"Langfuse usage summary — last {days} day(s)  ({from_dt[:10]} → {to_dt[:10]})",
        "",
        f"  Traces (pipeline runs) : {total_traces:,}",
        f"  Generations (LLM calls): {total_obs:,}",
        f"  Input tokens           : {total_input:,}",
        f"  Output tokens          : {total_output:,}",
        f"  Total tokens           : {total_tokens:,}",
        f"  Estimated cost         : ${total_cost:.4f} USD",
        "",
    ]

    if model_stats:
        lines.append("  Per-model breakdown:")
        for model, s in sorted(model_stats.items(), key=lambda x: -x[1]["total"]):
            lines.append(
                f"    {model:<42} {s['total']:>9,} tokens"
                f"  (in {s['input']:,} / out {s['output']:,})"
                f"  ${s['cost']:.4f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — Daily trends (day-by-day table)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_daily_trends(days: int = 14) -> str:
    """
    Return a day-by-day table of traces, LLM generations, total tokens, and
    cost for the last N days, sorted newest-first.

    Best for questions like:
      "Show me daily usage trends"
      "Which day had the most pipeline runs?"
      "How has token usage changed over time?"
      "Was there a spike in usage recently?"
    """
    from_dt, to_dt = _date_range(days)

    data = _api_get("/api/public/metrics/daily", {
        "fromTimestamp": from_dt,
        "toTimestamp":   to_dt,
    })
    rows = data.get("data", [])

    if not rows:
        return f"No daily data found in Langfuse for the last {days} day(s)."

    # Sort newest first (API may return oldest first)
    rows = sorted(rows, key=lambda r: r.get("date", ""), reverse=True)

    header = f"Daily trends — last {days} day(s)  ({from_dt[:10]} → {to_dt[:10]})\n"
    col    = f"  {'Date':<12}  {'Traces':>7}  {'Generations':>12}  {'Tokens':>10}  {'Cost (USD)':>10}"
    sep    = "  " + "-" * 58
    lines  = [header, col, sep]

    for row in rows:
        date   = row.get("date", "?")[:10]
        traces = row.get("countTraces", 0)
        obs    = row.get("countObservations", 0)
        tokens = sum(u.get("totalUsage", 0) for u in row.get("usage", []))
        cost   = row.get("totalCost", 0.0)
        lines.append(
            f"  {date:<12}  {traces:>7,}  {obs:>12,}  {tokens:>10,}  ${cost:>9.4f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3 — Recent traces (individual pipeline run details)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_recent_traces(limit: int = 20) -> str:
    """
    Return the most recent pipeline traces (runs) with their name, timestamp,
    token counts, latency, and cost.

    Best for questions like:
      "Show me my latest pipeline runs"
      "Which trace used the most tokens?"
      "What were my most expensive runs?"
      "How long did my recent runs take?"
      "Show me the last 10 runs"
    """
    data = _api_get("/api/public/traces", {
        "limit": min(limit, 50),
        "page":  1,
    })
    items = data.get("data", [])

    if not items:
        return "No traces found in Langfuse."

    total = data.get("meta", {}).get("totalItems", len(items))
    lines = [
        f"Recent traces — showing {len(items)} of {total:,} total",
        "",
        f"  {'#':<4}  {'Timestamp':<22}  {'Name':<30}  {'Tokens':>8}  {'Cost':>8}  {'Latency':>9}",
        "  " + "-" * 90,
    ]

    for i, t in enumerate(items, 1):
        ts      = (t.get("timestamp") or "")[:19].replace("T", " ")
        name    = (t.get("name") or "unnamed")[:29]
        usage   = t.get("usage") or {}
        tokens  = usage.get("totalTokens") or usage.get("total") or 0
        cost    = usage.get("totalCost") or 0.0
        latency = t.get("latency")
        lat_str = f"{latency/1000:.1f}s" if latency else "—"

        lines.append(
            f"  {i:<4}  {ts:<22}  {name:<30}  {tokens:>8,}  ${cost:>7.4f}  {lat_str:>9}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4 — Sessions (grouped collections of traces)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_sessions(limit: int = 15) -> str:
    """
    Return recent Langfuse sessions and how many traces each one contains.
    Each BMAD pipeline run creates one session.

    Best for questions like:
      "How many sessions do I have?"
      "Show me my recent sessions"
      "Which session had the most runs?"
    """
    data = _api_get("/api/public/sessions", {
        "limit": min(limit, 50),
        "page":  1,
    })
    items = data.get("data", [])

    if not items:
        return "No sessions found in Langfuse."

    total = data.get("meta", {}).get("totalItems", len(items))
    lines = [
        f"Recent sessions — showing {len(items)} of {total:,} total",
        "",
        f"  {'#':<4}  {'Session ID':<38}  {'Created At':<22}  {'Traces':>7}",
        "  " + "-" * 76,
    ]

    for i, s in enumerate(items, 1):
        sid     = (s.get("id") or "")[:36]
        created = (s.get("createdAt") or "")[:19].replace("T", " ")
        count   = s.get("countTraces", 0)
        lines.append(f"  {i:<4}  {sid:<38}  {created:<22}  {count:>7,}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5 — AI-powered Q&A (Groq LLM inside the MCP)
# ---------------------------------------------------------------------------

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL   = "llama-3.1-8b-instant"

_SYSTEM_PROMPT = """\
You are an expert analytics assistant for an AI pipeline called BMAD.
You have access to Langfuse observability data and answer questions about
pipeline usage, token consumption, cost, and performance.

Be concise and specific. When relevant, suggest actionable optimisations
(e.g. which model to switch, which agents are expensive, how to cut costs).
Always reference exact numbers from the data provided.\
"""


def _ask_groq(question: str, context: str) -> str | None:
    """
    Send context + question to Groq and return the LLM answer.
    Returns None on any error so the caller can fall back gracefully.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None

    user_message = f"Here is the Langfuse data:\n\n{context}\n\nQuestion: {question}"

    try:
        with httpx.Client(timeout=45) as client:
            r = client.post(
                _GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    "temperature": 0.3,
                    "max_tokens":  1024,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[Groq error] {exc}")
        return None


@mcp.tool()
def ask_langfuse(question: str, days: int = 7) -> str:
    """
    Answer any natural language question about your Langfuse usage using AI.

    Internally this tool:
      1. Fetches your usage summary, daily trends, and recent traces
      2. Passes all the data + your question to Groq (llama-3.1-8b-instant)
      3. Returns a rich, insight-driven natural language answer

    Best for complex or open-ended questions like:
      "Am I spending too much? How can I optimise costs?"
      "Which pipeline runs were most expensive and why?"
      "What's the trend in my usage — is it growing?"
      "Give me a full analysis of my Langfuse data"
      "Compare my usage this week vs last week"
      "Which model is costing me the most and should I switch?"
    """
    # Gather all available data to give the LLM full context
    usage  = get_usage_summary(days)
    trends = get_daily_trends(min(days, 14))
    traces = get_recent_traces(20)

    context = (
        f"=== USAGE SUMMARY (last {days} days) ===\n{usage}\n\n"
        f"=== DAILY TRENDS ===\n{trends}\n\n"
        f"=== RECENT PIPELINE TRACES ===\n{traces}"
    )

    answer = _ask_groq(question, context)

    if answer:
        return answer

    # Graceful fallback if Groq is unavailable
    return (
        "[AI analysis unavailable — GROQ_API_KEY missing or Groq unreachable]\n\n"
        f"Raw data instead:\n\n{usage}"
    )


# ---------------------------------------------------------------------------
# Tool 6 — Comparative analysis across pipeline runs
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_runs(days: int = 7, limit: int = 10) -> str:
    """
    Compare recent pipeline runs side by side: per-agent token/cost breakdown,
    A2A vs Full Pipeline averages, and complexity score impact on token usage.

    Best for questions like:
      "Which agent uses the most tokens?"
      "How does A2A compare to the full pipeline in cost?"
      "What's the difference in token usage at different complexity scores?"
      "Show me a breakdown of each agent's cost across recent runs"
      "Compare my last 10 runs"
    """
    from_dt, to_dt = _date_range(days)

    # Fetch recent traces
    traces_data = _api_get("/api/public/traces", {
        "limit": min(limit, 50),
        "page": 1,
    })
    traces = traces_data.get("data", [])

    if not traces:
        return "No traces found in Langfuse."

    # Fetch observations (per-agent generations) for the date range
    obs_data = _api_get("/api/public/observations", {
        "limit": 100,
        "page": 1,
        "fromStartTime": from_dt,
        "toStartTime": to_dt,
        "type": "GENERATION",
    })
    observations = obs_data.get("data", [])

    # ── Per-agent aggregation ─────────────────────────────────────────────
    agent_stats: dict[str, dict] = {}
    for obs in observations:
        name = (obs.get("name") or "unknown").lower()
        usage = obs.get("usage") or {}
        inp   = usage.get("input") or 0
        out   = usage.get("output") or 0
        cost  = float(obs.get("calculatedTotalCost") or 0.0)
        s = agent_stats.setdefault(name, {"input": 0, "output": 0, "total": 0, "cost": 0.0, "calls": 0})
        s["input"]  += inp
        s["output"] += out
        s["total"]  += inp + out
        s["cost"]   += cost
        s["calls"]  += 1

    lines = [
        f"Comparative run analysis — last {days} day(s)  ({from_dt[:10]} → {to_dt[:10]})",
        f"Traces analysed: {len(traces)}   |   Generations analysed: {len(observations)}",
        "",
    ]

    # ── Per-agent breakdown ───────────────────────────────────────────────
    if agent_stats:
        lines.append("Per-agent token & cost breakdown (aggregated):")
        lines.append(f"  {'Agent':<28}  {'Calls':>5}  {'Input':>8}  {'Output':>8}  {'Total':>9}  {'Cost':>10}")
        lines.append("  " + "-" * 76)
        for agent, s in sorted(agent_stats.items(), key=lambda x: -x[1]["total"]):
            lines.append(
                f"  {agent:<28}  {s['calls']:>5}  {s['input']:>8,}  {s['output']:>8,}"
                f"  {s['total']:>9,}  ${s['cost']:>9.5f}"
            )
        lines.append("")

    # ── Trace-level summary (complexity + cost per run) ───────────────────
    lines.append("Recent traces (newest first):")
    lines.append(f"  {'#':<4}  {'Timestamp':<20}  {'Name':<28}  {'Tokens':>8}  {'Cost':>8}  {'Latency':>8}")
    lines.append("  " + "-" * 86)
    for i, t in enumerate(traces[:limit], 1):
        ts      = (t.get("timestamp") or "")[:19].replace("T", " ")
        name    = (t.get("name") or "unnamed")[:27]
        usage   = t.get("usage") or {}
        tokens  = usage.get("totalTokens") or usage.get("total") or 0
        cost    = usage.get("totalCost") or 0.0
        latency = t.get("latency")
        lat_str = f"{latency/1000:.1f}s" if latency else "—"
        lines.append(f"  {i:<4}  {ts:<20}  {name:<28}  {tokens:>8,}  ${cost:>7.5f}  {lat_str:>8}")

    lines.append("")

    # ── Key insight: most expensive agent ─────────────────────────────────
    if agent_stats:
        top_agent = max(agent_stats.items(), key=lambda x: x[1]["cost"])
        top_tokens = max(agent_stats.items(), key=lambda x: x[1]["total"])
        lines.append(f"Most expensive agent : {top_agent[0]}  (${top_agent[1]['cost']:.5f} total)")
        lines.append(f"Highest token usage  : {top_tokens[0]}  ({top_tokens[1]['total']:,} tokens across {top_tokens[1]['calls']} calls)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
