"""SQL tools for the LangGraph SQL Agent.

Three tools that the LLM can call via tool-calling:
- list_tables: returns available table names
- get_schema: returns CREATE TABLE DDL for requested tables
- run_query: executes a SELECT query and returns results
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.agents.guardrails import check_sql
from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.metrics import (
    SQL_QUERIES_TOTAL,
    SQL_QUERY_DURATION,
    SQL_RESULT_ROWS,
)
from hse_prom_prog.tracing import langfuse_context, observe

logger = logging.getLogger(__name__)

# Module-level DB reference, set by SQLAgent before graph execution
_db: DatabaseConnection | None = None


def set_db(db: DatabaseConnection) -> None:
    """Set the database connection for tools to use."""
    global _db  # noqa: PLW0603
    _db = db


def _get_db() -> DatabaseConnection:
    if _db is None:
        msg = "Database connection not set. Call set_db() first."
        raise RuntimeError(msg)
    return _db


@tool
@observe(name="run_query")
def run_query(query: str) -> str:
    """Execute a SQL SELECT query against the PostgreSQL database.

    Args:
        query: A valid SELECT SQL query. Only SELECT is allowed.

    Returns:
        Query results as text, or an error message if the query fails.
        Use the error message to fix the query and retry.
    """
    # ``@observe`` is the *inner* decorator so it wraps the plain
    # function before LangChain's ``@tool`` turns it into a
    # StructuredTool. When the tool is invoked from langgraph the
    # observed function still runs and contextvars propagate, so the
    # span attaches to the current sql_agent trace.
    db = _get_db()
    sql = query.strip()
    langfuse_context.update_current_observation(input={"sql": sql})

    # Level 2 SQL Guardrail: regex + AST + table whitelist + limits
    guard = check_sql(sql)
    if not guard.allowed:
        logger.warning("[SQL Tools] Blocked by SQLGuard (%s): %s", guard.layer, guard.reason)
        SQL_QUERIES_TOTAL.labels(status="blocked").inc()
        langfuse_context.update_current_observation(
            output={
                "status": "blocked",
                "layer": guard.layer,
                "reason": guard.reason,
            },
        )
        return (
            f"ERROR: Query blocked by security policy "
            f"(layer={guard.layer}, reason={guard.reason}). "
            f"Only SELECT queries on report_agile_dashboard "
            f"and report_agile_dashboard_metrics are allowed."
        )

    logger.info("[SQL Tools] run_query: %s", sql[:200])

    sql_start = time.time()
    try:
        results = db.execute_query(sql)
    except SQLAlchemyError as e:
        duration = time.time() - sql_start
        SQL_QUERY_DURATION.observe(duration)
        SQL_QUERIES_TOTAL.labels(status="error").inc()
        error_msg = f"SQL Error: {e!s}"
        logger.warning("[SQL Tools] %s", error_msg)
        langfuse_context.update_current_observation(
            output={
                "status": "error",
                "error": str(e),
                "duration_ms": round(duration * 1000, 2),
            },
            level="ERROR",
        )
        return error_msg
    duration = time.time() - sql_start
    SQL_QUERY_DURATION.observe(duration)
    SQL_QUERIES_TOTAL.labels(status="success").inc()
    SQL_RESULT_ROWS.observe(len(results))
    langfuse_context.update_current_observation(
        output={
            "status": "success",
            "rows_count": len(results),
            "duration_ms": round(duration * 1000, 2),
        },
    )

    if not results:
        return json.dumps({"row_count": 0, "rows": []})

    # Small sample to LLM — just enough to reason about the result shape.
    # The agent re-executes the SQL for the final output to avoid context overflow.
    sample_rows = 3
    max_cols = 10
    first_row = results[0]
    columns = list(first_row.keys())[:max_cols]
    sample: list[dict[str, Any]] = []
    for row in results[:sample_rows]:
        trimmed: dict[str, Any] = {}
        for k in columns:
            v = row.get(k)
            trimmed[k] = v.isoformat() if hasattr(v, "isoformat") else v
        sample.append(trimmed)
    payload: dict[str, Any] = {
        "row_count": len(results),
        "sample": sample,
        "columns_shown": columns,
        "total_columns": len(first_row.keys()),
    }
    logger.info(
        "[SQL Tools] run_query returned %d rows (LLM sees %d)",
        len(results),
        sample_rows,
    )
    return json.dumps(payload, default=str, ensure_ascii=False)


# All tools for binding to the LLM
SQL_TOOLS: list[Any] = [run_query]
