"""SQL tools for the LangGraph SQL Agent.

Three tools that the LLM can call via tool-calling:
- list_tables: returns available table names
- get_schema: returns CREATE TABLE DDL for requested tables
- run_query: executes a SELECT query and returns results
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.agents.guardrails import check_sql
from hse_prom_prog.database.connection import DatabaseConnection

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
def run_query(query: str) -> str:
    """Execute a SQL SELECT query against the PostgreSQL database.

    Args:
        query: A valid SELECT SQL query. Only SELECT is allowed.

    Returns:
        Query results as text, or an error message if the query fails.
        Use the error message to fix the query and retry.
    """
    db = _get_db()
    sql = query.strip()

    # Level 2 SQL Guardrail: regex + AST + table whitelist + limits
    guard = check_sql(sql)
    if not guard.allowed:
        logger.warning("[SQL Tools] Blocked by SQLGuard (%s): %s", guard.layer, guard.reason)
        return (
            f"ERROR: Query blocked by security policy "
            f"(layer={guard.layer}, reason={guard.reason}). "
            f"Only SELECT queries on report_agile_dashboard "
            f"and report_agile_dashboard_metrics are allowed."
        )

    logger.info("[SQL Tools] run_query: %s", sql[:200])

    try:
        results = db.execute_query(sql)
    except SQLAlchemyError as e:
        error_msg = f"SQL Error: {e!s}"
        logger.warning("[SQL Tools] %s", error_msg)
        return error_msg

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
