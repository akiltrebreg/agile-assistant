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

from hse_prom_prog.agents.schema_loader import ALLOWED_TABLES, get_schema_compact
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
def list_tables() -> str:
    """List all available tables in the database.

    Returns a comma-separated list of table names that can be queried.
    Call this first to see what tables are available.
    """
    tables = sorted(ALLOWED_TABLES)
    logger.info("[SQL Tools] list_tables → %s", tables)
    return ", ".join(tables)


@tool
def get_schema(table_names: str) -> str:
    """Get the CREATE TABLE DDL schema for the specified tables.

    Args:
        table_names: Comma-separated table names
            (e.g. 'report_agile_dashboard')

    Returns:
        CREATE TABLE statements with column types and comments.
    """
    db = _get_db()
    schema = get_schema_compact(db.engine)
    requested = {t.strip() for t in table_names.split(",")}
    logger.info("[SQL Tools] get_schema for tables: %s", requested)

    # Filter schema to only requested tables
    parts = schema.split("\n\n")
    filtered = []
    for part in parts:
        for table in requested:
            if table in part:
                filtered.append(part)
                break
    return "\n\n".join(filtered) if filtered else schema


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

    # Safety check
    first_word = sql.split()[0].upper() if sql.split() else ""
    forbidden = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"}
    if first_word != "SELECT" or first_word in forbidden:
        return "ERROR: Only SELECT queries are allowed."

    logger.info("[SQL Tools] run_query: %s", sql[:200])

    try:
        results = db.execute_query(sql)
    except SQLAlchemyError as e:
        error_msg = f"SQL Error: {e!s}"
        logger.warning("[SQL Tools] %s", error_msg)
        return error_msg

    if not results:
        return json.dumps({"row_count": 0, "rows": []})

    # Limit rows sent back to LLM for context, but keep full count
    max_rows = 50
    payload = {
        "row_count": len(results),
        "rows": [
            {k: (v if not hasattr(v, "isoformat") else v.isoformat()) for k, v in row.items()}
            for row in results[:max_rows]
        ],
    }
    logger.info("[SQL Tools] run_query returned %d rows", len(results))
    return json.dumps(payload, default=str, ensure_ascii=False)


# All tools for binding to the LLM
SQL_TOOLS: list[Any] = [list_tables, get_schema, run_query]
