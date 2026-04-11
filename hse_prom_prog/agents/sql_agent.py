"""SQL Agent: text-to-SQL generation and execution via arctic-7b.

The model:
- extracts entities from the user question (team, sprint, issue key)
- picks the right table
- generates the SQL query

Schema is loaded from PostgreSQL via schema_loader (not hardcoded).
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.agents.schema_loader import get_schema
from hse_prom_prog.config import settings
from hse_prom_prog.database.connection import DatabaseConnection, get_database

logger = logging.getLogger(__name__)

# ── Prompt ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a PostgreSQL SQL generator. Output ONLY a valid SQL query.

RULES:
1. Think: FROM which table → WHERE filters → SELECT columns
2. Use ILIKE '%value%' for all text filters
3. Add LIMIT 100 unless using COUNT, SUM, AVG, GROUP BY, or ORDER BY ... LIMIT N
4. Output ONLY SQL. No explanations, no markdown."""

_USER_TEMPLATE = """\
SCHEMA:
{schema}

QUESTION: {question}

SQL:"""


# ── Agent ────────────────────────────────────────────────────


class SQLAgent:
    """Agent that generates SQL via text2sql LLM and executes it.

    Uses arctic-7b (or any OpenAI-compatible text2sql model) served
    by a separate vLLM instance (``SQL_VLLM_BASE_URL``).

    Attributes:
        db: Database connection instance.
    """

    def __init__(self, db_connection: DatabaseConnection | None = None) -> None:
        self.db = db_connection
        self._sql_client = OpenAI(
            base_url=settings.sql_vllm_base_url,
            api_key=settings.vllm_api_key,
        )
        self._sql_model = settings.sql_vllm_model
        logger.info(
            "[SQL Agent] Initialized (model=%s, url=%s)",
            self._sql_model,
            settings.sql_vllm_base_url,
        )

    def _ensure_db(self) -> None:
        if not self.db:
            self.db = get_database()
            logger.info("[SQL Agent] Created new database connection")

    # ── SQL generation ───────────────────────────────────────

    def _generate_sql(self, question: str) -> str:
        """Generate SQL from user question using text2sql LLM."""
        self._ensure_db()
        schema = get_schema(self.db.engine)

        response = self._sql_client.chat.completions.create(
            model=self._sql_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(schema=schema, question=question),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        sql = _clean_sql(raw)
        logger.info("[SQL Agent] Generated SQL: %s", sql[:200])
        return sql

    # ── Main entry point ─────────────────────────────────────

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate SQL from question, validate, execute.

        Args:
            state: Workflow state with 'original_query'.

        Returns:
            State update with sql_query, sql_result, error.
        """
        original_query = state.get("original_query", "")
        logger.info("[SQL Agent] Processing: %s", original_query[:80])

        self._ensure_db()

        # 1. Generate SQL
        try:
            sql = self._generate_sql(original_query)
        except Exception as e:
            logger.error("[SQL Agent] SQL generation failed: %s", e)
            return {
                "original_query": original_query,
                "sql_query": None,
                "sql_result": None,
                "error": f"SQL generation failed: {e}",
            }

        # 2. Validate
        if not _is_safe(sql):
            logger.warning("[SQL Agent] Unsafe SQL rejected: %s", sql[:200])
            return {
                "original_query": original_query,
                "sql_query": sql,
                "sql_result": None,
                "error": "Unsafe SQL rejected (only SELECT allowed)",
            }

        # 3. Execute
        try:
            results = self.db.execute_query(sql)
        except SQLAlchemyError as e:
            logger.error("[SQL Agent] Database error: %s", e)
            return {
                "original_query": original_query,
                "sql_query": sql,
                "sql_result": None,
                "error": f"Database error: {e!s}",
            }

        if not results:
            logger.warning("[SQL Agent] Query returned no results")

        logger.info("[SQL Agent] Returned %d row(s)", len(results))
        return {
            "original_query": original_query,
            "sql_query": sql,
            "sql_result": results,
            "error": None,
        }


# ── Utilities ────────────────────────────────────────────────


def _clean_sql(raw: str) -> str:
    """Strip markdown wrappers and extra text around SQL."""
    sql = raw.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1]
    if sql.endswith("```"):
        sql = sql.rsplit("```", 1)[0]
    return sql.split(";")[0].strip()


_FORBIDDEN = frozenset(
    {
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
    }
)


def _is_safe(sql: str) -> bool:
    """Only allow SELECT statements."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word == "SELECT" and first_word not in _FORBIDDEN
