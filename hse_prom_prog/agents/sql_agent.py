"""SQL Agent: text-to-SQL generation and execution via arctic-7b.

Architecture: entity extraction → LLM generates SQL structure with
:placeholders → values bound programmatically. The LLM handles SQL
structure (SELECT, FROM, WHERE, GROUP BY), code handles exact values.

Schema is loaded from PostgreSQL via schema_loader (not hardcoded).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import OpenAI
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.agents.entity_extractor import ExtractedEntities, extract_entities
from hse_prom_prog.agents.schema_loader import get_known_names, get_schema_compact
from hse_prom_prog.config import settings
from hse_prom_prog.database.connection import DatabaseConnection, get_database

logger = logging.getLogger(__name__)

# ── Prompt (completion-style for text2sql models) ───────────

_COMPLETION_TEMPLATE = """\
{schema}

-- Using valid PostgreSQL, answer the following questions.
-- All table and column names are snake_case (no quotes needed).
-- Use :param placeholders for filter values (never put literals in WHERE).
-- Use ILIKE :param for text filters, = :param for exact match.

-- Entities: team = lpop
-- Question: Все баги команды lpop
SELECT * FROM report_agile_dashboard
  WHERE feature_teams ILIKE :team AND issue_type ILIKE :issue_type LIMIT 100;

-- Entities: team = cthulhu
-- Question: Done total команды cthulhu
SELECT feature_teams, sprint_name, done_total
  FROM report_agile_dashboard_metrics
  WHERE feature_teams ILIKE :team ORDER BY sprint_name;

-- Entities: issue_key = AL-38787
-- Question: Расскажи о задаче AL-38787
SELECT * FROM report_agile_dashboard WHERE issue_key = :issue_key LIMIT 1;

-- Entities: {entities}
-- Question: {question}
SELECT"""


# ── Agent ────────────────────────────────────────────────────


class SQLAgent:
    """Agent that generates SQL via text2sql LLM and executes it.

    Pipeline: extract entities → generate SQL with :placeholders →
    fix identifiers → bind params → execute.
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

    def _generate_sql(self, question: str, entities: ExtractedEntities) -> str:
        """Generate SQL from question + extracted entities (completions API)."""
        self._ensure_db()
        schema = get_schema_compact(self.db.engine)

        prompt = _COMPLETION_TEMPLATE.format(
            schema=schema,
            entities=entities.to_prompt_facts(),
            question=question,
        )

        response = self._sql_client.completions.create(
            model=self._sql_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            stop=[";", "\n\n", "\n--"],
            extra_body={"repetition_penalty": 1.15},
        )
        raw = "SELECT" + response.choices[0].text
        sql = _clean_sql(raw)

        # Fix broken identifiers from arctic tokenizer
        table_names, col_names = get_known_names(self.db.engine)
        sql = _fix_identifiers(sql, table_names, col_names)

        logger.info("[SQL Agent] Generated SQL: %s", sql[:200])
        return sql

    # ── Main entry point ─────────────────────────────────────

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract entities → generate SQL → validate → execute.

        Args:
            state: Workflow state with 'original_query'.

        Returns:
            State update with sql_query, sql_result, error.
        """
        original_query = state.get("original_query", "")
        logger.info("[SQL Agent] Processing: %s", original_query[:80])

        self._ensure_db()

        # 1. Extract entities
        entities = extract_entities(original_query, self.db.engine)
        params = entities.to_sql_params()
        logger.info("[SQL Agent] Entities: %s", entities.to_prompt_facts())

        # 2. Generate SQL
        try:
            sql = self._generate_sql(original_query, entities)
        except Exception as e:
            logger.error("[SQL Agent] SQL generation failed: %s", e)
            return {
                "original_query": original_query,
                "sql_query": None,
                "sql_result": None,
                "error": f"SQL generation failed: {e}",
            }

        # 3. Validate
        if not _is_safe(sql):
            logger.warning("[SQL Agent] Unsafe SQL rejected: %s", sql[:200])
            return {
                "original_query": original_query,
                "sql_query": sql,
                "sql_result": None,
                "error": "Unsafe SQL rejected (only SELECT allowed)",
            }

        # 4. Execute with bound parameters
        try:
            results = self.db.execute_query(sql, params=params or None)
        except SQLAlchemyError as e:
            logger.error("[SQL Agent] Database error: %s | SQL: %s | params: %s", e, sql, params)
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
    sql = sql.strip()

    upper = sql.upper()
    select_idx = upper.find("SELECT")
    if select_idx > 0:
        sql = sql[select_idx:]

    return sql.split(";")[0].strip()


def _normalize(name: str) -> str:
    """Lowercase and remove all non-alphanumeric chars for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _fix_identifiers(sql: str, table_names: list[str], col_names: list[str]) -> str:
    """Fix broken identifiers produced by arctic tokenizer.

    The tokenizer splits underscored names (report_agile_dashboard →
    'report Ag agile Dashboard'). This function matches broken names
    to real ones using normalized comparison.
    """
    lookup: dict[str, str] = {}
    for name in table_names + col_names:
        lookup[_normalize(name)] = name

    # 1. Fix quoted identifiers: "broken name" → real_name
    def _fix_quoted(m: re.Match) -> str:
        broken = m.group(1)
        norm = _normalize(broken)
        if norm in lookup:
            return lookup[norm]
        for key, real in lookup.items():
            if key.startswith(norm) or norm.startswith(key):
                return real
        return m.group(0)

    sql = re.sub(r'"([^"]+)"', _fix_quoted, sql)

    # 2. Fix unquoted broken table names (e.g., report_agile Dashboard)
    for table in sorted(table_names, key=len, reverse=True):
        parts = table.split("_")
        pattern = r"\b" + r"[_\s]?".join(re.escape(p) for p in parts) + r"\b"
        sql = re.sub(pattern, table, sql, flags=re.IGNORECASE)

    return sql


_FORBIDDEN = frozenset(
    {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"}
)


def _is_safe(sql: str) -> bool:
    """Only allow SELECT statements."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word == "SELECT" and first_word not in _FORBIDDEN
