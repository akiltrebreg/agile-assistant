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

# ── Prompt (OmniSQL format for Arctic-Text2SQL-R1) ──────────

_OMNISQL_TEMPLATE = """\
Task Overview: You are a data science expert. Below, you are provided \
with a database schema and a natural language question. Your task is \
to understand the schema and generate a valid SQL query to answer the question.

Database Engine: PostgreSQL

Database Schema:
{schema}

Use :parameter_name placeholders for filter values instead of literals.
Example: SELECT * FROM report_agile_dashboard WHERE feature_teams ILIKE :team

This query will use the following parameters: {entities}

Question:
{question}"""


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
        """Generate SQL from question + extracted entities (OmniSQL format)."""
        self._ensure_db()
        schema = get_schema_compact(self.db.engine)

        prompt = _OMNISQL_TEMPLATE.format(
            schema=schema,
            entities=entities.to_prompt_facts(),
            question=question,
        )

        response = self._sql_client.chat.completions.create(
            model=self._sql_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        logger.info("[SQL Agent] Raw LLM output: %s", raw[:300])

        sql = _parse_answer(raw)

        # Fix broken identifiers from arctic tokenizer
        table_names, col_names = get_known_names(self.db.engine)
        sql = _fix_identifiers(sql, table_names, col_names)

        # Replace broken literal values with bind params from entity extractor
        sql = _replace_literals_with_params(sql, entities)

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


def _strip_fences(text: str) -> str:
    """Remove markdown code fences."""
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_answer(raw: str) -> str:
    """Extract SQL from Arctic-Text2SQL output with <think>/<answer> tags."""
    # 1. <answer> tag
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
    if answer_match:
        sql = _strip_fences(answer_match.group(1).strip())
        if sql.upper().startswith("SELECT"):
            return sql.split(";")[0].strip()

    # 2. Last ```sql block
    code_blocks = re.findall(r"```(?:sql)?\s*\n?(.*?)```", raw, re.DOTALL)
    if code_blocks:
        sql = code_blocks[-1].strip()
        if sql.upper().startswith("SELECT"):
            return sql.split(";")[0].strip()

    # 3. Last SELECT (rfind, not find)
    upper = raw.upper()
    idx = upper.rfind("SELECT")
    if idx >= 0:
        return re.split(r";|\n\n|```", raw[idx:])[0].strip()

    return raw.strip()


def _normalize(name: str) -> str:
    """Lowercase and remove all non-alphanumeric chars for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# SQL keywords that are NOT identifiers
_SQL_KEYWORDS = frozenset(
    {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "in",
        "is",
        "null",
        "like",
        "ilike",
        "as",
        "on",
        "join",
        "left",
        "right",
        "inner",
        "outer",
        "group",
        "by",
        "order",
        "asc",
        "desc",
        "limit",
        "offset",
        "having",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "distinct",
        "between",
        "case",
        "when",
        "then",
        "else",
        "end",
        "exists",
        "union",
        "all",
        "true",
        "false",
        "il",  # arctic sometimes generates "il like" instead of "ilike"
    }
)


def _is_structural_token(s: str) -> bool:
    """Check if token is a SQL keyword, punctuation, literal, or param."""
    lower = s.lower()
    if lower in _SQL_KEYWORDS:
        return True
    if s in ("(", ")", ",", "=", "<", ">", "!", "*"):
        return True
    if s.startswith("'") or s.startswith(":"):
        return True
    return s.replace(".", "", 1).isdigit()


def _similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer:
        return len(shorter) / len(longer)
    # Count common characters in order (LCS-like)
    matches = 0
    j = 0
    for ch in shorter:
        while j < len(longer):
            if longer[j] == ch:
                matches += 1
                j += 1
                break
            j += 1
    return matches / len(longer)


def _resolve_buf(buf: list[str], lookup: dict[str, str]) -> list[str]:
    """Match a buffer of consecutive identifier words against known names."""
    combined = _normalize("".join(buf))
    # Exact match
    if combined in lookup:
        return [lookup[combined]]
    # Fuzzy similarity match — handle tokenizer artifacts like extra "Ag" fragments
    best_score = 0.0
    best_name = None
    for norm_name, real_name in lookup.items():
        score = _similarity(norm_name, combined)
        if score > best_score:
            best_score = score
            best_name = real_name
    min_similarity = 0.8
    if best_score >= min_similarity and best_name:
        return [best_name]
    # Try matching the longest prefix
    for i in range(len(buf), 0, -1):
        partial = _normalize("".join(buf[:i]))
        if partial in lookup:
            return [lookup[partial], *buf[i:]]
    return list(buf)


def _fix_identifiers(sql: str, table_names: list[str], col_names: list[str]) -> str:
    """Fix broken identifiers produced by arctic tokenizer.

    Strategy: tokenize the SQL, find sequences of non-keyword words,
    normalize them (strip spaces/case), and match against known
    table/column names via normalized lookup.
    """
    lookup: dict[str, str] = {}
    for name in table_names + col_names:
        lookup[_normalize(name)] = name

    sql = re.sub(r"\bil\s+like\b", "ILIKE", sql, flags=re.IGNORECASE)

    def _fix_quoted(m: re.Match) -> str:
        norm = _normalize(m.group(1))
        return lookup.get(norm, m.group(1))

    sql = re.sub(r'"([^"]+)"', _fix_quoted, sql)

    tokens = re.split(r"(\s+|[(),=<>!*]|'[^']*'|:[a-z_]+)", sql)

    result: list[str] = []
    buf: list[str] = []

    for token in tokens:
        if not token:
            continue
        stripped = token.strip()
        if not stripped:
            if not buf:
                result.append(token)
            continue

        if _is_structural_token(stripped):
            if buf:
                result.extend(_resolve_buf(buf, lookup))
                buf.clear()
            result.append(token)
        else:
            buf.append(stripped)

    if buf:
        result.extend(_resolve_buf(buf, lookup))

    return " ".join(t for t in result if t.strip())


# Maps entity attr → (column_name, use_ilike)
_ENTITY_COL_MAP: dict[str, tuple[str, bool]] = {
    "issue_key": ("issue_key", False),
    "team": ("feature_teams", True),
    "sprint": ("sprint_name", True),
    "cluster": ("cluster", True),
    "issue_type": ("issue_type", True),
    "status": ("issue_status_act", True),
    "assignee": ("assignee_name", True),
    "unit": ("unit", True),
    "project": ("issue_project", True),
    "priority": ("issue_priority_for_bug", False),
}


def _replace_literals_with_params(sql: str, entities: ExtractedEntities) -> str:
    """Replace broken string literals with bind params from entity extractor.

    The model generates `WHERE issue_key = ' ' 'AL-38787'` (broken by tokenizer).
    This replaces the entire `column = <broken_value>` with `column = :param`
    or `column ILIKE :param` for each extracted entity.
    Skips if column already uses a :bind_param.
    """
    for attr, (col, use_ilike) in _ENTITY_COL_MAP.items():
        val = getattr(entities, attr, None)
        if val is None:
            continue

        op = "ILIKE" if use_ilike else "="
        replacement = f"{col} {op} :{attr}"

        # Skip if already using bind param for this column
        already_bound = rf"\b{re.escape(col)}\b\s*(?:=|ILIKE)\s*:{re.escape(attr)}\b"
        if re.search(already_bound, sql, re.IGNORECASE):
            continue

        # Match: column_name <operator> <broken_literal_value>
        # Value: one or more quoted strings like ' ' 'text', or unquoted word
        pattern = (
            rf"\b{re.escape(col)}\b\s*"
            r"(?:=|ILIKE|LIKE|il\s+like)\s*"
            r"(?:'[^']*'(?:\s*'[^']*')*|[^\s,)]+)"
        )
        sql = re.sub(pattern, replacement, sql, count=1, flags=re.IGNORECASE)

    return sql


_FORBIDDEN = frozenset(
    {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"}
)


def _is_safe(sql: str) -> bool:
    """Only allow SELECT statements."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word == "SELECT" and first_word not in _FORBIDDEN
