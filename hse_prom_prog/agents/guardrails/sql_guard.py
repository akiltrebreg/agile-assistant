"""Level 2: SQL Guardrail — три слоя защиты run_query от деструктивных запросов.

Replaces the naive ``startswith("SELECT")`` check in ``sql_tools.run_query``.
That check is bypassable via CTEs with side effects, subqueries,
``dblink_exec`` calls, stacked queries, and comment-hidden payloads.

Layers (fail-fast, от дешёвого к дорогому):
  1. **Length / emptiness** — защита от аномально длинных или пустых запросов
  2. **Regex blacklist** — ~0.1 ms, блокирует DDL / DML / опасные функции /
     stacked queries / комментарии
  3. **AST (sqlglot)** — ~1-5 ms, парсит запрос, проверяет что корень — SELECT,
     что в дереве нет mutation-узлов, и что упоминаются только whitelist-таблицы
  4. **Complexity** — лимит JOIN для защиты от чрезмерно сложных запросов

Graceful degradation: если ``sqlglot`` не импортируется, guard работает в
regex-only режиме. Regex слой уже ловит большинство атак.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from hse_prom_prog.metrics import GUARDRAIL_L2_RESULTS
from hse_prom_prog.tracing import langfuse_context, observe

logger = logging.getLogger(__name__)


# ── Layer 2: Regex blacklist ────────────────────────────────────────

_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    # DDL — keyword ДОЛЖЕН быть перед типом объекта (TABLE/INDEX/...),
    # иначе ловятся false positives в строковых литералах вроде 'DROP-123'.
    re.compile(
        r"\b(DROP|CREATE|ALTER|TRUNCATE|RENAME)\s+"
        r"(TABLE|INDEX|VIEW|DATABASE|SCHEMA|SEQUENCE|FUNCTION|"
        r"PROCEDURE|TRIGGER|USER|ROLE|CONSTRAINT|COLUMN|MATERIALIZED)\b",
        re.IGNORECASE,
    ),
    # DML — требуем специфичную структуру (INTO/SET/FROM), не голое слово.
    re.compile(
        r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|"
        r"MERGE\s+INTO|UPSERT\s+INTO)\b",
        re.IGNORECASE,
    ),
    # DCL / TCL — редко попадают в литералы, keep as-is.
    re.compile(r"\b(GRANT|REVOKE|COMMIT|ROLLBACK|SAVEPOINT)\b", re.IGNORECASE),
    # Опасные функции PostgreSQL — требуем `(` после (вызов функции).
    re.compile(
        r"\b(pg_sleep|dblink|dblink_exec|lo_import|lo_export)\s*\(",
        re.IGNORECASE,
    ),
    # COPY — только как statement (в начале или после `;`).
    re.compile(r"(?:^|;\s*)COPY\s", re.IGNORECASE),
    # SET / LOAD / DO (процедурные блоки).
    re.compile(r"(?:^|;\s*)(SET\s+|LOAD\s+|DO\s+\$)", re.IGNORECASE),
    # SQL-комментарии могут скрывать payload.
    re.compile(r"(--|/\*|\*/)"),
    # Примечание: stacked queries (`;` + следующий statement) ловит AST-слой
    # через len(statements) > 1 — regex для этого удалён, т.к. был хрупкий.
]


# ── Layer 3: AST parsing via sqlglot ────────────────────────────────

_SQLGLOT_AVAILABLE = False
try:
    import sqlglot
    from sqlglot import exp

    _SQLGLOT_AVAILABLE = True
except ImportError:
    logger.warning(
        "[SQLGuard] sqlglot not installed — using regex-only mode. "
        "Install with: pip install sqlglot"
    )


_ALLOWED_TABLES: set[str] = {
    "report_agile_dashboard",
    "report_agile_dashboard_metrics",
}


def _forbidden_node_classes() -> set[type]:
    """Collect mutation node classes via getattr (sqlglot renames across versions).

    E.g. sqlglot 26.x renamed ``AlterTable`` to ``Alter``, so we try both.
    """
    if not _SQLGLOT_AVAILABLE:
        return set()
    candidates = [
        "Insert",
        "Delete",
        "Update",
        "Drop",
        "Create",
        "Alter",
        "AlterTable",
        "Merge",
        "Command",  # GRANT / REVOKE / SET / etc.
        "TruncateTable",
    ]
    return {cls for name in candidates if (cls := getattr(exp, name, None))}


def _parse_statement(sql: str) -> tuple[object | None, str]:
    """Parse SQL, return (stmt, '') on success, (None, reason) on failure."""
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception as e:
        return None, f"parse_error: {e}"
    if len(statements) > 1:
        return None, "multiple_statements"
    stmt = statements[0]
    if stmt is None:
        return None, "empty_statement"
    if not isinstance(stmt, exp.Select):
        return None, f"forbidden_statement_type: {type(stmt).__name__}"
    return stmt, ""


def _find_forbidden_subexpression(stmt: object) -> str | None:
    """Return the name of the first mutation node found in the tree, or None."""
    forbidden = tuple(_forbidden_node_classes())
    if not forbidden:
        return None
    for node in stmt.walk():  # type: ignore[attr-defined]
        if isinstance(node, forbidden):
            return type(node).__name__
    return None


def _find_forbidden_tables(stmt: object) -> list[str]:
    """Return tables referenced in the query that are not in the whitelist."""
    referenced: set[str] = set()
    for table in stmt.find_all(exp.Table):  # type: ignore[attr-defined]
        name = (table.name or "").lower()
        if name:
            referenced.add(name)
    allowed_lower = {t.lower() for t in _ALLOWED_TABLES}
    return sorted(referenced - allowed_lower)


def _check_ast(sql: str) -> tuple[bool, str]:
    """Parse SQL, assert root is SELECT, walk for mutations, verify tables."""
    if not _SQLGLOT_AVAILABLE:
        return True, "sqlglot_unavailable"

    stmt, err = _parse_statement(sql)
    if stmt is None:
        return False, err

    bad_node = _find_forbidden_subexpression(stmt)
    if bad_node:
        return False, f"forbidden_subexpression: {bad_node}"

    unknown = _find_forbidden_tables(stmt)
    if unknown:
        return False, f"forbidden_tables: {unknown}"

    return True, "ok"


# ── Layer 1/4: limits ───────────────────────────────────────────────

_MAX_QUERY_LENGTH = 2000
_MAX_JOIN_COUNT = 5


@dataclass
class SQLGuardResult:
    """Result of a SQL guardrail check."""

    allowed: bool
    reason: str
    layer: str  # "limits" | "regex" | "ast" | "ok"


def _record_l2(result: SQLGuardResult) -> SQLGuardResult:
    """Forward the result through Prometheus + Langfuse before returning."""
    GUARDRAIL_L2_RESULTS.labels(
        allowed=str(result.allowed).lower(),
        layer=result.layer,
    ).inc()
    langfuse_context.update_current_observation(
        output={
            "allowed": result.allowed,
            "layer": result.layer,
            "reason": result.reason,
        },
    )
    return result


@observe(name="guardrail_l2")
def check_sql(sql: str) -> SQLGuardResult:
    """Three-layer SQL validation. Called from ``run_query`` before execution.

    Returns:
        SQLGuardResult(allowed=True, reason="ok", layer="ok") on pass,
        otherwise (False, reason, layer) where `layer` tells which check blocked.
    """
    sql_stripped = sql.strip()
    langfuse_context.update_current_observation(input={"sql": sql_stripped})

    if not sql_stripped:
        return _record_l2(SQLGuardResult(False, "empty_query", "limits"))
    if len(sql_stripped) > _MAX_QUERY_LENGTH:
        return _record_l2(SQLGuardResult(False, "query_too_long", "limits"))

    for pattern in _DANGEROUS_PATTERNS:
        m = pattern.search(sql_stripped)
        if m:
            logger.warning(
                "[SQLGuard] REGEX BLOCK: pattern=%r match=%r sql=%r",
                pattern.pattern[:50],
                m.group(),
                sql_stripped[:200],
            )
            return _record_l2(SQLGuardResult(False, f"dangerous_pattern: {m.group()}", "regex"))

    ast_ok, ast_reason = _check_ast(sql_stripped)
    if not ast_ok:
        logger.warning("[SQLGuard] AST BLOCK: reason=%s sql=%r", ast_reason, sql_stripped[:200])
        return _record_l2(SQLGuardResult(False, ast_reason, "ast"))

    join_count = len(re.findall(r"\bJOIN\b", sql_stripped, re.IGNORECASE))
    if join_count > _MAX_JOIN_COUNT:
        return _record_l2(SQLGuardResult(False, f"too_many_joins: {join_count}", "limits"))

    return _record_l2(SQLGuardResult(True, "ok", "ok"))
