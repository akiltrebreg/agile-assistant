"""Load PostgreSQL schema from information_schema + COMMENT ON.

Schema is cached for 10 minutes. New tables/columns are picked up
automatically when the cache expires — no code changes needed.

Requires: COMMENT ON TABLE/COLUMN in init.sql for human-readable descriptions.
Columns without comments still appear in the schema, just without a description.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Whitelist: only these tables are exposed to the model
ALLOWED_TABLES = {
    "report_agile_dashboard",
    "report_agile_dashboard_metrics",
}

_CACHE_TTL = 600  # 10 minutes
_cache: dict[str, tuple[float, str]] = {}


@dataclass
class ColumnInfo:
    """Schema metadata for a single PostgreSQL column.

    Attributes:
        name: Column name.
        data_type: Simplified data type (see ``_TYPE_MAP``).
        is_nullable: ``True`` when the column allows ``NULL``.
        is_pk: ``True`` when the column participates in the primary key.
        comment: ``COMMENT ON COLUMN`` text, or ``None`` if missing.
    """

    name: str
    data_type: str
    is_nullable: bool
    is_pk: bool = False
    comment: str | None = None


@dataclass
class TableInfo:
    """Schema metadata for a single PostgreSQL table.

    Attributes:
        name: Table name.
        comment: ``COMMENT ON TABLE`` text, or ``None`` if missing.
        columns: Ordered column metadata.
    """

    name: str
    comment: str | None = None
    columns: list[ColumnInfo] = field(default_factory=list)


_TYPE_MAP: dict[str, str] = {
    "character varying": "text",
    "timestamp without time zone": "timestamp",
    "timestamp with time zone": "timestamptz",
    "double precision": "real",
    "bigint": "int",
    "integer": "int",
    "numeric": "real",
    "boolean": "bool",
    "text": "text",
    "real": "real",
}


def _simplify_type(pg_type: str) -> str:
    """Map a PostgreSQL data type to its short alias from ``_TYPE_MAP``."""
    return _TYPE_MAP.get(pg_type, pg_type)


def _load_tables(engine: Engine) -> list[TableInfo]:
    """Read table and column structure from ``information_schema`` and ``pg_catalog``."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT t.table_name, pgd.description AS table_comment
                FROM information_schema.tables t
                LEFT JOIN pg_catalog.pg_class c ON c.relname = t.table_name
                LEFT JOIN pg_catalog.pg_description pgd
                    ON pgd.objoid = c.oid AND pgd.objsubid = 0
                WHERE t.table_schema = 'public'
                  AND t.table_type = 'BASE TABLE'
                  AND t.table_name = ANY(:tables)
                ORDER BY t.table_name
            """),
            {"tables": list(ALLOWED_TABLES)},
        ).fetchall()

        tables = {r.table_name: TableInfo(name=r.table_name, comment=r.table_comment) for r in rows}

        rows = conn.execute(
            text("""
                SELECT
                    c.table_name, c.column_name, c.data_type, c.is_nullable,
                    pgd.description AS column_comment,
                    CASE WHEN pk.column_name IS NOT NULL
                         THEN true ELSE false END AS is_pk
                FROM information_schema.columns c
                LEFT JOIN pg_catalog.pg_class cl ON cl.relname = c.table_name
                LEFT JOIN pg_catalog.pg_description pgd
                    ON pgd.objoid = cl.oid
                   AND pgd.objsubid = c.ordinal_position
                LEFT JOIN (
                    SELECT ku.table_name, ku.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage ku
                        ON tc.constraint_name = ku.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                ) pk ON pk.table_name = c.table_name
                   AND pk.column_name = c.column_name
                WHERE c.table_schema = 'public'
                  AND c.table_name = ANY(:tables)
                ORDER BY c.table_name, c.ordinal_position
            """),
            {"tables": list(ALLOWED_TABLES)},
        ).fetchall()

        for r in rows:
            if r.table_name in tables:
                tables[r.table_name].columns.append(
                    ColumnInfo(
                        name=r.column_name,
                        data_type=_simplify_type(r.data_type),
                        is_nullable=r.is_nullable == "YES",
                        is_pk=r.is_pk,
                        comment=r.column_comment,
                    )
                )

    return list(tables.values())


def render_schema(tables: list[TableInfo]) -> str:
    """Render the schema as ``CREATE TABLE`` DDL for the LLM prompt."""
    parts: list[str] = []
    for t in tables:
        desc = f" -- {t.comment}" if t.comment else ""
        lines = [f"CREATE TABLE {t.name} ({desc}"]
        for i, c in enumerate(t.columns):
            comma = "," if i < len(t.columns) - 1 else ""
            comment = f" -- {c.comment}" if c.comment else ""
            lines.append(f"    {c.name} {c.data_type}{comma}{comment}")
        lines.append(");")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def get_schema_compact(engine: Engine) -> str:
    """Return the DDL schema text, cached for 10 minutes.

    Args:
        engine: SQLAlchemy engine used on cache miss.

    Returns:
        ``CREATE TABLE`` DDL covering every whitelisted table.
    """
    now = time.time()
    key = "schema_compact"
    if key in _cache:
        ts, cached = _cache[key]
        age = now - ts
        if age < _CACHE_TTL:
            logger.info("[SchemaLoader] Cache HIT (age=%.1fs, ttl=%ds)", age, _CACHE_TTL)
            return cached
        logger.info("[SchemaLoader] Cache EXPIRED (age=%.1fs)", age)

    t0 = time.time()
    tables = _load_tables(engine)
    result = render_schema(tables)
    _cache[key] = (now, result)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "[SchemaLoader] Cache MISS → loaded %d tables, %d chars in %.1f ms",
        len(tables),
        len(result),
        elapsed_ms,
    )
    return result
