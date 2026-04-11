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
    name: str
    data_type: str
    is_nullable: bool
    is_pk: bool = False
    comment: str | None = None


@dataclass
class TableInfo:
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
    return _TYPE_MAP.get(pg_type, pg_type)


def _load_tables(engine: Engine) -> list[TableInfo]:
    """Read table structure from information_schema + pg_catalog."""
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


def render_schema(tables: list[TableInfo], compact: bool = False) -> str:
    """Render schema as text for the LLM prompt.

    Args:
        tables: Table metadata.
        compact: If True, use CREATE TABLE DDL format (best for text2sql models).
    """
    parts: list[str] = []
    for t in tables:
        if compact:
            col_defs = ", ".join(f'"{c.name}" {c.data_type}' for c in t.columns)
            parts.append(f'CREATE TABLE "{t.name}" ({col_defs})')
        else:
            desc = f" -- {t.comment}" if t.comment else ""
            lines = [f"TABLE {t.name}{desc}", "COLUMNS:"]
            for c in t.columns:
                pk = " PK" if c.is_pk else ""
                comment = f" -- {c.comment}" if c.comment else ""
                lines.append(f"  {c.name} {c.data_type}{pk}{comment}")
            parts.append("\n".join(lines))
    return "\n".join(parts) if compact else "\n\n".join(parts)


def get_schema(engine: Engine) -> str:
    """Return schema text (cached for 10 min)."""
    now = time.time()
    key = "schema"
    if key in _cache:
        ts, cached = _cache[key]
        if now - ts < _CACHE_TTL:
            return cached

    tables = _load_tables(engine)
    result = render_schema(tables)
    _cache[key] = (now, result)
    logger.info("[SchemaLoader] Loaded schema: %d tables, %d chars", len(tables), len(result))
    return result


def get_schema_compact(engine: Engine) -> str:
    """Return compact schema text (no column comments, cached for 10 min)."""
    now = time.time()
    key = "schema_compact"
    if key in _cache:
        ts, cached = _cache[key]
        if now - ts < _CACHE_TTL:
            return cached

    tables = _load_tables(engine)
    result = render_schema(tables, compact=True)
    _cache[key] = (now, result)
    logger.info(
        "[SchemaLoader] Loaded compact schema: %d tables, %d chars", len(tables), len(result)
    )
    return result


def get_known_names(engine: Engine) -> tuple[list[str], list[str]]:
    """Return (table_names, column_names) from schema (cached)."""
    now = time.time()
    key = "known_names"
    if key in _cache:
        ts, cached = _cache[key]
        if now - ts < _CACHE_TTL:
            return cached

    tables = _load_tables(engine)
    table_names = [t.name for t in tables]
    col_names = list({c.name for t in tables for c in t.columns})
    result = (table_names, col_names)
    _cache[key] = (now, result)
    return result
