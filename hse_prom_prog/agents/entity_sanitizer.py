"""Entity sanitizer for Supervisor agent.

Normalizes extracted entities to canonical enum values using:
1. Static synonym mapping (Russian/colloquial forms -> canonical English)
2. Dynamic DB validation (optional — checks that value exists in DB)

Design principles:
- Synonym mapping is STATIC (business logic, lives in code)
- Enum validation is DYNAMIC (loaded from DB with TTL cache)
- Unknown values -> None (better to drop than to hallucinate)
- Case-insensitive matching throughout
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Static synonym maps
# ---------------------------------------------------------------------------

_ISSUE_TYPE_SYNONYMS: dict[str, str] = {
    "баг": "Bug",
    "бага": "Bug",
    "баги": "Bug",
    "багов": "Bug",
    "bug": "Bug",
    "сторис": "Story",
    "стори": "Story",
    "story": "Story",
    "stories": "Story",
    "улучшение": "Improvement",
    "улучшения": "Improvement",
    "improvement": "Improvement",
    "эпик": "Epic",
    "эпики": "Epic",
    "epic": "Epic",
    "задача": "Task",
    "task": "Task",
    "саб-таска": "Sub-task",
    "подзадача": "Sub-task",
    "sub-task": "Sub-task",
    "subtask": "Sub-task",
}

_STATUS_SYNONYMS: dict[str, str] = {
    "открыта": "Open",
    "открытые": "Open",
    "open": "Open",
    "в работе": "In Progress",
    "в прогрессе": "In Progress",
    "in progress": "In Progress",
    "сделана": "Done",
    "готово": "Done",
    "done": "Done",
    "закрыта": "Closed",
    "закрытые": "Closed",
    "closed": "Closed",
    "отменена": "Cancelled",
    "отменённые": "Cancelled",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
}

_METRIC_NAME_SYNONYMS: dict[str, str | None] = {
    "скорость": "velocity",
    "velocity": "velocity",
    "процент выполнения": "done_total",
    "done total": "done_total",
    "done_total": "done_total",
    "сброс скоупа": "scope_drop",
    "scope drop": "scope_drop",
    "scope_drop": "scope_drop",
    "цель спринта": "sprint_goal",
    "sprint goal": "sprint_goal",
    "sprint_goal": "sprint_goal",
    "доля отменённого": "cancel_rate",
    "cancel rate": "cancel_rate",
    "cancel_rate": "cancel_rate",
    "initial commitment": "initial_commitment_sp",
    "initial_commitment_sp": "initial_commitment_sp",
    "начальный объём": "initial_commitment_sp",
    "added work": "added_work_sp",
    "added_work_sp": "added_work_sp",
    "добавленная работа": "added_work_sp",
    "final commitment": "final_commitment_sp",
    "final_commitment_sp": "final_commitment_sp",
    "итоговый объём": "final_commitment_sp",
    "complete sp": "complete_sp",
    "complete_sp": "complete_sp",
    "завершённые sp": "complete_sp",
    "метрики": None,  # generic — not a specific metric
    "метрика": None,
}

_SYNONYM_MAPS: dict[str, dict[str, str | None]] = {
    "issue_type": _ISSUE_TYPE_SYNONYMS,  # type: ignore[dict-item]
    "status": _STATUS_SYNONYMS,  # type: ignore[dict-item]
    "metric_name": _METRIC_NAME_SYNONYMS,
}

# ---------------------------------------------------------------------------
# 2. Dynamic DB enum loader (optional layer, TTL cache)
# ---------------------------------------------------------------------------

_db_enums_cache: dict[str, set[str]] = {}
_db_enums_loaded_at: float = 0.0
_DB_ENUM_TTL: float = 3600.0  # 1 hour


def _load_enum_values(engine: Any, queries: dict[str, str]) -> dict[str, set[str]]:
    """Run DISTINCT queries, collect enum values. Empty dict if DB fails."""
    from sqlalchemy import text  # noqa: PLC0415

    out: dict[str, set[str]] = {}
    try:
        with engine.connect() as conn:
            for field, query in queries.items():
                rows = conn.execute(text(query)).fetchall()
                values = {r[0] for r in rows if r[0]}
                out[field] = values
                logger.info(
                    "[EntitySanitizer] Loaded %d %s values from DB",
                    len(values),
                    field,
                )
    except Exception as e:
        logger.warning("[EntitySanitizer] Failed to load enums: %s", e)
    return out


def _load_metric_columns(engine: Any) -> set[str]:
    """Load metric column names from information_schema."""
    from sqlalchemy import text  # noqa: PLC0415

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'report_agile_dashboard_metrics' "
                    "AND data_type IN ('real', 'numeric', 'double precision', "
                    "'integer', 'bigint') "
                    "ORDER BY column_name"
                )
            ).fetchall()
            return {r[0] for r in rows if r[0]}
    except Exception as e:
        logger.warning("[EntitySanitizer] Failed to load metric columns: %s", e)
        return set()


def load_db_enums(engine: Any) -> dict[str, set[str]]:
    """Load distinct enum values from DB for issue_type, status, metric_name."""
    enums = _load_enum_values(
        engine,
        {
            "issue_type": (
                "SELECT DISTINCT issue_type FROM report_agile_dashboard "
                "WHERE issue_type IS NOT NULL"
            ),
            "status": (
                "SELECT DISTINCT issue_status_act FROM report_agile_dashboard "
                "WHERE issue_status_act IS NOT NULL"
            ),
        },
    )
    metrics = _load_metric_columns(engine)
    if metrics:
        enums["metric_name"] = metrics
    return enums


def _get_db_enums(engine: Any | None) -> dict[str, set[str]]:
    """Get DB enums with TTL cache."""
    global _db_enums_cache, _db_enums_loaded_at  # noqa: PLW0603

    if engine is None:
        return {}
    now = time.time()
    if now - _db_enums_loaded_at > _DB_ENUM_TTL or not _db_enums_cache:
        _db_enums_cache = load_db_enums(engine)
        _db_enums_loaded_at = now
    return _db_enums_cache


# ---------------------------------------------------------------------------
# 3. Enum normalization + DB validation
# ---------------------------------------------------------------------------


def normalize_enum_value(field: str, value: str | None) -> str | None:
    """Map raw value to canonical enum using synonyms. None if unknown."""
    if not value or not isinstance(value, str):
        return None

    synonyms = _SYNONYM_MAPS.get(field)
    if synonyms is None:
        return value  # Not an enum field

    normalized = value.strip()
    result = synonyms.get(normalized.lower())
    if result is not None:
        return result

    # Exact canonical match (LLM returned canonical directly)
    canonical_values = {v for v in synonyms.values() if v is not None}
    for canon in canonical_values:
        if canon.lower() == normalized.lower():
            return canon

    logger.debug("[EntitySanitizer] Unknown %s='%s' — dropping", field, value)
    return None


def validate_against_db(
    field: str,
    value: str | None,
    db_enums: dict[str, set[str]],
) -> str | None:
    """Case-insensitive DB-existence check. Returns DB-canonical casing."""
    if value is None:
        return None
    allowed = db_enums.get(field)
    if allowed is None:
        return value
    for db_val in allowed:
        if db_val.lower() == value.lower():
            return db_val
    logger.debug(
        "[EntitySanitizer] %s='%s' not found in DB — keeping anyway",
        field,
        value,
    )
    return value


# ---------------------------------------------------------------------------
# 4. Hallucination filter
# ---------------------------------------------------------------------------

_KNOWN_PLACEHOLDERS: set[str] = {
    "...",
    "…",
    "NULL",
    "null",
    "none",
    "None",
    "N/A",
    "n/a",
    "ABC-123",
    "AL-38787",
    "DATA-1234",
    "RND-55",
    "John Doe",
    "Jane Doe",
    "Cluster A",
    "Cluster B",
    "#1 Q1'26",
    "ASSIGNEE-NAME",
    "CLUSTER-NAME",
    "SPRINT-NAME",
    "METRIC-NAME",
    "ALL-TASKS",
    "ALL-BAIGS",
    "Issue Type Value",
    "Status Value",
    "Assignee Name",
    "Cluster Name",
}

_TEMPLATE_RE = re.compile(r"^[A-Z][A-Z0-9\s\-]{4,}$")
_ISSUE_KEY_FORMAT_RE = re.compile(r"^[A-Za-z]{2,}-\d+$")

_FIELD_MARKERS: dict[str, list[str]] = {
    "sprint_name": ["спринт", "спринте", "спринта", "sprint"],
    "cluster": ["кластер", "кластере", "кластера", "cluster"],
    "assignee": ["исполнител", "assignee"],
}


def _has_contextual_marker(field: str, user_query: str) -> bool:
    """True if a field-specific marker word appears in the query."""
    markers = _FIELD_MARKERS.get(field)
    if markers is None:
        return True
    ql = user_query.lower()
    return any(m in ql for m in markers)


def _word_prefix_match(value: str, query: str, min_prefix: int = 4) -> bool:
    """Russian-morphology-safe word match via prefix."""
    query_words = set(query.lower().split())
    for vw in value.lower().split():
        if len(vw) < min_prefix:
            if vw not in query_words:
                return False
        else:
            prefix = vw[:min_prefix]
            if not any(qw.startswith(prefix) for qw in query_words):
                return False
    return True


def _is_generic_hallucination(value: str, user_query: str) -> bool:
    """Placeholder/template check — applies to all free-text fields."""
    ql = user_query.lower()
    if value in _KNOWN_PLACEHOLDERS and value.lower() not in ql:
        return True
    return bool(_TEMPLATE_RE.match(value) and not _ISSUE_KEY_FORMAT_RE.match(value))


_FIELD_HALLUC_CHECKS: dict[str, Any] = {}


def _check_issue_key(value: str, user_query: str) -> bool:
    if not _ISSUE_KEY_FORMAT_RE.match(value):
        return True
    return value.lower() not in user_query.lower()


def _check_assignee(value: str, user_query: str) -> bool:
    if not _has_contextual_marker("assignee", user_query):
        return True
    return value.lower() not in user_query.lower()


def _check_sprint_or_cluster(field: str, value: str, user_query: str) -> bool:
    if not _has_contextual_marker(field, user_query):
        return True
    return not _word_prefix_match(value, user_query)


def _check_team_name(value: str, user_query: str) -> bool:
    return value.lower() not in user_query.lower()


def is_hallucination(field: str, value: str, user_query: str) -> bool:
    """Detect likely hallucinated values in free-text fields."""
    if _is_generic_hallucination(value, user_query):
        return True
    if field == "issue_key":
        return _check_issue_key(value, user_query)
    if field == "assignee":
        return _check_assignee(value, user_query)
    if field in ("sprint_name", "cluster"):
        return _check_sprint_or_cluster(field, value, user_query)
    if field == "team_name":
        return _check_team_name(value, user_query)
    return False


# ---------------------------------------------------------------------------
# 5. Enum query-presence check
# ---------------------------------------------------------------------------


def _enum_mentioned_in_query(
    field: str,
    canonical_value: str,
    user_query: str,
) -> bool:
    """True if canonical value or any of its synonyms appear in query."""
    ql = user_query.lower()
    forms: set[str] = {canonical_value.lower()}
    if "_" in canonical_value:
        forms.add(canonical_value.replace("_", " ").lower())

    synonyms = _SYNONYM_MAPS.get(field, {})
    for synonym, canon in synonyms.items():
        if canon == canonical_value:
            forms.add(synonym.lower())

    for form in forms:
        pattern = r"(?:^|\s|[,;:!?])" + re.escape(form) + r"(?:\s|$|[,;:!?])"
        if re.search(pattern, ql):
            return True
    return False


# ---------------------------------------------------------------------------
# 6. Main entry points
# ---------------------------------------------------------------------------

_ENUM_FIELDS = frozenset({"issue_type", "status", "metric_name"})
_FREE_TEXT_FIELDS = frozenset({"issue_key", "assignee", "sprint_name", "cluster"})


def _sanitize_team_name(value: Any, user_query: str) -> str | list[str] | None:
    """Handle team_name: string or list of strings, drop hallucinated."""
    if isinstance(value, list):
        cleaned = [
            v
            for v in value
            if isinstance(v, str) and v.strip() and not is_hallucination("team_name", v, user_query)
        ]
        if not cleaned:
            return None
        return cleaned if len(cleaned) > 1 else cleaned[0]
    if isinstance(value, str) and value.strip():
        v = value.strip()
        if not is_hallucination("team_name", v, user_query):
            return v
    return None


def _sanitize_enum_field(
    field: str,
    value: str,
    user_query: str,
    db_enums: dict[str, set[str]],
) -> str | None:
    """Normalize + DB-validate + query-presence check for enum field."""
    normalized = normalize_enum_value(field, value)
    if normalized is None:
        return None
    validated = validate_against_db(field, normalized, db_enums)
    if validated is None:
        return None
    if not _enum_mentioned_in_query(field, validated, user_query):
        logger.debug(
            "[EntitySanitizer] %s='%s' valid but not in query — dropping",
            field,
            validated,
        )
        return None
    return validated


def _sanitize_free_text(field: str, value: Any, user_query: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    return v if not is_hallucination(field, v, user_query) else None


def _sanitize_passthrough(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _sanitize_field(
    field: str,
    value: Any,
    user_query: str,
    db_enums: dict[str, set[str]],
) -> Any | None:
    """Sanitize a single field. Returns cleaned value or None to drop."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if field == "team_name":
        return _sanitize_team_name(value, user_query)
    if field in _ENUM_FIELDS:
        return (
            _sanitize_enum_field(field, value, user_query, db_enums)
            if isinstance(value, str)
            else None
        )
    if field in _FREE_TEXT_FIELDS:
        return _sanitize_free_text(field, value, user_query)
    return _sanitize_passthrough(value)


def sanitize_entities(
    entities: dict[str, Any],
    user_query: str,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Sanitize all entity fields extracted by Supervisor.

    Per-field pipeline:
      - None / empty string -> dropped
      - team_name -> hallucination filter (supports list)
      - enum fields -> synonyms -> DB validation -> query-presence check
      - free-text fields -> hallucination filter
      - unknown field -> passthrough

    Args:
        entities: Raw entities from LLM.
        user_query: Original query for hallucination detection.
        engine: Optional SQLAlchemy engine for DB validation.

    Returns:
        Cleaned entities (only valid values).
    """
    db_enums = _get_db_enums(engine)
    result: dict[str, Any] = {}
    for field, value in entities.items():
        cleaned = _sanitize_field(field, value, user_query, db_enums)
        if cleaned is not None:
            result[field] = cleaned
    return result
