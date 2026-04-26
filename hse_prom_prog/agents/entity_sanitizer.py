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

import functools
import logging
import re
import time
from typing import Any

import pymorphy3

from hse_prom_prog.metrics import (
    SANITIZER_ANAPHORA_CARRIES,
    SANITIZER_CORRECTIONS,
    SANITIZER_FALLBACK_EXTRACTIONS,
)
from hse_prom_prog.tracing import langfuse_context, observe

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
        if result != normalized:
            # Layer 1 fired: synonym table actually changed the value
            # (e.g. Russian noun -> canonical English enum). Identity
            # hits (LLM already returned the canonical) don't count.
            SANITIZER_CORRECTIONS.labels(layer="1_synonym").inc()
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
        SANITIZER_CORRECTIONS.labels(layer="3_hallucination").inc()
        return True
    if field == "issue_key":
        flagged = _check_issue_key(value, user_query)
    elif field == "assignee":
        flagged = _check_assignee(value, user_query)
    elif field in ("sprint_name", "cluster"):
        flagged = _check_sprint_or_cluster(field, value, user_query)
    elif field == "team_name":
        flagged = _check_team_name(value, user_query)
    else:
        return False
    if flagged:
        SANITIZER_CORRECTIONS.labels(layer="3_hallucination").inc()
    return flagged


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
        SANITIZER_CORRECTIONS.labels(layer="4_enum_check").inc()
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


# ---------------------------------------------------------------------------
# 6a. Lemmatizer (pymorphy3 singleton, lazy init, cached per word)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")
_morph_analyzer: pymorphy3.MorphAnalyzer | None = None


def _get_morph() -> pymorphy3.MorphAnalyzer:
    """Return the singleton MorphAnalyzer, loading dictionaries on first use."""
    global _morph_analyzer  # noqa: PLW0603
    if _morph_analyzer is None:
        _morph_analyzer = pymorphy3.MorphAnalyzer()
    return _morph_analyzer


@functools.lru_cache(maxsize=4096)
def _lemma(word: str) -> str:
    """Lemma of a single (lowercased) word. Cached because queries repeat."""
    return _get_morph().parse(word)[0].normal_form


def _query_lemmas(query: str) -> frozenset[str]:
    """Lemmatize every alphanumeric token of the query (lowercased)."""
    return frozenset(_lemma(t.lower()) for t in _TOKEN_RE.findall(query))


# Lemma forms of anaphoric pointers. pymorphy3 collapses any case/gender/
# number form (этой, этом, этим / том, той, тех / них, им, ему, …) onto
# one of these — so the list stays small and surface-form coverage is
# automatic. Compare against substring matching, which mis-fired on
# "так" inside "такое" of "Что такое X?".
_ANAPHORA_LEMMAS: frozenset[str] = frozenset(
    {
        # Demonstratives — covers все падежи/рода/числа of эт-, т-, так-.
        "этот",
        "это",
        "тот",
        "такой",
        # Adverbs / particles.
        "ещё",
        "тоже",
        "также",
        "аналогичный",
        "аналогично",
        # Personal pronouns: 3rd-person forms used to refer back.
        "они",
        "он",
        "она",
        "её",
        "свой",
    }
)

# Fields eligible for carry-forward from the prior turn. issue_type / status
# / metric_name / issue_key are excluded intentionally: they are enums or
# unique keys that the user typically re-states explicitly when relevant.
_CARRY_FORWARD_FIELDS: tuple[str, ...] = (
    "team_name",
    "sprint_name",
    "cluster",
    "assignee",
)


def _has_anaphora(user_query: str) -> bool:
    """Return True if any token of the query lemmatizes to an anaphora marker."""
    return bool(_query_lemmas(user_query) & _ANAPHORA_LEMMAS)


def _carry_forward_entities(
    entities: dict[str, Any],
    prev_entities: dict[str, Any] | None,
    user_query: str,
) -> dict[str, Any]:
    """Fill empty ``entities`` fields from ``prev_entities`` on anaphora.

    Called as layer 6 of the sanitizer. Runs *after* layers 1-5 so any
    LLM-introduced value takes precedence; we only fill what Supervisor
    genuinely couldn't extract from the current query.
    """
    if not prev_entities or not _has_anaphora(user_query):
        return entities

    result = dict(entities)
    for field in _CARRY_FORWARD_FIELDS:
        if not result.get(field) and prev_entities.get(field):
            result[field] = prev_entities[field]
            SANITIZER_CORRECTIONS.labels(layer="6_anaphora").inc()
            SANITIZER_ANAPHORA_CARRIES.labels(entity_type=field).inc()
            logger.info(
                "[EntitySanitizer] carry-forward %s=%r from previous turn",
                field,
                prev_entities[field],
            )
    return result


# ---------------------------------------------------------------------------
# 7. Fallback enum extractor — synonyms map as a backstop extractor.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def _synonym_lemmas(syn_key: str) -> frozenset[str]:
    """Lemmatized token set of a synonym key."""
    return frozenset(_lemma(t.lower()) for t in _TOKEN_RE.findall(syn_key))


def _fallback_extract_enum(
    field: str,
    query_lemmas: frozenset[str],
    db_enums: dict[str, set[str]],
) -> str | None:
    """Match query lemmas against the synonym dictionary for ``field``.

    Returns the canonical enum value when every lemma of any synonym key
    is present in the query and DB validation accepts it. Generic synonyms
    that map to ``None`` (e.g. "метрики") are skipped — they don't pin
    down a specific value.
    """
    synonyms = _SYNONYM_MAPS.get(field)
    if synonyms is None:
        return None
    for syn_key, canonical in synonyms.items():
        if canonical is None:
            continue
        s_lemmas = _synonym_lemmas(syn_key)
        if not s_lemmas or not s_lemmas <= query_lemmas:
            continue
        validated = validate_against_db(field, canonical, db_enums)
        if validated is not None:
            return validated
    return None


def _fill_missing_enums_from_query(
    entities: dict[str, Any],
    user_query: str,
    db_enums: dict[str, set[str]],
) -> dict[str, Any]:
    """Layer 7: fill empty enum fields by matching synonym keys against
    lemmatized query tokens.

    Reuses ``_SYNONYM_MAPS`` as a single source of truth — the same
    dictionary that normalizes LLM-supplied values in layer 1 acts as
    a fallback extractor here. Only fields the LLM left empty are touched,
    so explicit extraction always wins.
    """
    out: dict[str, Any] = dict(entities)
    missing = [f for f in _ENUM_FIELDS if not out.get(f)]
    if not missing:
        return out
    query_lemmas = _query_lemmas(user_query)
    if not query_lemmas:
        return out
    for field in missing:
        extracted = _fallback_extract_enum(field, query_lemmas, db_enums)
        if extracted is not None:
            out[field] = extracted
            SANITIZER_CORRECTIONS.labels(layer="7_fallback").inc()
            SANITIZER_FALLBACK_EXTRACTIONS.labels(field=field).inc()
            logger.info(
                "[EntitySanitizer] fallback-extracted %s=%r from query lemmas",
                field,
                extracted,
            )
    return out


@observe(name="entity_sanitizer")
def sanitize_entities(
    entities: dict[str, Any],
    user_query: str,
    engine: Any | None = None,
    prev_entities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sanitize all entity fields extracted by Supervisor.

    Per-field pipeline:
      - None / empty string -> dropped
      - team_name -> hallucination filter (supports list)
      - enum fields -> synonyms -> DB validation -> query-presence check
      - free-text fields -> hallucination filter
      - unknown field -> passthrough
      - [layer 6] carry-forward from ``prev_entities`` when the query
        contains an anaphoric marker (lemma-based) and the field is
        still empty
      - [layer 7] fallback extraction: for each enum field still empty,
        match synonym keys against lemmatized query tokens

    Args:
        entities: Raw entities from LLM.
        user_query: Original query for hallucination detection.
        engine: Optional SQLAlchemy engine for DB validation.
        prev_entities: Entities extracted from the previous user turn —
            used for anaphora-driven carry-forward. ``None`` disables it.

    Returns:
        Cleaned entities (only valid values).
    """
    db_enums = _get_db_enums(engine)
    result: dict[str, Any] = {}
    for field, value in entities.items():
        cleaned = _sanitize_field(field, value, user_query, db_enums)
        if cleaned is not None:
            result[field] = cleaned
    result = _carry_forward_entities(result, prev_entities, user_query)
    result = _fill_missing_enums_from_query(result, user_query, db_enums)

    langfuse_context.update_current_observation(
        input={"raw_entities": entities, "has_prev_entities": bool(prev_entities)},
        output={"sanitized_entities": result},
    )
    return result
