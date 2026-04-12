"""Extract entities from user question for parameterized SQL generation.

Responsibilities:
- Regex extraction for issue keys (AL-12345)
- Fuzzy matching against cached DISTINCT values from DB
- Returns structured entities that the SQL agent uses as bind parameters

The LLM generates SQL structure with :placeholders, this module provides values.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_CACHE_TTL = 600  # 10 minutes
_cache: dict[str, tuple[float, object]] = {}

# Columns to load DISTINCT values for fuzzy matching
_ENTITY_COLUMNS: dict[str, str] = {
    # column_name → table_name
    "feature_teams": "report_agile_dashboard",
    "sprint_name": "report_agile_dashboard",
    "cluster": "report_agile_dashboard",
    "issue_type": "report_agile_dashboard",
    "issue_status_act": "report_agile_dashboard",
    "assignee_name": "report_agile_dashboard",
    "unit": "report_agile_dashboard",
    "issue_project": "report_agile_dashboard",
    "issue_priority_for_bug": "report_agile_dashboard",
}

_ISSUE_KEY_RE = re.compile(r"\b([A-Z]{1,10}-\d+)\b")

# Common Russian/English words that accidentally match DB values.
# "done" matches status "Done", "story" matches type "Story", etc.
_STOP_WORDS = frozenset(
    {
        "done",
        "open",
        "new",
        "all",
        "total",
        "story",
        "bug",
        "sprint",
        "scope",
        "drop",
        "cancel",
        "rate",
        "goal",
        "средний",
        "средняя",
        "среднее",
        "каждой",
        "каждого",
        "каждом",
        "команды",
        "команда",
        "задачи",
        "задач",
        "статус",
        "статусом",
        "кластер",
        "кластере",
        "спринте",
        "спринта",
        "баги",
        "багов",
        "самый",
        "самая",
        "самое",
        "высокий",
        "большой",
        "максимальный",
        "минимальный",
        "размер",
        "количество",
        "покажи",
        "расскажи",
        "информация",
        "данные",
        "метрики",
        "velocity",
        "initial",
        "final",
        "complete",
        "added",
        "progress",
        "resolved",
        "closed",
    }
)


@dataclass
class ExtractedEntities:
    """Entities extracted from user question."""

    issue_key: str | None = None
    team: str | None = None
    sprint: str | None = None
    cluster: str | None = None
    issue_type: str | None = None
    status: str | None = None
    assignee: str | None = None
    unit: str | None = None
    project: str | None = None
    priority: str | None = None
    raw_params: dict[str, str] = field(default_factory=dict)

    # Entity fields that map to SQL params
    _FIELD_MAP: dict[str, bool] = field(default_factory=lambda: {}, repr=False, init=False)

    _ILIKE_FIELDS = frozenset(
        {"team", "sprint", "cluster", "issue_type", "status", "assignee", "unit", "project"}
    )
    _EXACT_FIELDS = frozenset({"issue_key", "priority"})

    def to_prompt_facts(self) -> str:
        """Format as facts for the LLM prompt."""
        facts = [
            f"{attr} = {val}"
            for attr in (*self._ILIKE_FIELDS, *self._EXACT_FIELDS)
            if (val := getattr(self, attr, None))
        ]
        return ", ".join(facts) if facts else "none"

    def to_sql_params(self) -> dict[str, str]:
        """Convert to SQL bind parameters (with ILIKE wildcards)."""
        params: dict[str, str] = {}
        for attr in self._ILIKE_FIELDS:
            if val := getattr(self, attr, None):
                params[attr] = f"%{val}%"
        for attr in self._EXACT_FIELDS:
            if val := getattr(self, attr, None):
                params[attr] = val
        return params


def _load_distinct_values(engine: Engine) -> dict[str, list[str]]:
    """Load DISTINCT values for entity columns (cached)."""
    now = time.time()
    key = "distinct_values"
    if key in _cache:
        ts, cached = _cache[key]
        if now - ts < _CACHE_TTL:
            return cached  # type: ignore[return-value]

    result: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for col, table in _ENTITY_COLUMNS.items():
            rows = conn.execute(
                text(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL")
            ).fetchall()
            result[col] = [str(r[0]).strip() for r in rows if r[0]]

    _cache[key] = (now, result)
    total = sum(len(v) for v in result.values())
    logger.info("[EntityExtractor] Loaded %d distinct values across %d columns", total, len(result))
    return result


def _fuzzy_match(query: str, candidates: list[str], threshold: float = 0.75) -> str | None:
    """Find the best fuzzy match for query among candidates.

    Strict matching to avoid false positives:
    - Only exact substring match if candidate is 4+ chars
    - Skip candidates that are common/stop words
    - Higher fuzzy threshold (0.75)
    """
    query_lower = query.lower()
    min_len = 4

    # Exact substring match — only if candidate is specific enough
    for c in candidates:
        c_lower = c.lower()
        if c_lower in _STOP_WORDS or len(c_lower) < min_len:
            continue
        if c_lower in query_lower:
            return c

    # Fuzzy match on multi-word fragments
    best_score = 0.0
    best_match = None
    words = query_lower.split()
    for c in candidates:
        c_lower = c.lower()
        if c_lower in _STOP_WORDS or len(c_lower) < min_len:
            continue
        best_match, best_score = _best_fragment_match(
            words,
            c_lower,
            best_match,
            best_score,
        )
    return best_match if best_score >= threshold else None


def _best_fragment_match(
    words: list[str],
    candidate: str,
    best_match: str | None,
    best_score: float,
) -> tuple[str | None, float]:
    """Find best matching fragment from words for a candidate."""
    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words) + 1)):
            fragment = " ".join(words[i:j])
            if all(w in _STOP_WORDS for w in fragment.split()):
                continue
            score = SequenceMatcher(None, fragment, candidate).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate
    return best_match, best_score


def extract_entities(question: str, engine: Engine) -> ExtractedEntities:
    """Extract entities from user question using regex + fuzzy matching."""
    entities = ExtractedEntities()

    # 1. Issue key via regex
    key_match = _ISSUE_KEY_RE.search(question)
    if key_match:
        entities.issue_key = key_match.group(1)

    # 2. Fuzzy match against DB values
    distinct = _load_distinct_values(engine)

    # Map: column_name → entity attribute
    col_to_attr = {
        "feature_teams": "team",
        "sprint_name": "sprint",
        "cluster": "cluster",
        "issue_type": "issue_type",
        "issue_status_act": "status",
        "assignee_name": "assignee",
        "unit": "unit",
        "issue_project": "project",
        "issue_priority_for_bug": "priority",
    }

    for col, attr in col_to_attr.items():
        values = distinct.get(col, [])
        if not values:
            continue
        match = _fuzzy_match(question, values)
        if match:
            setattr(entities, attr, match)

    logger.info("[EntityExtractor] Extracted: %s", entities.to_prompt_facts())
    return entities
