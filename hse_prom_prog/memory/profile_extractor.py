"""Rule-based extraction of user preferences from message metadata.

Runs over the ``metadata`` JSONB of all messages belonging to a user and
derives a compact ``preferences`` dict. No LLM calls — deterministic,
instant, trivially testable.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TEAM_SHARE_THRESHOLD = 0.6
FREQUENT_METRICS_TOP_N = 3
# Below this many entity-bearing messages we don't have enough signal
# to commit to a default_team or detail-level preference — the first
# few queries are noisy (e.g. one team + one greeting → 100% share).
MIN_MESSAGES_FOR_PROFILE = 5


@dataclass
class _Tally:
    teams: Counter[str] = field(default_factory=Counter)
    metrics: Counter[str] = field(default_factory=Counter)
    sprints: Counter[str] = field(default_factory=Counter)
    query_types: Counter[str] = field(default_factory=Counter)


class ProfileExtractor:
    """Compute a ``preferences`` dict from message metadata."""

    def extract(
        self,
        messages_metadata: list[dict[str, Any]],
        min_messages: int = MIN_MESSAGES_FOR_PROFILE,
    ) -> dict[str, Any]:
        """Derive preferences from a flat list of message metadata dicts.

        Expected per-message keys (all optional): ``entities`` (dict with
        ``team_name`` / ``metric_name`` / ``sprint_name``) and ``query_type``.
        Missing or malformed entries are silently skipped.

        Returns an empty ``preferences`` dict when fewer than
        ``min_messages`` messages carry a non-empty ``entities`` payload —
        a guard against premature commitment (e.g. user's first two
        queries happen to mention the same team, leading the supervisor
        to inject ``default_team`` thereafter on shaky evidence).

        Args:
            messages_metadata: List of message metadata dicts to fold.
            min_messages: Minimum number of entity-bearing messages required
                before any preference is emitted. Defaults to
                ``MIN_MESSAGES_FOR_PROFILE``.

        Returns:
            Preferences dict (possibly empty) suitable for persistence.
        """
        if self._entity_message_count(messages_metadata) < min_messages:
            return {}

        tally = self._tally(messages_metadata)

        preferences: dict[str, Any] = {}
        if top := self._dominant_team(tally.teams):
            preferences["default_team"] = top
        if tally.metrics:
            preferences["frequent_metrics"] = self._top_n(tally.metrics)
        if tally.sprints:
            preferences["frequent_sprints"] = self._top_n(tally.sprints)
        if detail := self._detail_level(tally.query_types):
            preferences["preferred_detail_level"] = detail
        return preferences

    @staticmethod
    def _entity_message_count(messages_metadata: list[dict[str, Any]]) -> int:
        """Count messages whose ``entities`` payload is a non-empty dict.

        Args:
            messages_metadata: Metadata dicts to scan.

        Returns:
            Number of dicts containing a truthy ``entities`` mapping.
        """
        count = 0
        for meta in messages_metadata:
            if not isinstance(meta, dict):
                continue
            entities = meta.get("entities")
            if isinstance(entities, dict) and entities:
                count += 1
        return count

    @staticmethod
    def _tally(messages_metadata: list[dict[str, Any]]) -> _Tally:
        """Aggregate counters of teams, metrics, sprints and query types."""
        tally = _Tally()
        for meta in messages_metadata:
            if not isinstance(meta, dict):
                continue
            entities = meta.get("entities") or {}
            if isinstance(entities, dict):
                if team := entities.get("team_name"):
                    tally.teams[team] += 1
                if metric := entities.get("metric_name"):
                    tally.metrics[metric] += 1
                if sprint := entities.get("sprint_name"):
                    tally.sprints[sprint] += 1
            if qt := meta.get("query_type"):
                tally.query_types[qt] += 1
        return tally

    @staticmethod
    def _dominant_team(teams: Counter[str]) -> str | None:
        """Return the team that dominates above ``DEFAULT_TEAM_SHARE_THRESHOLD``."""
        if not teams:
            return None
        top_team, top_count = teams.most_common(1)[0]
        total = sum(teams.values())
        if total and top_count / total > DEFAULT_TEAM_SHARE_THRESHOLD:
            return top_team
        return None

    @staticmethod
    def _top_n(counter: Counter[str]) -> list[str]:
        """Return the ``FREQUENT_METRICS_TOP_N`` most common items."""
        return [item for item, _ in counter.most_common(FREQUENT_METRICS_TOP_N)]

    @staticmethod
    def _detail_level(query_types: Counter[str]) -> str | None:
        """Pick ``"brief"`` vs ``"detailed"`` based on observed query types."""
        # SQL-heavy users want brief numeric answers; RAG/hybrid users want
        # explanations. Tie → detailed (safer default).
        brief_score = query_types.get("sql", 0)
        detail_score = query_types.get("rag", 0) + query_types.get("hybrid", 0)
        if not brief_score and not detail_score:
            return None
        return "brief" if brief_score > detail_score else "detailed"
