"""Unit tests for ``ProfileExtractor.extract``.

Pure list-of-dicts in, dict out. No I/O, no LLM. Bugs to catch:

  * default_team committed on too-thin evidence (2 messages, both same team)
  * 60% threshold off-by-one (50% must NOT commit; 70% must)
  * frequent_metrics returning more than top-3 or wrong order
  * malformed metadata crashing the tally
  * detail_level tie-break direction (should be "detailed" — safer default)
"""

from __future__ import annotations

from typing import Any

import pytest

from agile_assistant.memory.profile_extractor import (
    DEFAULT_TEAM_SHARE_THRESHOLD,
    FREQUENT_METRICS_TOP_N,
    MIN_MESSAGES_FOR_PROFILE,
    ProfileExtractor,
)


def _meta(
    *,
    team: str | None = None,
    metric: str | None = None,
    sprint: str | None = None,
    query_type: str | None = None,
) -> dict[str, Any]:
    """Build a single message metadata dict shaped like what the workflow saves."""
    entities: dict[str, Any] = {}
    if team is not None:
        entities["team_name"] = team
    if metric is not None:
        entities["metric_name"] = metric
    if sprint is not None:
        entities["sprint_name"] = sprint
    out: dict[str, Any] = {}
    if entities:
        out["entities"] = entities
    if query_type is not None:
        out["query_type"] = query_type
    return out


@pytest.fixture
def extractor() -> ProfileExtractor:
    return ProfileExtractor()


# ===================================================================== #
# Min-evidence guard
# ===================================================================== #


@pytest.mark.unit
class TestMinEvidence:
    def test_below_min_messages_returns_empty(self, extractor: ProfileExtractor) -> None:
        # Only 4 entity-bearing messages — under MIN_MESSAGES_FOR_PROFILE (5).
        # Even if all 4 mention the same team (100% share), the extractor
        # must refuse to commit a default — premature commitment is the bug.
        metas = [_meta(team="cthulhu", metric="velocity") for _ in range(4)]
        assert extractor.extract(metas) == {}

    def test_min_messages_constant(self) -> None:
        # Pin the constant so a relaxation is a deliberate decision.
        assert MIN_MESSAGES_FOR_PROFILE == 5

    def test_messages_without_entities_dont_count(self, extractor: ProfileExtractor) -> None:
        # 10 messages but only 3 carry entities — still below threshold.
        metas: list[dict[str, Any]] = [{} for _ in range(7)]
        metas += [_meta(team="cthulhu") for _ in range(3)]
        assert extractor.extract(metas) == {}

    def test_min_messages_param_overridable(self, extractor: ProfileExtractor) -> None:
        # The default can be lowered for tests / future use cases.
        metas = [_meta(team="cthulhu") for _ in range(2)]
        result = extractor.extract(metas, min_messages=2)
        assert result.get("default_team") == "cthulhu"


# ===================================================================== #
# default_team — 60 % threshold
# ===================================================================== #


@pytest.mark.unit
class TestDefaultTeam:
    def test_seventy_percent_share_commits(self, extractor: ProfileExtractor) -> None:
        # 7 of 10 mention "cthulhu" → share=0.7 > 0.6 → committed.
        metas = [_meta(team="cthulhu") for _ in range(7)]
        metas += [_meta(team="khorne") for _ in range(3)]
        assert extractor.extract(metas)["default_team"] == "cthulhu"

    def test_fifty_percent_share_does_not_commit(self, extractor: ProfileExtractor) -> None:
        # 5/10 → share=0.5, strictly below the 0.6 threshold.
        metas = [_meta(team="cthulhu") for _ in range(5)]
        metas += [_meta(team="khorne") for _ in range(5)]
        assert "default_team" not in extractor.extract(metas)

    def test_threshold_value(self) -> None:
        # Pin the constant — if this gets relaxed to 0.5, the workflow will
        # start injecting default_team in cases where the user is genuinely
        # working across multiple teams.
        assert DEFAULT_TEAM_SHARE_THRESHOLD == 0.6

    def test_dominant_team_picked_when_three_way_split(self, extractor: ProfileExtractor) -> None:
        # 8/10 cthulhu, 1 khorne, 1 nurgle → cthulhu wins decisively.
        metas = [_meta(team="cthulhu") for _ in range(8)]
        metas += [_meta(team="khorne"), _meta(team="nurgle")]
        assert extractor.extract(metas)["default_team"] == "cthulhu"


# ===================================================================== #
# frequent_metrics — top-3 with tie-handling
# ===================================================================== #


@pytest.mark.unit
class TestFrequentMetrics:
    def test_top_three_returned_in_frequency_order(self, extractor: ProfileExtractor) -> None:
        # velocity:5, done_total:3, scope_drop:2, cancel_rate:1 → top-3:
        # [velocity, done_total, scope_drop].
        metas = [_meta(metric="velocity") for _ in range(5)]
        metas += [_meta(metric="done_total") for _ in range(3)]
        metas += [_meta(metric="scope_drop") for _ in range(2)]
        metas += [_meta(metric="cancel_rate")]
        # Pad with team mentions so we clear MIN_MESSAGES_FOR_PROFILE.
        metas += [_meta(team="x") for _ in range(5)]
        result = extractor.extract(metas)
        assert result["frequent_metrics"] == ["velocity", "done_total", "scope_drop"]

    def test_top_n_constant(self) -> None:
        assert FREQUENT_METRICS_TOP_N == 3

    def test_fewer_than_three_metrics_returned_as_is(self, extractor: ProfileExtractor) -> None:
        # Only two distinct metrics → list of length 2, not padded with None.
        metas = [_meta(metric="velocity") for _ in range(3)]
        metas += [_meta(metric="done_total") for _ in range(2)]
        metas += [_meta(team="x") for _ in range(5)]
        result = extractor.extract(metas)
        assert result["frequent_metrics"] == ["velocity", "done_total"]


# ===================================================================== #
# frequent_sprints — same shape as metrics
# ===================================================================== #


@pytest.mark.unit
class TestFrequentSprints:
    def test_sprints_aggregated(self, extractor: ProfileExtractor) -> None:
        metas = [_meta(sprint="Спринт 12") for _ in range(4)]
        metas += [_meta(sprint="Спринт 13")]
        metas += [_meta(team="x") for _ in range(5)]
        result = extractor.extract(metas)
        assert result["frequent_sprints"] == ["Спринт 12", "Спринт 13"]


# ===================================================================== #
# preferred_detail_level
# ===================================================================== #


@pytest.mark.unit
class TestDetailLevel:
    def _padded(self, query_types: list[str]) -> list[dict[str, Any]]:
        # Detail-level scoring is independent of entity counts but the
        # MIN_MESSAGES_FOR_PROFILE guard requires entity-bearing messages.
        return [_meta(team="x", query_type=qt) for qt in query_types]

    def test_sql_heavy_user_prefers_brief(self, extractor: ProfileExtractor) -> None:
        result = extractor.extract(self._padded(["sql"] * 6 + ["rag"]))
        assert result["preferred_detail_level"] == "brief"

    def test_rag_heavy_user_prefers_detailed(self, extractor: ProfileExtractor) -> None:
        result = extractor.extract(self._padded(["rag"] * 4 + ["hybrid"] * 2 + ["sql"]))
        assert result["preferred_detail_level"] == "detailed"

    def test_tie_resolves_to_detailed(self, extractor: ProfileExtractor) -> None:
        # Equal counts → "detailed" (the safer default — rather show
        # context the user doesn't need than withhold context they do).
        result = extractor.extract(self._padded(["sql"] * 3 + ["rag"] * 3))
        assert result["preferred_detail_level"] == "detailed"

    def test_no_query_type_yields_no_preference(self, extractor: ProfileExtractor) -> None:
        # If we have entity messages but no query_type tags, detail-level
        # cannot be inferred — key must be absent from preferences.
        metas = [_meta(team="cthulhu") for _ in range(5)]
        result = extractor.extract(metas)
        assert "preferred_detail_level" not in result


# ===================================================================== #
# Robustness — malformed inputs
# ===================================================================== #


@pytest.mark.unit
class TestRobustness:
    def test_non_dict_metadata_skipped(self, extractor: ProfileExtractor) -> None:
        # Real metadata comes from JSONB — a string or None must not crash.
        valid = [_meta(team="cthulhu") for _ in range(5)]
        # type-check intentionally bypassed to mimic JSONB corruption.
        metas: list[Any] = ["broken", None, 42, *valid]
        result = extractor.extract(metas)
        assert result["default_team"] == "cthulhu"

    def test_entities_not_a_dict_skipped(self, extractor: ProfileExtractor) -> None:
        # `entities` was written as a list (legacy / migration bug) — must
        # be ignored, not blow up the extractor.
        metas: list[dict[str, Any]] = [{"entities": ["nope"]} for _ in range(3)]
        metas += [_meta(team="cthulhu") for _ in range(5)]
        result = extractor.extract(metas)
        assert result["default_team"] == "cthulhu"

    def test_empty_input_returns_empty(self, extractor: ProfileExtractor) -> None:
        assert extractor.extract([]) == {}


# ===================================================================== #
# Combined preferences end-to-end
# ===================================================================== #


@pytest.mark.unit
class TestEndToEnd:
    def test_realistic_profile_combines_all_signals(self, extractor: ProfileExtractor) -> None:
        metas = [
            _meta(team="cthulhu", metric="velocity", query_type="sql"),
            _meta(team="cthulhu", metric="velocity", query_type="sql"),
            _meta(team="cthulhu", metric="done_total", query_type="sql"),
            _meta(team="cthulhu", metric="done_total", query_type="hybrid"),
            _meta(team="cthulhu", metric="scope_drop", query_type="rag"),
            _meta(team="khorne", metric="cancel_rate", query_type="sql"),
        ]
        result = extractor.extract(metas)
        assert result["default_team"] == "cthulhu"
        assert result["frequent_metrics"][0] == "velocity"
        assert "scope_drop" in result["frequent_metrics"]
        # 4 sql vs (1 rag + 1 hybrid) = 4 vs 2 → brief.
        assert result["preferred_detail_level"] == "brief"
