"""Unit tests for the 7-layer entity sanitizer.

The sanitizer is the deterministic last-line of defence between LLM-extracted
entities and downstream SQL/RAG agents. A bug here either lets a hallucination
slip into a query (false negative) or wipes a valid value (false positive).
Tests are organised one class per layer, plus an end-to-end class for cross-
layer cases. No layer touches the network; the only optional I/O (the DB
enum loader, layer 2) is exercised against a fake SQLAlchemy engine.

Run subset:
    pytest tests/unit/agents/test_entity_sanitizer.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hse_prom_prog.agents import entity_sanitizer as es

# --------------------------------------------------------------------- #
# Local fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_db_enum_cache() -> None:
    """The DB enum cache lives in module globals — clear it before every test
    so cache state from a previous test cannot leak into the next one."""
    es._db_enums_cache = {}
    es._db_enums_loaded_at = 0.0


@pytest.fixture
def empty_db_enums() -> dict[str, set[str]]:
    """No DB enums loaded — layer 2 becomes a passthrough."""
    return {}


@pytest.fixture
def db_enums_with_canonical() -> dict[str, set[str]]:
    """Mimics what ``load_db_enums`` returns for a populated DB."""
    return {
        "issue_type": {"Bug", "Story", "Task", "Epic", "Improvement", "Sub-task"},
        "status": {"Open", "In Progress", "Done", "Closed", "Cancelled"},
        "metric_name": {"velocity", "done_total", "scope_drop", "complete_sp"},
    }


def _fake_engine_returning(enum_rows: dict[str, list[str]], metric_cols: list[str]):
    """Build a fake SQLAlchemy engine whose ``connect()`` returns rows shaped
    like ``[(value,), (value,), ...]`` for each query. Used by load-cache tests."""
    call_log: list[str] = []

    def _execute(query_obj):
        sql = str(query_obj)
        call_log.append(sql)
        if "issue_type" in sql:
            return MagicMock(fetchall=lambda: [(v,) for v in enum_rows["issue_type"]])
        if "issue_status_act" in sql:
            return MagicMock(fetchall=lambda: [(v,) for v in enum_rows["status"]])
        if "information_schema" in sql:
            return MagicMock(fetchall=lambda: [(v,) for v in metric_cols])
        return MagicMock(fetchall=lambda: [])

    conn = MagicMock()
    conn.execute = _execute
    conn.__enter__ = lambda self: conn
    conn.__exit__ = lambda *_: False

    engine = MagicMock()
    engine.connect = lambda: conn
    engine._call_log = call_log
    return engine


# ===================================================================== #
# Layer 1 — synonym normalization
# ===================================================================== #


@pytest.mark.unit
class TestLayer1Synonyms:
    """``normalize_enum_value`` — static synonym table per enum field."""

    @pytest.mark.parametrize(
        ("field", "raw", "expected"),
        [
            ("issue_type", "баг", "Bug"),
            ("issue_type", "баги", "Bug"),
            ("issue_type", "Bug", "Bug"),
            ("issue_type", "сторис", "Story"),
            ("issue_type", "эпик", "Epic"),
            ("issue_type", "подзадача", "Sub-task"),
            ("status", "в работе", "In Progress"),
            ("status", "in progress", "In Progress"),
            ("status", "сделана", "Done"),
            ("status", "отменена", "Cancelled"),
            ("metric_name", "скорость", "velocity"),
            ("metric_name", "scope drop", "scope_drop"),
            ("metric_name", "завершённые sp", "complete_sp"),
        ],
    )
    def test_known_synonym_maps_to_canonical(self, field: str, raw: str, expected: str) -> None:
        assert es.normalize_enum_value(field, raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["БАГ", "Баг", "bUg", "  баг  "],
    )
    def test_synonym_match_is_case_insensitive_and_trimmed(self, raw: str) -> None:
        assert es.normalize_enum_value("issue_type", raw) == "Bug"

    def test_unknown_synonym_returns_none(self) -> None:
        # Free-text guess that's not in the synonym table at all.
        assert es.normalize_enum_value("issue_type", "incident") is None

    def test_canonical_value_matched_directly_when_not_in_synonym_keys(self) -> None:
        # "Improvement" is a canonical value with no Russian alias key
        # of the same casing; the canonical-fallback loop must find it.
        assert es.normalize_enum_value("issue_type", "Improvement") == "Improvement"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_empty_value_returns_none(self, value: str | None) -> None:
        assert es.normalize_enum_value("issue_type", value) is None

    def test_non_enum_field_passes_through(self) -> None:
        # team_name is not an enum — synonym table is not applied;
        # the original value should be returned unchanged.
        assert es.normalize_enum_value("team_name", "cthulhu") == "cthulhu"

    def test_generic_metric_synonym_maps_to_none(self) -> None:
        # "метрики" / "метрика" are intentionally None in the table
        # — they don't pin down a specific metric.
        assert es.normalize_enum_value("metric_name", "метрики") is None
        assert es.normalize_enum_value("metric_name", "метрика") is None

    def test_non_string_value_returns_none(self) -> None:
        # Sanity guard against an LLM emitting a list / int.
        assert es.normalize_enum_value("issue_type", 42) is None  # type: ignore[arg-type]


# ===================================================================== #
# Layer 2 — DB validation + TTL cache
# ===================================================================== #


@pytest.mark.unit
class TestLayer2DBValidation:
    """``validate_against_db`` and ``_get_db_enums`` (TTL cache)."""

    def test_known_value_returns_db_canonical_casing(
        self, db_enums_with_canonical: dict[str, set[str]]
    ) -> None:
        # User input lowercase, DB stores "In Progress" — DB casing wins.
        assert (
            es.validate_against_db("status", "in progress", db_enums_with_canonical)
            == "In Progress"
        )

    def test_unknown_value_kept_as_is(self, db_enums_with_canonical: dict[str, set[str]]) -> None:
        # If LLM produced a status not in DB we keep it (debug-log only)
        # — sanitizer is conservative; layer 4 is the actual gate.
        assert es.validate_against_db("status", "Triaged", db_enums_with_canonical) == "Triaged"

    def test_none_value_short_circuits(self, db_enums_with_canonical: dict[str, set[str]]) -> None:
        assert es.validate_against_db("status", None, db_enums_with_canonical) is None

    def test_field_missing_from_enum_dict_passes_through(
        self, empty_db_enums: dict[str, set[str]]
    ) -> None:
        # No db_enums loaded → behave as if the layer was disabled.
        assert es.validate_against_db("status", "Open", empty_db_enums) == "Open"

    def test_get_db_enums_returns_empty_when_engine_is_none(self) -> None:
        assert es._get_db_enums(None) == {}

    def test_get_db_enums_caches_within_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _fake_engine_returning(
            enum_rows={"issue_type": ["Bug", "Story"], "status": ["Open"]},
            metric_cols=["velocity", "complete_sp"],
        )
        # First call populates the cache (3 SQL statements expected:
        # issue_type, status, metric columns).
        first = es._get_db_enums(engine)
        assert "issue_type" in first
        assert "status" in first
        assert "metric_name" in first
        first_calls = len(engine._call_log)
        assert first_calls >= 3

        # Second call within TTL → cache hit, no extra SQL.
        es._get_db_enums(engine)
        assert len(engine._call_log) == first_calls

    def test_get_db_enums_reloads_after_ttl_expires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _fake_engine_returning(
            enum_rows={"issue_type": ["Bug"], "status": ["Open"]},
            metric_cols=["velocity"],
        )
        # Freeze time at t=0 for first call, then jump past the TTL.
        fake_time = [0.0]
        monkeypatch.setattr(es.time, "time", lambda: fake_time[0])
        es._get_db_enums(engine)
        first_calls = len(engine._call_log)

        fake_time[0] = es._DB_ENUM_TTL + 1.0
        es._get_db_enums(engine)
        assert len(engine._call_log) > first_calls

    def test_load_db_enums_swallows_db_errors(self) -> None:
        # If the connect() raises, _load_enum_values returns {} (logged).
        # The caller (load_db_enums) then returns whatever _load_metric_columns
        # produced — which is also {} on failure. Sanitizer must keep working.
        broken = MagicMock()
        broken.connect.side_effect = RuntimeError("db down")
        result = es.load_db_enums(broken)
        assert result == {}


# ===================================================================== #
# Layer 3 — hallucination filter
# ===================================================================== #


@pytest.mark.unit
class TestLayer3Hallucinations:
    """``is_hallucination`` per field — placeholders, templates, query presence."""

    @pytest.mark.parametrize(
        "placeholder",
        ["John Doe", "Jane Doe", "Cluster A", "...", "N/A", "TEAM-NAME", "ASSIGNEE-NAME"],
    )
    def test_known_placeholder_not_in_query_is_flagged(self, placeholder: str) -> None:
        # The user query mentions nothing — a value drawn from the
        # KNOWN_PLACEHOLDERS list is almost certainly LLM filler.
        assert es.is_hallucination("team_name", placeholder, "покажи задачи") is True

    def test_known_placeholder_actually_in_query_is_kept(self) -> None:
        # Edge: if the user *literally* typed the placeholder, trust them.
        assert es.is_hallucination("team_name", "John Doe", "John Doe in team") is False

    def test_template_pattern_caught_for_uppercase_value(self) -> None:
        # "TEAM-A" matches the all-caps template pattern but is not a
        # valid issue-key format → flagged as a generic placeholder.
        assert es.is_hallucination("team_name", "TEAM-A", "что у этой команды") is True

    def test_template_pattern_does_not_flag_real_issue_key(self) -> None:
        # "AL-12345" matches the template_re too but ALSO matches the
        # issue-key format → the template guard is bypassed.
        assert es.is_hallucination("issue_key", "AL-12345", "расскажи про al-12345") is False

    def test_issue_key_missing_from_query_is_hallucinated(self) -> None:
        assert es.is_hallucination("issue_key", "AL-99999", "расскажи про AL-1") is True

    def test_issue_key_with_invalid_format_is_hallucinated(self) -> None:
        assert es.is_hallucination("issue_key", "garbage", "расскажи про garbage") is True

    def test_assignee_requires_marker_word_in_query(self) -> None:
        # No "исполнитель"/"assignee" marker → flagged regardless of presence.
        assert es.is_hallucination("assignee", "иванов", "статусы по иванову") is True

    def test_assignee_with_marker_and_value_in_query_kept(self) -> None:
        assert es.is_hallucination("assignee", "иванов", "исполнитель иванов") is False

    def test_sprint_name_with_anchor_word_and_prefix_match_kept(self) -> None:
        # Single-word value — `_word_prefix_match` requires *every* word
        # of the value to match a query-word prefix; multi-word sprint
        # names are tested separately in TestLayer5PrefixMatch.
        assert (
            es.is_hallucination(
                "sprint_name",
                "Мандариновый",
                "что в спринте мандариновый",
            )
            is False
        )

    def test_sprint_name_without_anchor_word_is_hallucinated(self) -> None:
        # "спринт"/"sprint" not in query → flagged.
        assert (
            es.is_hallucination(
                "sprint_name",
                "Мандариновый",
                "что у мандаринов",
            )
            is True
        )

    def test_team_name_value_must_appear_in_query(self) -> None:
        assert es.is_hallucination("team_name", "cthulhu", "задачи cthulhu") is False
        assert es.is_hallucination("team_name", "cthulhu", "задачи команды") is True

    def test_unknown_field_is_passthrough(self) -> None:
        # Defensive default for fields we haven't taught the filter about.
        assert es.is_hallucination("custom_field", "anything", "query") is False


# ===================================================================== #
# Layer 4 — enum query-presence check
# ===================================================================== #


@pytest.mark.unit
class TestLayer4EnumQueryPresence:
    """``_enum_mentioned_in_query`` — value (or any synonym) actually mentioned."""

    def test_canonical_value_mentioned_directly(self) -> None:
        assert es._enum_mentioned_in_query("status", "Open", "show me Open tickets")

    def test_synonym_in_russian_matches_english_canonical(self) -> None:
        assert es._enum_mentioned_in_query("status", "In Progress", "что в работе сейчас")

    def test_underscore_canonical_also_matches_space_form(self) -> None:
        # canonical "scope_drop" → query has "scope drop" → match.
        assert es._enum_mentioned_in_query(
            "metric_name", "scope_drop", "покажи scope drop за квартал"
        )

    def test_value_not_mentioned_anywhere_returns_false(self) -> None:
        assert not es._enum_mentioned_in_query("status", "Open", "что в работе сейчас")

    def test_word_boundary_prevents_substring_false_positive(self) -> None:
        # "open" must not match inside "opening".
        assert not es._enum_mentioned_in_query("status", "Open", "opening hours")

    def test_punctuation_counts_as_word_boundary(self) -> None:
        # Trailing comma / question mark must still allow a match.
        assert es._enum_mentioned_in_query("status", "Open", "статусы: open, done?")


# ===================================================================== #
# Layer 5 — Russian-morphology prefix match
# ===================================================================== #


@pytest.mark.unit
class TestLayer5PrefixMatch:
    """``_word_prefix_match`` — used by sprint/cluster hallucination check."""

    def test_long_word_matches_via_prefix(self) -> None:
        # "Мандариновый" (12 chars) — first 4 chars "манд" must appear
        # as the prefix of some query word.
        assert es._word_prefix_match("Мандариновый", "что в мандариновом спринте")

    def test_prefix_too_short_for_match(self) -> None:
        assert not es._word_prefix_match("Мандариновый", "что в спринте манго")

    def test_short_word_requires_exact_match(self) -> None:
        # "API" is 3 chars (< min_prefix=4) → must appear verbatim.
        assert es._word_prefix_match("API", "что у api")
        assert not es._word_prefix_match("API", "что у apiserver")

    def test_multiword_value_requires_all_words_to_match(self) -> None:
        # Both "logistics" and "team" must each find a match.
        assert es._word_prefix_match("Logistics Team", "logistics team report")
        # Drop one — "team" is not present in query → False.
        assert not es._word_prefix_match("Logistics Team", "logistics report")

    def test_cluster_partial_prefix(self) -> None:
        # "Logi" is the user's shorthand → matches "Logistics" via prefix.
        assert es._word_prefix_match("Logistics", "logi cluster")


# ===================================================================== #
# Layer 6 — anaphora carry-forward
# ===================================================================== #


@pytest.mark.unit
class TestLayer6AnaphoraCarryForward:
    """``_carry_forward_entities`` — fill empty fields from previous turn."""

    @pytest.mark.parametrize(
        "query",
        [
            "что у этой команды",
            "у этой команды какие задачи",
            "та же команда сейчас",
            "у них статус",
            "ещё одну задачу",
            "также покажи метрики",
        ],
    )
    def test_anaphora_marker_triggers_carry_forward(self, query: str) -> None:
        prev = {"team_name": "cthulhu"}
        result = es._carry_forward_entities({}, prev, query)
        assert result == {"team_name": "cthulhu"}

    def test_no_anaphora_marker_means_no_carry_forward(self) -> None:
        prev = {"team_name": "cthulhu"}
        # Plain content query, no demonstratives / pronouns / "тоже"/"также".
        result = es._carry_forward_entities({}, prev, "покажи задачи команды")
        assert result == {}

    def test_existing_value_takes_precedence_over_prev(self) -> None:
        # Layer 6 must NOT overwrite a value the LLM/Supervisor already
        # extracted from the current query.
        prev = {"team_name": "cthulhu"}
        result = es._carry_forward_entities({"team_name": "khorne"}, prev, "у этой команды что")
        assert result["team_name"] == "khorne"

    def test_prev_entities_none_is_identity(self) -> None:
        result = es._carry_forward_entities({"team_name": "cthulhu"}, None, "у этой команды")
        assert result == {"team_name": "cthulhu"}

    def test_carry_forward_only_for_whitelisted_fields(self) -> None:
        # issue_type is intentionally NOT in _CARRY_FORWARD_FIELDS — enums
        # should not silently bleed across turns.
        prev = {"issue_type": "Bug", "team_name": "cthulhu"}
        result = es._carry_forward_entities({}, prev, "у этой команды")
        assert "issue_type" not in result
        assert result["team_name"] == "cthulhu"

    def test_multiple_fields_carried_at_once(self) -> None:
        prev = {"team_name": "cthulhu", "sprint_name": "Мандариновый"}
        result = es._carry_forward_entities({}, prev, "у этой команды и спринта")
        assert result == {"team_name": "cthulhu", "sprint_name": "Мандариновый"}

    def test_empty_prev_does_not_crash(self) -> None:
        result = es._carry_forward_entities({}, {}, "у этой команды")
        assert result == {}


# ===================================================================== #
# Layer 7 — fallback enum extractor
# ===================================================================== #


@pytest.mark.unit
class TestLayer7FallbackExtraction:
    """``_fill_missing_enums_from_query`` — synonym table as backstop extractor."""

    def test_metric_extracted_from_query_when_llm_left_empty(
        self, db_enums_with_canonical: dict[str, set[str]]
    ) -> None:
        result = es._fill_missing_enums_from_query(
            {}, "покажи velocity команды", db_enums_with_canonical
        )
        assert result["metric_name"] == "velocity"

    def test_issue_type_extracted_from_russian_synonym(
        self, db_enums_with_canonical: dict[str, set[str]]
    ) -> None:
        result = es._fill_missing_enums_from_query(
            {}, "сколько багов в спринте", db_enums_with_canonical
        )
        assert result["issue_type"] == "Bug"

    def test_existing_enum_value_is_not_overwritten(
        self, db_enums_with_canonical: dict[str, set[str]]
    ) -> None:
        result = es._fill_missing_enums_from_query(
            {"issue_type": "Story"},
            "сколько багов в спринте",
            db_enums_with_canonical,
        )
        # LLM already produced "Story" → fallback must not flip it.
        assert result["issue_type"] == "Story"

    def test_empty_query_yields_no_extraction(
        self, db_enums_with_canonical: dict[str, set[str]]
    ) -> None:
        result = es._fill_missing_enums_from_query({}, "", db_enums_with_canonical)
        assert result == {}

    def test_generic_synonym_with_none_canonical_skipped(
        self, db_enums_with_canonical: dict[str, set[str]]
    ) -> None:
        # "метрики" maps to None in the synonym table — must not be
        # promoted to a metric_name value.
        result = es._fill_missing_enums_from_query({}, "покажи метрики", db_enums_with_canonical)
        assert "metric_name" not in result

    def test_db_validation_is_conservative_in_fallback(self) -> None:
        # validate_against_db is intentionally conservative: it normalises
        # casing for known values but keeps unknown ones (debug-log only).
        # So layer 7 still injects a synonym-derived canonical even when
        # the DB enum set doesn't list it. Layer 4 is the hard gate for
        # LLM-supplied values, but layer 7 bypasses it (the value was
        # extracted from the query lemmas, so it's by construction "in
        # the query"). Pinning this behaviour makes accidental changes
        # to validate_against_db visible.
        narrow_enums = {"metric_name": {"velocity"}}
        result = es._fill_missing_enums_from_query({}, "покажи complete sp", narrow_enums)
        assert result["metric_name"] == "complete_sp"

    def test_db_canonical_casing_wins_in_fallback(self) -> None:
        # When the value IS in the DB enum set, validate_against_db
        # rewrites to DB casing — and fallback returns the rewritten value.
        weird_casing_enums = {"metric_name": {"VELOCITY"}}
        result = es._fill_missing_enums_from_query({}, "покажи velocity", weird_casing_enums)
        assert result["metric_name"] == "VELOCITY"

    def test_no_db_enums_means_synonym_canonical_passes(
        self, empty_db_enums: dict[str, set[str]]
    ) -> None:
        # No DB layer loaded → validate_against_db is identity → fallback works.
        result = es._fill_missing_enums_from_query({}, "покажи velocity", empty_db_enums)
        assert result["metric_name"] == "velocity"


# ===================================================================== #
# End-to-end — sanitize_entities through all layers
# ===================================================================== #


@pytest.mark.unit
class TestEndToEndSanitize:
    """``sanitize_entities`` — full pipeline orchestration."""

    def test_all_empty_in_all_empty_out(self) -> None:
        assert es.sanitize_entities({}, "any query") == {}

    def test_hallucinated_team_name_dropped(self) -> None:
        result = es.sanitize_entities({"team_name": "John Doe"}, "покажи задачи")
        assert "team_name" not in result

    def test_team_name_list_with_one_real_one_fake(self) -> None:
        # "John Doe" is a placeholder → dropped. "cthulhu" appears in
        # query → kept. Single survivor is unwrapped from the list.
        result = es.sanitize_entities({"team_name": ["John Doe", "cthulhu"]}, "задачи cthulhu")
        assert result["team_name"] == "cthulhu"

    def test_team_name_list_all_hallucinated_dropped(self) -> None:
        result = es.sanitize_entities({"team_name": ["John Doe", "Jane Doe"]}, "покажи задачи")
        assert "team_name" not in result

    def test_synonym_normalised_then_kept_when_in_query(self) -> None:
        # "баги" in entities (LLM verbatim) + "баги" in query → normalised
        # to "Bug" by layer 1, then layer 4 sees "баги" synonym in query.
        result = es.sanitize_entities({"issue_type": "баги"}, "сколько багов в спринте")
        assert result["issue_type"] == "Bug"

    def test_unknown_field_passes_through_unchanged(self) -> None:
        # Sanitizer should not silently drop fields it doesn't recognise.
        result = es.sanitize_entities({"custom_label": "important"}, "any query")
        assert result["custom_label"] == "important"

    def test_anaphora_then_fallback_extraction_combined(self) -> None:
        # Anaphora carries team_name from prev turn, fallback extracts
        # metric_name from "velocity" in the current query → both layers
        # contribute to the final result.
        result = es.sanitize_entities(
            entities={},
            user_query="покажи velocity у этой команды",
            prev_entities={"team_name": "cthulhu"},
        )
        assert result["team_name"] == "cthulhu"
        assert result["metric_name"] == "velocity"

    def test_engine_none_skips_db_validation(self) -> None:
        # Without an engine, layer 2 is bypassed but every other layer
        # still runs — this is the production fallback when DB is down.
        result = es.sanitize_entities(
            {"issue_type": "баг", "team_name": "cthulhu"},
            "баг у команды cthulhu",
            engine=None,
        )
        assert result["issue_type"] == "Bug"
        assert result["team_name"] == "cthulhu"
