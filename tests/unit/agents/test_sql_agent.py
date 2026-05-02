"""Unit tests for SQL Agent helpers + schema_loader cache.

We intentionally do NOT exercise ``SQLAgent.process`` end-to-end here —
that path drags in the ChatOpenAI factory, the LangGraph runtime, the
schema loader, and three module-level singletons. Instead we cover the
pure helper functions that drive the graph (think-block cleanup, retry
gating, semantic checks, entity-hint rendering, prior-SQL extraction)
and the deterministic ``run_query`` SQL tool. Together these account
for the SQL Agent's actual correctness surface; the graph wiring itself
belongs in contract tests (Phase 5+).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from hse_prom_prog.agents import schema_loader, sql_agent, sql_tools

# ===================================================================== #
# _strip_think — remove <think>...</think> from AI message content
# ===================================================================== #


@pytest.mark.unit
class TestStripThink:
    def test_removes_think_blocks_from_ai_messages(self) -> None:
        msg = AIMessage(
            content="<think>internal reasoning here</think>SELECT * FROM t;",
            id="m1",
        )
        out = sql_agent._strip_think([msg])
        assert "<think>" not in out[0].content
        assert "internal reasoning" not in out[0].content
        assert "SELECT * FROM t" in out[0].content

    def test_preserves_tool_calls_and_id(self) -> None:
        # Cleaning the body must not drop the structured tool call —
        # the LangGraph router reads it on the next iteration. LangChain
        # normalises tool_calls (adds a "type" field), so we compare the
        # meaningful identity fields rather than the whole dict.
        tool_call = {"name": "run_query", "args": {"query": "SELECT 1"}, "id": "tc1"}
        msg = AIMessage(
            content="<think>x</think>",
            tool_calls=[tool_call],
            id="m1",
        )
        out = sql_agent._strip_think([msg])
        assert len(out[0].tool_calls) == 1
        kept = out[0].tool_calls[0]
        assert kept["name"] == "run_query"
        assert kept["args"] == {"query": "SELECT 1"}
        assert kept["id"] == "tc1"
        assert out[0].id == "m1"

    def test_leaves_non_ai_messages_untouched(self) -> None:
        human = HumanMessage(content="<think>not a real think block</think>")
        tool_msg = ToolMessage(content='{"rows": []}', tool_call_id="tc1", name="run_query")
        out = sql_agent._strip_think([human, tool_msg])
        # HumanMessage / ToolMessage bodies are passed through verbatim —
        # only AI content gets the regex.
        assert out[0] is human
        assert out[1] is tool_msg

    def test_ai_message_without_content_passes_through(self) -> None:
        msg = AIMessage(content="", tool_calls=[{"name": "run_query", "args": {}, "id": "x"}])
        out = sql_agent._strip_think([msg])
        # No content → falsy guard triggers, msg is appended unchanged.
        assert out[0] is msg


# ===================================================================== #
# _has_query_result — detect successful tool result vs retry hint
# ===================================================================== #


@pytest.mark.unit
class TestHasQueryResult:
    def test_no_tool_message_returns_false(self) -> None:
        assert sql_agent._has_query_result([HumanMessage(content="q")]) is False

    def test_successful_tool_result_returns_true(self) -> None:
        msgs = [
            HumanMessage(content="q"),
            ToolMessage(content='{"rows": [{"k": 1}]}', tool_call_id="tc", name="run_query"),
        ]
        assert sql_agent._has_query_result(msgs) is True

    def test_error_tool_result_does_not_count(self) -> None:
        msgs = [
            HumanMessage(content="q"),
            ToolMessage(content="SQL Error: undefined table", tool_call_id="tc", name="run_query"),
        ]
        assert sql_agent._has_query_result(msgs) is False

    def test_retry_hint_after_success_blocks_free_mode(self) -> None:
        # A HumanMessage following the last successful tool result is a
        # retry hint (semantic check fired) — model must stay in forced
        # mode and call run_query again, NOT switch to free mode.
        msgs = [
            HumanMessage(content="q"),
            ToolMessage(content='{"rows": []}', tool_call_id="tc", name="run_query"),
            HumanMessage(content="The query returned data, but it is wrong. Rewrite."),
        ]
        assert sql_agent._has_query_result(msgs) is False


# ===================================================================== #
# _should_continue / _after_tools — graph routing
# ===================================================================== #


@pytest.mark.unit
class TestShouldContinue:
    def test_ai_message_with_tool_calls_routes_to_tools(self) -> None:
        msg = AIMessage(content="", tool_calls=[{"name": "run_query", "args": {}, "id": "x"}])
        assert sql_agent._should_continue({"messages": [msg]}) == "tools"

    def test_ai_message_without_tool_calls_routes_to_extract(self) -> None:
        msg = AIMessage(content="some final text")
        assert sql_agent._should_continue({"messages": [msg]}) == "extract"


@pytest.mark.unit
class TestAfterTools:
    def test_below_max_retries_routes_back_to_model(self) -> None:
        assert sql_agent._after_tools({"retry_count": 1}) == "model"

    def test_at_max_retries_routes_to_extract(self) -> None:
        assert sql_agent._after_tools({"retry_count": sql_agent._MAX_RETRIES}) == "extract"

    def test_default_retry_count_routes_to_model(self) -> None:
        # Missing retry_count must default to 0 — not crash.
        assert sql_agent._after_tools({}) == "model"


# ===================================================================== #
# Helpers for SQL extraction
# ===================================================================== #


@pytest.mark.unit
class TestSqlExtractionHelpers:
    def test_get_last_sqls_collects_in_order(self) -> None:
        ai1 = AIMessage(
            content="",
            tool_calls=[{"name": "run_query", "args": {"query": "SELECT 1"}, "id": "1"}],
        )
        ai2 = AIMessage(
            content="",
            tool_calls=[{"name": "run_query", "args": {"query": "SELECT 2"}, "id": "2"}],
        )
        assert sql_agent._get_last_sqls([ai1, ai2]) == ["SELECT 1", "SELECT 2"]

    def test_get_last_sqls_ignores_other_tools(self) -> None:
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "list_tables", "args": {}, "id": "1"}],
        )
        assert sql_agent._get_last_sqls([ai]) == []

    def test_get_user_query_returns_first_human_content(self) -> None:
        msgs = [HumanMessage(content="первый вопрос"), HumanMessage(content="второй")]
        assert sql_agent._get_user_query(msgs) == "первый вопрос"

    def test_get_user_query_empty_when_no_human(self) -> None:
        assert sql_agent._get_user_query([AIMessage(content="x")]) == ""


# ===================================================================== #
# _semantic_check — 4 deterministic mismatch detectors
# ===================================================================== #


@pytest.mark.unit
class TestSemanticCheck:
    def test_metric_question_without_aggregator_with_avg_in_sql(self) -> None:
        hint = sql_agent._semantic_check(
            "velocity команды cthulhu",
            "SELECT feature_teams, AVG(complete_sp) FROM report_agile_dashboard_metrics "
            "WHERE feature_teams ILIKE '%cthulhu%' GROUP BY feature_teams",
        )
        assert hint is not None
        assert "Type A" in hint

    def test_metric_question_with_aggregator_word_passes(self) -> None:
        # User actually asked for an average — AVG is correct, no hint.
        hint = sql_agent._semantic_check(
            "среднее velocity у cthulhu",
            "SELECT feature_teams, AVG(complete_sp) FROM report_agile_dashboard_metrics "
            "WHERE feature_teams ILIKE '%cthulhu%' GROUP BY feature_teams",
        )
        assert hint is None

    def test_minmax_with_team_context_using_aggregate_flagged(self) -> None:
        hint = sql_agent._semantic_check(
            "у какой команды максимальный velocity?",
            "SELECT MAX(complete_sp) FROM report_agile_dashboard_metrics",
        )
        assert hint is not None
        assert "ORDER BY" in hint

    def test_count_tasks_against_metrics_table_flagged(self) -> None:
        sql = (
            "SELECT COUNT(*) FROM report_agile_dashboard_metrics "
            "WHERE feature_teams ILIKE '%cthulhu%'"
        )
        hint = sql_agent._semantic_check("сколько задач у команды cthulhu", sql)
        assert hint is not None
        assert "report_agile_dashboard" in hint

    @pytest.mark.parametrize(
        "sql_filter",
        [
            "WHERE feature_teams ILIKE '%team%'",
            "WHERE feature_teams ILIKE '%team_name%'",
            "WHERE feature_teams ILIKE '%<team_name>%'",
        ],
    )
    def test_placeholder_leak_flagged(self, sql_filter: str) -> None:
        # Catches "team" / "team_name" / "<team_name>" leaking from the
        # prompt template into the actual WHERE clause.
        sql = (
            f"SELECT feature_teams, AVG(scope_drop) "
            f"FROM report_agile_dashboard_metrics {sql_filter}"
        )
        hint = sql_agent._semantic_check("среднее scope drop по командам", sql)
        assert hint is not None
        assert "placeholder" in hint.lower()

    def test_real_team_name_does_not_match_placeholder(self) -> None:
        hint = sql_agent._semantic_check(
            "среднее scope drop у cthulhu",
            "SELECT AVG(scope_drop) FROM report_agile_dashboard_metrics "
            "WHERE feature_teams ILIKE '%cthulhu%'",
        )
        assert hint is None


# ===================================================================== #
# _check_retry — graph-side retry logic
# ===================================================================== #


def _state_with_messages(messages: list, retry: int = 0) -> dict[str, Any]:
    return {"messages": messages, "retry_count": retry}


@pytest.mark.unit
class TestCheckRetry:
    def test_sql_error_increments_retry(self) -> None:
        msgs = [
            HumanMessage(content="q"),
            AIMessage(
                content="",
                tool_calls=[{"name": "run_query", "args": {"query": "SELECT bad"}, "id": "1"}],
            ),
            ToolMessage(
                content="SQL Error: column 'bad' does not exist",
                tool_call_id="1",
                name="run_query",
            ),
        ]
        out = sql_agent._check_retry(_state_with_messages(msgs, retry=0))
        assert out["retry_count"] == 1

    def test_duplicate_sql_adds_change_query_hint(self) -> None:
        # Two AI messages with the SAME SQL, the second producing an error —
        # the retry helper must inject a hint telling the model to change.
        sql = "SELECT bad FROM t"
        msgs = [
            HumanMessage(content="q"),
            AIMessage(
                content="", tool_calls=[{"name": "run_query", "args": {"query": sql}, "id": "1"}]
            ),
            ToolMessage(content="SQL Error: nope", tool_call_id="1", name="run_query"),
            AIMessage(
                content="", tool_calls=[{"name": "run_query", "args": {"query": sql}, "id": "2"}]
            ),
            ToolMessage(content="SQL Error: still nope", tool_call_id="2", name="run_query"),
        ]
        out = sql_agent._check_retry(_state_with_messages(msgs, retry=1))
        assert out["retry_count"] == 2
        assert "messages" in out
        assert "identical to the previous attempt" in out["messages"][0].content

    def test_max_retries_reached_does_not_inject_hint(self) -> None:
        msgs = [
            HumanMessage(content="q"),
            AIMessage(
                content="", tool_calls=[{"name": "run_query", "args": {"query": "S"}, "id": "1"}]
            ),
            ToolMessage(content="SQL Error: x", tool_call_id="1", name="run_query"),
        ]
        out = sql_agent._check_retry(_state_with_messages(msgs, retry=sql_agent._MAX_RETRIES - 1))
        assert out["retry_count"] == sql_agent._MAX_RETRIES
        # No retry hint — we're capped, the next router will hit "extract".
        assert "messages" not in out

    def test_successful_sql_with_semantic_mismatch_triggers_retry(self) -> None:
        # User asked for a metric without aggregator words; SQL used AVG.
        # Semantic check fires → retry++ + hint message injected.
        sql = "SELECT AVG(complete_sp) FROM report_agile_dashboard_metrics"
        msgs = [
            HumanMessage(content="velocity команды cthulhu"),
            AIMessage(
                content="", tool_calls=[{"name": "run_query", "args": {"query": sql}, "id": "1"}]
            ),
            ToolMessage(content='{"rows": [{"avg": 42}]}', tool_call_id="1", name="run_query"),
        ]
        out = sql_agent._check_retry(_state_with_messages(msgs, retry=0))
        assert out["retry_count"] == 1
        assert "messages" in out
        assert "wrong" in out["messages"][0].content.lower()

    def test_successful_sql_without_mismatch_no_retry(self) -> None:
        # Aggregator word present in query → AVG is correct → no retry.
        sql = "SELECT AVG(complete_sp) FROM report_agile_dashboard_metrics"
        msgs = [
            HumanMessage(content="среднее velocity"),
            AIMessage(
                content="", tool_calls=[{"name": "run_query", "args": {"query": sql}, "id": "1"}]
            ),
            ToolMessage(content='{"rows": [{"avg": 42}]}', tool_call_id="1", name="run_query"),
        ]
        out = sql_agent._check_retry(_state_with_messages(msgs, retry=0))
        assert out["retry_count"] == 0
        assert "messages" not in out


# ===================================================================== #
# _collect_successful_sqls — final extraction-phase walk
# ===================================================================== #


@pytest.mark.unit
class TestCollectSuccessfulSqls:
    def test_dedupes_consecutive_repeats(self) -> None:
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "run_query", "args": {"query": "SELECT 1"}, "id": "1"}],
            ),
            ToolMessage(content='{"rows": []}', tool_call_id="1", name="run_query"),
            AIMessage(
                content="",
                tool_calls=[{"name": "run_query", "args": {"query": "SELECT 1"}, "id": "2"}],
            ),
            ToolMessage(content='{"rows": []}', tool_call_id="2", name="run_query"),
        ]
        successful, last, error = sql_agent._collect_successful_sqls(msgs)
        # Same SQL twice → only one entry in successful.
        assert successful == ["SELECT 1"]
        assert last == "SELECT 1"
        assert error == ""

    def test_records_error_when_present(self) -> None:
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "run_query", "args": {"query": "BAD"}, "id": "1"}],
            ),
            ToolMessage(content="SQL Error: nope", tool_call_id="1", name="run_query"),
        ]
        successful, last, error = sql_agent._collect_successful_sqls(msgs)
        assert successful == []
        assert last == "BAD"
        assert "SQL Error" in error

    def test_clears_error_when_later_attempt_succeeds(self) -> None:
        # Error then success → final error must be cleared.
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "run_query", "args": {"query": "BAD"}, "id": "1"}],
            ),
            ToolMessage(content="SQL Error: x", tool_call_id="1", name="run_query"),
            AIMessage(
                content="",
                tool_calls=[{"name": "run_query", "args": {"query": "GOOD"}, "id": "2"}],
            ),
            ToolMessage(content='{"rows": []}', tool_call_id="2", name="run_query"),
        ]
        successful, _, error = sql_agent._collect_successful_sqls(msgs)
        assert successful == ["GOOD"]
        assert error == ""


# ===================================================================== #
# _format_entities_hint — Supervisor → SQL prompt bridge
# ===================================================================== #


@pytest.mark.unit
class TestFormatEntitiesHint:
    def test_empty_entities_returns_empty_string(self) -> None:
        assert sql_agent._format_entities_hint({}) == ""

    def test_single_team_renders_ilike_line(self) -> None:
        out = sql_agent._format_entities_hint({"team_name": "cthulhu"})
        assert "feature_teams ILIKE '%cthulhu%'" in out
        # Header text appears once at the top of the block.
        assert "ОБЯЗАТЕЛЬНО" in out

    def test_team_list_rendered_as_alternatives(self) -> None:
        out = sql_agent._format_entities_hint({"team_name": ["cthulhu", "khorne"]})
        assert "одной из" in out
        assert "'cthulhu'" in out
        assert "'khorne'" in out

    def test_enum_field_rendered_with_equality(self) -> None:
        out = sql_agent._format_entities_hint({"issue_type": "Bug"})
        assert "issue_type = 'Bug'" in out

    def test_metric_field_renders_column_pointer(self) -> None:
        out = sql_agent._format_entities_hint({"metric_name": "velocity"})
        assert "колонка `velocity`" in out

    def test_combined_entities_render_multiple_lines(self) -> None:
        out = sql_agent._format_entities_hint(
            {"team_name": "cthulhu", "issue_type": "Bug", "sprint_name": "26Q1.1"}
        )
        assert "feature_teams ILIKE '%cthulhu%'" in out
        assert "issue_type = 'Bug'" in out
        assert "sprint_name ILIKE '%26Q1.1%'" in out


# ===================================================================== #
# _extract_previous_sql — pull last SQL from conversation context
# ===================================================================== #


@pytest.mark.unit
class TestExtractPreviousSql:
    def test_none_context_returns_none(self) -> None:
        assert sql_agent._extract_previous_sql(None) is None

    def test_returns_most_recent_sql_query(self) -> None:
        ctx = {
            "summary": "",
            "recent_turns": [
                {
                    "role": "assistant",
                    "metadata": {"query_type": "sql", "last_sql": "SELECT old"},
                },
                {
                    "role": "assistant",
                    "metadata": {"query_type": "sql", "last_sql": "SELECT new"},
                },
            ],
            "history_token_count": 0,
            "needs_summarization": False,
        }
        # Walks newest → oldest.
        assert sql_agent._extract_previous_sql(ctx) == "SELECT new"

    def test_skips_non_sql_turns(self) -> None:
        ctx = {
            "summary": "",
            "recent_turns": [
                {
                    "role": "assistant",
                    "metadata": {"query_type": "sql", "last_sql": "SELECT 1"},
                },
                {
                    "role": "assistant",
                    "metadata": {"query_type": "rag", "last_sql": "RAG answer"},
                },
            ],
            "history_token_count": 0,
            "needs_summarization": False,
        }
        # Most recent is rag (not sql) → walks back to the sql turn.
        assert sql_agent._extract_previous_sql(ctx) == "SELECT 1"

    def test_no_sql_turns_returns_none(self) -> None:
        ctx = {
            "summary": "",
            "recent_turns": [{"role": "user", "metadata": {}}],
            "history_token_count": 0,
            "needs_summarization": False,
        }
        assert sql_agent._extract_previous_sql(ctx) is None


# ===================================================================== #
# _parse_tool_result
# ===================================================================== #


@pytest.mark.unit
class TestParseToolResult:
    def test_well_formed_payload_returns_rows(self) -> None:
        rows = [{"k": 1}, {"k": 2}]
        content = json.dumps({"row_count": 2, "rows": rows})
        assert sql_agent._parse_tool_result(content) == rows

    def test_missing_rows_key_returns_empty_list(self) -> None:
        # The defensive default — a payload that doesn't carry rows
        # (sample-only or aggregated) shouldn't crash the workflow.
        assert sql_agent._parse_tool_result('{"row_count": 0}') == []

    def test_invalid_json_returns_empty_list(self) -> None:
        assert sql_agent._parse_tool_result("not json at all") == []


# ===================================================================== #
# Schema loader — TTL cache
# ===================================================================== #


@pytest.fixture(autouse=True)
def _reset_schema_cache() -> None:
    """Clear the module-level cache so per-test state is isolated."""
    schema_loader._cache.clear()


@pytest.mark.unit
class TestSchemaLoaderCache:
    def test_first_call_loads_then_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        load_calls = {"count": 0}

        def fake_load(_engine: Any) -> list:
            load_calls["count"] += 1
            return []

        monkeypatch.setattr(schema_loader, "_load_tables", fake_load)
        monkeypatch.setattr(schema_loader, "render_schema", lambda _: "DDL")

        engine = MagicMock()
        first = schema_loader.get_schema_compact(engine)
        second = schema_loader.get_schema_compact(engine)

        assert first == "DDL"
        assert second == "DDL"
        assert load_calls["count"] == 1  # cached on second call

    def test_cache_expires_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        load_calls = {"count": 0}

        def fake_load(_engine: Any) -> list:
            load_calls["count"] += 1
            return []

        monkeypatch.setattr(schema_loader, "_load_tables", fake_load)
        monkeypatch.setattr(schema_loader, "render_schema", lambda _: "DDL")

        # Freeze time at t=0 for first call, then advance past TTL.
        fake_time = [1000.0]
        monkeypatch.setattr(schema_loader.time, "time", lambda: fake_time[0])

        engine = MagicMock()
        schema_loader.get_schema_compact(engine)
        fake_time[0] += schema_loader._CACHE_TTL + 1
        schema_loader.get_schema_compact(engine)

        assert load_calls["count"] == 2


# ===================================================================== #
# run_query SQL tool — guardrail integration + result shape
# ===================================================================== #


@pytest.mark.unit
class TestRunQueryTool:
    def test_guardrail_blocks_dangerous_sql(self) -> None:
        # The tool wraps SQLGuard from L2 (already covered in Phase 1) —
        # we verify here that a blocked SQL never touches the DB and
        # surfaces an ERROR-prefixed message that the agent can detect.
        db = MagicMock()
        sql_tools.set_db(db)
        result = sql_tools.run_query.invoke({"query": "DROP TABLE users"})
        assert result.startswith("ERROR:")
        assert "blocked" in result.lower()
        db.execute_query.assert_not_called()

    def test_successful_query_returns_sample_payload(self) -> None:
        db = MagicMock()
        rows = [{"issue_key": f"AL-{i}", "summary": f"task {i}"} for i in range(5)]
        db.execute_query.return_value = rows
        sql_tools.set_db(db)

        result = sql_tools.run_query.invoke({"query": "SELECT * FROM report_agile_dashboard"})
        payload = json.loads(result)
        assert payload["row_count"] == 5
        # Tool intentionally returns ≤3 rows in the sample to avoid
        # blowing the agent's context — pin this so the size is loud.
        assert len(payload["sample"]) == 3

    def test_empty_result_returns_empty_payload(self) -> None:
        db = MagicMock()
        db.execute_query.return_value = []
        sql_tools.set_db(db)
        result = sql_tools.run_query.invoke(
            {"query": "SELECT * FROM report_agile_dashboard WHERE 1=0"}
        )
        payload = json.loads(result)
        assert payload == {"row_count": 0, "rows": []}

    def test_sqlalchemy_error_returns_sql_error_string(self) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        db = MagicMock()
        db.execute_query.side_effect = SQLAlchemyError("undefined column 'foo'")
        sql_tools.set_db(db)
        result = sql_tools.run_query.invoke({"query": "SELECT foo FROM report_agile_dashboard"})
        # Prefix is what _check_retry uses to detect failure → must stay stable.
        assert result.startswith("SQL Error:")
        assert "undefined column" in result

    def test_set_db_required_before_invoke(self) -> None:
        # Defensive: invoking without set_db must raise loudly, not pass
        # a None DB to execute_query and segfault sqlalchemy.
        with patch.object(sql_tools, "_db", None), pytest.raises(RuntimeError, match="set_db"):
            sql_tools.run_query.invoke({"query": "SELECT 1 FROM report_agile_dashboard"})
