"""Unit tests for ``SupervisorAgent.process`` — fast path + slow path.

Two routing modes:

  * **fast path** — regex finds an issue key (e.g. AL-1234) → no LLM call,
    intent fixed at ``task``, query_type at ``sql``.
  * **slow path** — LLM classifies; output is JSON-parsed, sanitised
    through the entity sanitiser (real impl, no mock — Phase 1 covers it),
    then post-processed by 7 deterministic rules.

The classifier prompt itself is heavy (~3500 static tokens) and not under
test here — we mock the LLM at ``llm_client.invoke``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from agile_assistant.agents.supervisor import SupervisorAgent


@pytest.fixture
def supervisor(mock_llm_client: MagicMock) -> SupervisorAgent:
    """Supervisor wired with the conftest LLM mock and no DB engine.

    Without a DB engine the entity sanitiser skips DB validation but still
    runs synonym normalisation, hallucination filter and post-processing.
    """
    return SupervisorAgent(mock_llm_client, db_engine=None)


def _llm_returns(mock: MagicMock, payload: dict[str, Any]) -> None:
    """Make the mock LLM return the JSON-serialised payload as if from vLLM."""
    mock.invoke.return_value = json.dumps(payload, ensure_ascii=False)


# ===================================================================== #
# Fast path — regex issue-key detection
# ===================================================================== #


@pytest.mark.unit
class TestFastPath:
    @pytest.mark.parametrize(
        ("query", "expected_key"),
        [
            ("AL-12345", "AL-12345"),
            ("al-1", "AL-1"),
            ("расскажи про AL-38787", "AL-38787"),
            ("DATA-9999 в работе?", "DATA-9999"),
        ],
    )
    def test_issue_key_detected_normalised_uppercase(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
        query: str,
        expected_key: str,
    ) -> None:
        result = supervisor.process(query)
        assert result["intent"] == "task"
        assert result["query_type"] == "sql"
        assert result["entities"] == {"issue_key": expected_key}
        assert result["route"] == "db_query"
        assert result["original_query"] == query
        # Whole point of the fast path — LLM is bypassed entirely.
        mock_llm_client.invoke.assert_not_called()

    def test_no_issue_key_falls_through_to_slow_path(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Without an issue key in the query, the LLM is the only route.
        _llm_returns(
            mock_llm_client,
            {"intent": "general", "query_type": "simple", "entities": {}},
        )
        supervisor.process("привет")
        mock_llm_client.invoke.assert_called_once()


# ===================================================================== #
# Slow path — JSON parsing
# ===================================================================== #


@pytest.mark.unit
class TestSlowPathParsing:
    def test_valid_json_returns_classification(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        _llm_returns(
            mock_llm_client,
            {
                "intent": "tasks_filter",
                "query_type": "sql",
                "entities": {"team_name": "cthulhu"},
            },
        )
        result = supervisor.process("задачи команды cthulhu")
        assert result["intent"] == "tasks_filter"
        assert result["query_type"] == "sql"
        assert result["entities"]["team_name"] == "cthulhu"

    def test_markdown_fence_stripped_before_json_parse(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Some LLM modes wrap structured output in ```json fences — the
        # parser must peel those before calling json.loads.
        body = json.dumps({"intent": "general", "query_type": "simple", "entities": {}})
        mock_llm_client.invoke.return_value = f"```json\n{body}\n```"
        result = supervisor.process("привет")
        assert result["intent"] == "general"
        assert result["query_type"] == "simple"

    def test_embedded_json_recovered_via_regex(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Degraded responses sometimes prepend chatter — the {…} regex
        # fallback must still extract a valid object.
        body = json.dumps({"intent": "general", "query_type": "simple", "entities": {}})
        mock_llm_client.invoke.return_value = (
            f"Sure, here is the JSON: {body} — let me know if you need more."
        )
        result = supervisor.process("привет")
        assert result["intent"] == "general"

    def test_invalid_json_falls_back_to_error_intent(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Pure garbage with no { … } substring — parser must mark as error
        # so the caller can show a "classifier unavailable" message instead
        # of silently routing as a generic query.
        mock_llm_client.invoke.return_value = "not even close to JSON"
        result = supervisor.process("любой запрос")
        assert result["intent"] == "error"
        assert result["query_type"] == "error"

    def test_unknown_intent_normalised_to_general(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # The schema constrains the intent enum but a degraded LLM can
        # still return a value outside it — must be normalised to "general".
        _llm_returns(
            mock_llm_client,
            {"intent": "summarise", "query_type": "sql", "entities": {}},
        )
        result = supervisor.process("вопрос")
        assert result["intent"] == "general"

    def test_unknown_query_type_falls_back_per_intent(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # query_type "weird" → falls back via _INTENT_TO_QUERY_TYPE; for
        # intent=task that mapping yields "sql".
        _llm_returns(
            mock_llm_client,
            {"intent": "task", "query_type": "weird", "entities": {}},
        )
        result = supervisor.process("вопрос")
        assert result["query_type"] == "sql"


# ===================================================================== #
# Slow path — LLM exception
# ===================================================================== #


@pytest.mark.unit
class TestSlowPathLLMException:
    def test_llm_raises_returns_error_classification(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.side_effect = TimeoutError("vllm timeout")
        result = supervisor.process("какой-то запрос")
        assert result["intent"] == "error"
        assert result["query_type"] == "error"
        assert result["route"] == "direct_response"
        assert "TimeoutError" in result["error"]
        assert "vllm timeout" in result["error"]


# ===================================================================== #
# Post-processing rules
# ===================================================================== #


@pytest.mark.unit
class TestPostProcessing:
    def test_dangerous_prefix_blocked_to_simple(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Even if the LLM tried to classify "DROP TABLE foo" as something,
        # Rule 1 short-circuits before the result reaches the workflow.
        _llm_returns(
            mock_llm_client,
            {"intent": "task", "query_type": "sql", "entities": {}},
        )
        result = supervisor.process("DROP TABLE report_agile_dashboard")
        assert result["query_type"] == "simple"
        assert result["intent"] == "general"
        assert result["entities"] == {}

    def test_plural_task_noun_without_issue_key_becomes_tasks_filter(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # LLM returned intent=task on a plural noun query — Rule 6 demotes
        # to tasks_filter to keep the SQL Agent on the right branch.
        _llm_returns(
            mock_llm_client,
            {"intent": "task", "query_type": "sql", "entities": {}},
        )
        result = supervisor.process("покажи все баги команды cthulhu")
        assert result["intent"] == "tasks_filter"

    def test_hybrid_without_marker_downgraded_to_sql(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # The query is a plain factual ask; the LLM tried to upgrade to
        # hybrid (a known tendency on follow-ups). Rule 4 reverses that.
        _llm_returns(
            mock_llm_client,
            {
                "intent": "metric",
                "query_type": "hybrid",
                "entities": {"team_name": "cthulhu", "metric_name": "velocity"},
            },
        )
        result = supervisor.process("velocity команды cthulhu")
        assert result["query_type"] == "sql"

    def test_rag_with_stale_entities_dropped(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Rule 7 — rag / general queries don't carry filters; whatever
        # the LLM emitted (often anaphora-leaked from prior turns) is
        # discarded.
        _llm_returns(
            mock_llm_client,
            {
                "intent": "general",
                "query_type": "rag",
                "entities": {"team_name": "cthulhu"},
            },
        )
        result = supervisor.process("Что такое velocity?")
        assert result["query_type"] == "rag"
        assert result["entities"] == {}


# ===================================================================== #
# Profile default_team injection
# ===================================================================== #


@pytest.mark.unit
class TestProfileDefaults:
    def test_default_team_filled_when_entities_missing_team(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Sanitised entities lack team_name — profile fills in the gap.
        # Runs AFTER sanitisation on purpose (profile is trusted).
        _llm_returns(
            mock_llm_client,
            {
                "intent": "metric",
                "query_type": "sql",
                "entities": {"metric_name": "velocity"},
            },
        )
        profile = {"preferences": {"default_team": "cthulhu"}}
        result = supervisor.process("покажи velocity", user_profile=profile)
        assert result["entities"]["team_name"] == "cthulhu"

    def test_default_team_does_not_override_explicit_team(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # User explicitly named a team in the query — must not be silently
        # replaced by the profile default.
        _llm_returns(
            mock_llm_client,
            {
                "intent": "tasks_filter",
                "query_type": "sql",
                "entities": {"team_name": "khorne"},
            },
        )
        profile = {"preferences": {"default_team": "cthulhu"}}
        result = supervisor.process("задачи команды khorne", user_profile=profile)
        assert result["entities"]["team_name"] == "khorne"

    def test_no_profile_means_no_team_injection(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Sanity: missing profile doesn't crash and doesn't add a key.
        _llm_returns(
            mock_llm_client,
            {
                "intent": "metric",
                "query_type": "sql",
                "entities": {"metric_name": "velocity"},
            },
        )
        result = supervisor.process("покажи velocity", user_profile=None)
        assert "team_name" not in result["entities"]


# ===================================================================== #
# Route derivation & state shape
# ===================================================================== #


@pytest.mark.unit
class TestRouteAndShape:
    def test_simple_query_routes_to_direct_response(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        _llm_returns(
            mock_llm_client,
            {"intent": "general", "query_type": "simple", "entities": {}},
        )
        result = supervisor.process("привет")
        assert result["query_type"] == "simple"
        assert result["route"] == "direct_response"

    def test_non_simple_query_routes_to_db_query(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        _llm_returns(
            mock_llm_client,
            {
                "intent": "tasks_filter",
                "query_type": "sql",
                "entities": {"team_name": "cthulhu"},
            },
        )
        result = supervisor.process("задачи команды cthulhu")
        assert result["route"] == "db_query"

    def test_state_update_keys_are_stable(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        # Pin the contract: every invocation returns these 5 keys (slow path)
        # so workflow nodes can read them by name without guards.
        _llm_returns(
            mock_llm_client,
            {"intent": "general", "query_type": "simple", "entities": {}},
        )
        result = supervisor.process("привет")
        assert {"original_query", "intent", "entities", "query_type", "route"} <= set(result.keys())
