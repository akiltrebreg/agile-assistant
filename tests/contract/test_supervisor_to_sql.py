"""Contract test: Supervisor → SQL Agent.

The Supervisor's ``process()`` returns a state-update dict. The SQL Agent's
``process()`` reads several keys from that update *via the merged workflow
state*. If either side renames or drops a key, routing silently breaks: the
SQL Agent falls back to a query without entity hints, which produces wrong
SQL on follow-up questions.

This test pins the *contract surface* — the keys SQL Agent actually reads
from a state that came directly out of Supervisor.process(). We do NOT
exercise the SQL graph itself (covered in unit tests); we replace the
LangGraph at ``self._graph`` with a mock and assert what Supervisor's
output enabled the SQL Agent to do.

Keys the SQL Agent reads from Supervisor's output:
  * ``original_query`` — used in HumanMessage and as the agent's "query"
  * ``intent``         — labels metrics (sql_empty_results) and trace input
  * ``entities``       — formatted into the system prompt (``_format_entities_hint``)

Keys NOT read from Supervisor's output but consumed from the merged state:
  * ``conversation_context`` — for ``_extract_previous_sql`` (anaphora)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from hse_prom_prog.agents.sql_agent import SQLAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent


@pytest.fixture
def supervisor(mock_llm_client: MagicMock) -> SupervisorAgent:
    return SupervisorAgent(mock_llm_client, db_engine=None)


@pytest.fixture
def sql_agent_with_mocked_graph(mock_db: MagicMock, monkeypatch: pytest.MonkeyPatch) -> SQLAgent:
    """SQLAgent whose internal LangGraph + schema loader are replaced.

    The graph mock returns a benign final state; we only care that the
    SQL Agent successfully *constructed* its system prompt from the
    Supervisor's output and that the state shape was compatible.

    ``get_schema_compact`` is patched so we don't need a live DB engine
    on the mock — the test focuses on contract surface, not DDL parsing.
    """
    from hse_prom_prog.agents import sql_agent as sa

    monkeypatch.setattr(sa, "get_schema_compact", lambda _engine: "FAKE_SCHEMA_DDL")
    monkeypatch.setattr(sa, "set_db", lambda _db: None)
    mock_db.engine = MagicMock()
    agent = SQLAgent(db_connection=mock_db)
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {
        "final_sql": "SELECT 1",
        "final_result": [{"value": 1}],
        "final_error": "",
    }
    agent._graph = fake_graph  # type: ignore[assignment]
    return agent


def _llm_returns(mock: MagicMock, payload: dict[str, Any]) -> None:
    mock.invoke.return_value = json.dumps(payload, ensure_ascii=False)


# ===================================================================== #
# 5.1 — fast-path output drives SQL Agent
# ===================================================================== #


@pytest.mark.contract
class TestSupervisorFastPathToSql:
    """Issue-key fast path → SQL Agent must see ``entities.issue_key``.

    Whole point: a regression that drops ``entities`` from the fast path
    output would silently disable the entities-hint block in the SQL
    prompt, leaking back to "ORDER BY metric LIMIT 1" generation.
    """

    def test_fastpath_output_has_keys_sql_agent_reads(self, supervisor: SupervisorAgent) -> None:
        out = supervisor.process("AL-1234")
        # All four keys SQL Agent will read from merged state.
        assert "original_query" in out
        assert "intent" in out
        assert "entities" in out
        assert "query_type" in out
        # Fast path is sql by definition (key found → DB lookup).
        assert out["query_type"] == "sql"

    def test_fastpath_entities_consumable_by_sql_agent(
        self,
        supervisor: SupervisorAgent,
        sql_agent_with_mocked_graph: SQLAgent,
    ) -> None:
        # End-to-end: Supervisor → state → SQL Agent. Should not raise.
        sup_out = supervisor.process("AL-9999")
        # Workflow does ``state | sup_out``; we simulate that here.
        merged = {**sup_out}
        result = sql_agent_with_mocked_graph.process(merged)
        # SQL Agent re-emits original_query for downstream consumers.
        assert result["original_query"] == "AL-9999"
        assert result["sql_query"] == "SELECT 1"
        assert result["error"] is None

    def test_fastpath_entities_appear_in_sql_prompt(
        self,
        supervisor: SupervisorAgent,
        sql_agent_with_mocked_graph: SQLAgent,
    ) -> None:
        # The contract that matters: ``entities.issue_key`` from Supervisor
        # must end up rendered into the SQL Agent's system prompt
        # (otherwise anaphora resolution silently degrades).
        sup_out = supervisor.process("расскажи про AL-42")
        sql_agent_with_mocked_graph.process(sup_out)
        invoke_args = sql_agent_with_mocked_graph._graph.invoke.call_args  # type: ignore[attr-defined]
        messages = invoke_args.kwargs.get("messages") or invoke_args.args[0].get("messages")
        system_msg = messages[0].content
        assert "AL-42" in system_msg


# ===================================================================== #
# 5.1 — slow-path output drives SQL Agent
# ===================================================================== #


@pytest.mark.contract
class TestSupervisorSlowPathToSql:
    """Slow path (LLM) → SQL Agent. Pins richer entity dicts (team, sprint)
    flow into the system prompt without key renames."""

    def test_slowpath_team_filter_lands_in_sql_prompt(
        self,
        supervisor: SupervisorAgent,
        sql_agent_with_mocked_graph: SQLAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        _llm_returns(
            mock_llm_client,
            {
                "intent": "tasks_filter",
                "query_type": "sql",
                "entities": {"team_name": "cthulhu"},
            },
        )
        sup_out = supervisor.process("задачи команды cthulhu")
        sql_agent_with_mocked_graph.process(sup_out)
        # The team name must be present in the system prompt — otherwise
        # the LLM has no reason to add a WHERE team_name filter.
        invoke_args = sql_agent_with_mocked_graph._graph.invoke.call_args  # type: ignore[attr-defined]
        messages = invoke_args.kwargs.get("messages") or invoke_args.args[0].get("messages")
        system_msg = messages[0].content
        assert "cthulhu" in system_msg

    def test_slowpath_intent_propagates_to_sql_metrics(
        self,
        supervisor: SupervisorAgent,
        sql_agent_with_mocked_graph: SQLAgent,
        mock_llm_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # SQL Agent records ``intent`` as a Prometheus label. If Supervisor
        # ever renamed the key the labels would silently turn into "unknown",
        # masking real distribution shifts.
        _llm_returns(
            mock_llm_client,
            {"intent": "metric", "query_type": "sql", "entities": {}},
        )
        captured: dict[str, str] = {}
        from hse_prom_prog.agents import sql_agent as sa

        original = sa.SQL_AGENT_DURATION

        class _DurationStub:
            def labels(self, **kw: Any) -> Any:
                captured["intent"] = kw.get("intent", "")
                return original.labels(**kw)

        monkeypatch.setattr(sa, "SQL_AGENT_DURATION", _DurationStub())
        sup_out = supervisor.process("какая velocity у cthulhu?")
        sql_agent_with_mocked_graph.process(sup_out)
        assert captured.get("intent") == "metric"

    def test_slowpath_empty_entities_does_not_inject_hint_block(
        self,
        supervisor: SupervisorAgent,
        sql_agent_with_mocked_graph: SQLAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # General queries with no entities → SQL Agent prompt MUST NOT carry
        # the "Известные сущности:" header (would prime the LLM with empty
        # filters and degrade quality).
        _llm_returns(
            mock_llm_client,
            {"intent": "general", "query_type": "sql", "entities": {}},
        )
        sup_out = supervisor.process("сколько всего задач в базе?")
        sql_agent_with_mocked_graph.process(sup_out)
        invoke_args = sql_agent_with_mocked_graph._graph.invoke.call_args  # type: ignore[attr-defined]
        messages = invoke_args.kwargs.get("messages") or invoke_args.args[0].get("messages")
        system_msg = messages[0].content
        assert "Известные сущности" not in system_msg


# ===================================================================== #
# Shape stability — the contract surface
# ===================================================================== #


@pytest.mark.contract
class TestSupervisorOutputShape:
    """Pin the exact keys Supervisor emits — adding/removing keys here
    forces a conscious update to whichever downstream agent reads them."""

    _EXPECTED_FAST_KEYS = frozenset({"original_query", "intent", "entities", "query_type", "route"})

    def test_fastpath_output_keys_pinned(self, supervisor: SupervisorAgent) -> None:
        out = supervisor.process("AL-7")
        assert frozenset(out.keys()) == self._EXPECTED_FAST_KEYS

    def test_slowpath_output_keys_superset_of_fastpath(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # Slow path may add ``error`` on classifier failure but must always
        # carry the fast-path key set so the SQL Agent's read pattern works
        # uniformly across both branches.
        _llm_returns(
            mock_llm_client,
            {"intent": "general", "query_type": "simple", "entities": {}},
        )
        out = supervisor.process("привет")
        assert self._EXPECTED_FAST_KEYS.issubset(out.keys())

    def test_classifier_error_still_carries_query_type_and_route(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # If the LLM blows up, downstream nodes still need a query_type to
        # route on. ``error`` is a pseudo-type owned by Response Agent.
        mock_llm_client.invoke.side_effect = TimeoutError("vllm down")
        out = supervisor.process("какой-то запрос без ключа")
        assert out["query_type"] == "error"
        assert out["route"] == "direct_response"
        assert "error" in out  # the human-readable message
