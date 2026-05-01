"""Unit tests for ``ResponseAgent.process`` — 7 branches.

Branch routing (in resolve order):

  1. ``error``   — explicit error from Supervisor; fixed message, NO LLM
  2. ``simple``  — direct LLM answer (greeting / off-topic-but-allowed)
  3. ``both-fail`` (hybrid/rag with note + neither source) — fixed message
  4. ``rag``     — passthrough of RAG agent's text + sources block (NO LLM)
  5. ``hybrid``  — LLM combines DB data + RAG context + sources block
  6. ``sql``     — three intent flavours (task / tasks_filter / metric),
                   plus error / empty-result short-circuits

Tests are organised one class per branch. Where the LLM is invoked, the
``mock_llm_client`` fixture from conftest.py is used so we can both
verify the call happened and inject failures.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hse_prom_prog.agents.response_agent import ResponseAgent


@pytest.fixture
def agent(mock_llm_client: MagicMock) -> ResponseAgent:
    """Agent wired with the conftest mock LLM (default empty-string return)."""
    return ResponseAgent(mock_llm_client)


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "query_type": "sql",
        "route": "db_query",
        "intent": "task",
        "original_query": "тестовый запрос",
        "entities": {},
        "sql_result": [],
        "rag_response": "",
        "rag_sources": [],
        "error": "",
        "validation_result": {"use_sql": True, "use_rag": False, "note": None},
        "conversation_context": None,
        "user_profile": None,
    }
    base.update(overrides)
    return base


# ===================================================================== #
# Branch 1 — error
# ===================================================================== #


@pytest.mark.unit
class TestErrorBranch:
    def test_query_type_error_returns_fixed_message_without_llm(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(_state(query_type="error", error="boom"))
        assert "классификатор" in out["final_response"].lower()
        # Fixed-text branch — LLM must not be invoked at all.
        mock_llm_client.invoke.assert_not_called()

    def test_intent_error_takes_same_branch(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(_state(intent="error"))
        assert "классификатор" in out["final_response"].lower()
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# Branch 2 — simple (direct LLM)
# ===================================================================== #


@pytest.mark.unit
class TestSimpleBranch:
    def test_query_type_simple_invokes_llm(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.return_value = "Привет! Я могу помочь с..."
        out = agent.process(_state(query_type="simple", intent="general"))
        assert out["final_response"] == "Привет! Я могу помочь с..."
        assert mock_llm_client.invoke.call_count == 1

    def test_route_direct_response_takes_simple_branch(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        # The router can land here via either of two state shapes — both
        # paths must trip the same generator.
        mock_llm_client.invoke.return_value = "ok"
        agent.process(_state(route="direct_response", query_type="sql"))
        assert mock_llm_client.invoke.call_count == 1

    def test_simple_llm_exception_falls_back(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.side_effect = RuntimeError("vllm down")
        out = agent.process(_state(query_type="simple"))
        assert "Не удалось" in out["final_response"]
        assert "vllm down" in out["final_response"]

    def test_simple_timeout_exception_still_falls_back(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        # _is_timeout matches by class name — a TimeoutError must produce
        # the same user-visible fallback as any other exception.
        mock_llm_client.invoke.side_effect = TimeoutError("upstream timeout")
        out = agent.process(_state(query_type="simple"))
        assert "Не удалось" in out["final_response"]


# ===================================================================== #
# Branch 3 — both-fail (hybrid / rag with no usable source)
# ===================================================================== #


@pytest.mark.unit
class TestBothFailBranch:
    def test_hybrid_with_no_data_returns_fixed_offtopic_message(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(
            _state(
                query_type="hybrid",
                validation_result={
                    "use_sql": False,
                    "use_rag": False,
                    "note": "No data from SQL or RAG",
                },
            )
        )
        # Fixed Russian off-topic-style message — no LLM call.
        assert "базе знаний" in out["final_response"]
        assert "Agile-практиками" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()

    def test_rag_with_no_documents_takes_same_branch(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(
            _state(
                query_type="rag",
                validation_result={
                    "use_sql": False,
                    "use_rag": False,
                    "note": "No relevant documents found",
                },
            )
        )
        assert "Agile-практиками" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# Branch 4 — RAG passthrough (NO LLM)
# ===================================================================== #


@pytest.mark.unit
class TestRagBranch:
    def test_rag_response_with_sources_renders_block(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(
            _state(
                query_type="rag",
                rag_response="Velocity — это метрика скорости команды.",
                rag_sources=["agile-guide.pdf", "scrum-book.pdf"],
                validation_result={"use_sql": False, "use_rag": True, "note": None},
            )
        )
        assert "Velocity" in out["final_response"]
        assert "**Источники:**" in out["final_response"]
        assert "- agile-guide.pdf" in out["final_response"]
        assert "- scrum-book.pdf" in out["final_response"]
        # RAG passthrough doesn't re-prompt the LLM — pin this so a future
        # change adds a deliberate doc-rewriter, not silent latency.
        mock_llm_client.invoke.assert_not_called()

    def test_rag_response_without_sources_omits_block(self, agent: ResponseAgent) -> None:
        out = agent.process(
            _state(
                query_type="rag",
                rag_response="some answer",
                rag_sources=[],
                validation_result={"use_sql": False, "use_rag": True, "note": None},
            )
        )
        assert "**Источники:**" not in out["final_response"]


# ===================================================================== #
# Branch 5 — Hybrid
# ===================================================================== #


@pytest.mark.unit
class TestHybridBranch:
    def test_both_sources_invoke_llm_with_combined_prompt(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.return_value = "ДАННЫЕ: ...\nАНАЛИЗ: ..."
        out = agent.process(
            _state(
                query_type="hybrid",
                intent="task",
                sql_result=[{"issue_key": "AL-1", "summary": "fix"}],
                rag_response="Practice description",
                rag_sources=["doc.pdf"],
                validation_result={"use_sql": True, "use_rag": True, "note": None},
            )
        )
        # Body comes from the LLM, sources block gets appended verbatim.
        assert "ДАННЫЕ" in out["final_response"]
        assert "**Источники:**" in out["final_response"]
        assert "- doc.pdf" in out["final_response"]
        assert mock_llm_client.invoke.call_count == 1

    def test_hybrid_llm_exception_falls_back(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.side_effect = RuntimeError("connection reset")
        out = agent.process(
            _state(
                query_type="hybrid",
                sql_result=[{"k": "AL-1"}],
                rag_response="ctx",
                rag_sources=["doc.pdf"],
                validation_result={"use_sql": True, "use_rag": True, "note": None},
            )
        )
        assert "Не удалось" in out["final_response"]


# ===================================================================== #
# Branch 6 — SQL (multiple intents)
# ===================================================================== #


@pytest.mark.unit
class TestSqlBranch:
    def test_sql_error_short_circuits_with_fixed_message(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(_state(error="UndefinedTable: foo"))
        assert "Ошибка при выполнении" in out["final_response"]
        assert "UndefinedTable" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()

    def test_empty_result_with_issue_key_uses_specific_message(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(_state(sql_result=[], entities={"issue_key": "AL-99999"}))
        assert "AL-99999" in out["final_response"]
        assert "не найдена" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()

    def test_empty_result_without_issue_key_uses_generic_message(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        out = agent.process(_state(sql_result=[], entities={}))
        assert "ничего не найдено" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()

    def test_task_intent_invokes_task_generator(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.return_value = "Задача AL-1 в работе."
        out = agent.process(
            _state(
                intent="task",
                sql_result=[{"issue_key": "AL-1", "issue_status_act": "In Progress"}],
            )
        )
        assert out["final_response"] == "Задача AL-1 в работе."
        prompt = mock_llm_client.invoke.call_args.args[0]
        # Task prompt mentions the issue key under its Russian label.
        assert "Ключ задачи" in prompt
        assert "AL-1" in prompt

    def test_tasks_filter_intent_passes_assignee_flag(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        # When the user filtered by assignee, the prompt must include the
        # assignee column so the model can confirm. We probe via the
        # rendered prompt body rather than a structured arg — that way
        # the test catches both an entity-routing regression and a
        # template-change regression.
        mock_llm_client.invoke.return_value = "Найдено 2 задачи"
        agent.process(
            _state(
                intent="tasks_filter",
                sql_result=[
                    {
                        "issue_key": "AL-1",
                        "summary": "x",
                        "assignee_name": "Иванов",
                        "issue_status_act": "Open",
                        "feature_teams": "cthulhu",
                        "storypoints_act": 3,
                    }
                ],
                entities={"assignee": "Иванов"},
            )
        )
        prompt = mock_llm_client.invoke.call_args.args[0]
        assert "@Иванов" in prompt

    def test_metric_intent_invokes_metric_generator(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.return_value = "Velocity = 42"
        agent.process(_state(intent="metric", sql_result=[{"velocity": 42, "team": "cthulhu"}]))
        prompt = mock_llm_client.invoke.call_args.args[0]
        # The metric prompt mentions the metrics word and contains the value.
        assert "метрики" in prompt.lower()
        assert "42" in prompt

    def test_unknown_intent_falls_back_to_task_generator(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        # Defensive: a query routed to SQL with intent="general" still
        # generates a coherent reply (not a crash and not an empty string).
        mock_llm_client.invoke.return_value = "ok"
        agent.process(
            _state(
                intent="general",
                sql_result=[{"issue_key": "AL-1"}],
            )
        )
        prompt = mock_llm_client.invoke.call_args.args[0]
        # Lands in the same prompt template as intent="task".
        assert "Ключ задачи" in prompt

    def test_sql_llm_exception_falls_back(
        self, agent: ResponseAgent, mock_llm_client: MagicMock
    ) -> None:
        mock_llm_client.invoke.side_effect = RuntimeError("boom")
        out = agent.process(_state(intent="task", sql_result=[{"issue_key": "AL-1"}]))
        assert "Не удалось" in out["final_response"]
        assert "boom" in out["final_response"]


# ===================================================================== #
# Format helpers — small, but used everywhere
# ===================================================================== #


@pytest.mark.unit
class TestFormatHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "не указано"),
            (True, "да"),
            (False, "нет"),
            (3.14, "3.1"),
            ([1, 2, 3], "1, 2, 3"),
            ("plain", "plain"),
            (42, "42"),
        ],
    )
    def test_format_value_branches(self, agent: ResponseAgent, value: Any, expected: str) -> None:
        assert agent._format_value(value) == expected

    def test_prepare_task_list_truncates_at_max_rows(self, agent: ResponseAgent) -> None:
        from hse_prom_prog.agents.response_agent import _MAX_ROWS_IN_PROMPT

        rows = [{"issue_key": f"AL-{i}", "summary": "x"} for i in range(_MAX_ROWS_IN_PROMPT + 5)]
        out = agent._prepare_task_list(rows)
        assert "ещё 5 задач" in out
        # Showed first N, not all 25.
        assert "AL-0" in out
        assert f"AL-{_MAX_ROWS_IN_PROMPT - 1}" in out
        assert f"AL-{_MAX_ROWS_IN_PROMPT}" not in out

    def test_prepare_task_list_includes_assignee_when_flag(self, agent: ResponseAgent) -> None:
        rows = [{"issue_key": "AL-1", "summary": "x", "assignee_name": "Иванов"}]
        with_flag = agent._prepare_task_list(rows, include_assignee=True)
        without_flag = agent._prepare_task_list(rows, include_assignee=False)
        assert "@Иванов" in with_flag
        assert "@Иванов" not in without_flag

    def test_rag_passthrough_with_empty_sources(self, agent: ResponseAgent) -> None:
        # The dedicated helper is also exercised via the rag branch tests,
        # but pinning it directly catches a refactor that removes the
        # empty-sources short-circuit.
        result = agent._generate_rag_response("answer text", rag_sources=[])
        assert result == "answer text"
