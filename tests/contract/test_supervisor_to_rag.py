"""Contract test: Supervisor → RAG Agent.

The RAG Agent reads exactly one field from upstream state:
``original_query``. Everything else (``intent``, ``entities``,
``query_type``) is informational. That makes the contract surface much
narrower than Supervisor → SQL, but it still has two failure modes
worth pinning:

  1. ``query_type=rag`` should never carry an issue_key in entities —
     a leftover key from an earlier turn would be dead weight in the
     RAG path (RAG Agent doesn't use it, but downstream Response Agent
     does fall through ``intent`` for branch routing).
  2. ``query_type=hybrid`` puts both SQL Agent and RAG Agent in scope.
     The RAG side must still get a non-empty ``original_query`` after
     the SQL side wrote into the merged state — i.e. SQL's update must
     not clobber it.

This test does NOT exercise the Qdrant retriever or the cross-encoder
reranker — those are unit-tested. We replace the retriever with a
stub that records the query it received and returns deterministic docs.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from hse_prom_prog.agents.rag_agent import RAGAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent


@pytest.fixture
def supervisor(mock_llm_client: MagicMock) -> SupervisorAgent:
    return SupervisorAgent(mock_llm_client, db_engine=None)


@pytest.fixture
def disabled_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reranker is heavy (loads a 200MB cross-encoder) — disable for tests.

    Same pattern as the unit tests for RAG Agent.
    """
    from hse_prom_prog.agents import rag_agent

    monkeypatch.setattr(rag_agent.settings, "reranker_enabled", False)


@pytest.fixture
def rag_agent_with_recording_retriever(
    mock_llm_client: MagicMock, disabled_reranker: None
) -> tuple[RAGAgent, MagicMock]:
    """RAGAgent paired with a retriever that records the query it got."""
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="Sprint Goal — это …", metadata={"source": "guide.md"}),
    ]
    mock_llm_client.invoke.return_value = "Ответ на основе документов."
    agent = RAGAgent(mock_llm_client, retriever)
    return agent, retriever


def _llm_returns(mock: MagicMock, payload: dict[str, Any]) -> None:
    mock.invoke.return_value = json.dumps(payload, ensure_ascii=False)


# ===================================================================== #
# 5.2 — Supervisor's RAG output drives RAG Agent
# ===================================================================== #


@pytest.mark.contract
class TestSupervisorRagOutputToRag:
    """When Supervisor classifies a query as RAG, the RAG Agent must be
    able to consume that state without missing keys."""

    def test_rag_classification_provides_original_query(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
        rag_agent_with_recording_retriever: tuple[RAGAgent, MagicMock],
    ) -> None:
        # The RAG Agent's only hard input requirement. Supervisor and RAG
        # share the same mock LLM so we sequence the responses: classifier
        # JSON first, then the RAG-generation answer.
        mock_llm_client.invoke.side_effect = [
            json.dumps({"intent": "general", "query_type": "rag", "entities": {}}),
            "Ответ на основе документов.",
        ]
        sup_out = supervisor.process("что такое sprint goal?")
        agent, retriever = rag_agent_with_recording_retriever
        result = agent.process(sup_out)
        # Retriever was called with the exact original query — no
        # silent rewrites between the two agents.
        retriever.invoke.assert_called_once_with("что такое sprint goal?")
        assert result["rag_response"] == "Ответ на основе документов."
        assert "guide.md" in result["rag_sources"]

    def test_rag_agent_tolerates_extra_keys_from_supervisor(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
        rag_agent_with_recording_retriever: tuple[RAGAgent, MagicMock],
    ) -> None:
        # Supervisor emits intent / entities / query_type / route — RAG
        # Agent ignores them but must not crash on unexpected fields.
        _llm_returns(
            mock_llm_client,
            {
                "intent": "general",
                "query_type": "rag",
                "entities": {"team_name": "cthulhu"},  # leftover, irrelevant
            },
        )
        sup_out = supervisor.process("как работает retro?")
        agent, _ = rag_agent_with_recording_retriever
        # Should not raise.
        result = agent.process(sup_out)
        assert "rag_response" in result
        assert "rag_sources" in result


# ===================================================================== #
# Hybrid path — SQL output must NOT clobber RAG's input
# ===================================================================== #


@pytest.mark.contract
class TestHybridStateMerging:
    """In the hybrid path the workflow runs SQL then RAG, merging state.
    The contract: SQL Agent's output dict must NOT carry ``original_query=None``
    (which would null out the field the RAG Agent reads next)."""

    def test_sql_output_keeps_original_query_for_rag(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # Pin: SQL Agent re-emits ``original_query`` in its output dict.
        # The hybrid node merges state as ``{**state, **sql_out, **rag_out}``;
        # if SQL ever returned ``"original_query": None`` we'd silently
        # send an empty query into Qdrant retrieval.
        from hse_prom_prog.agents.sql_agent import SQLAgent

        _llm_returns(
            mock_llm_client,
            {"intent": "metric", "query_type": "hybrid", "entities": {}},
        )
        sup_out = supervisor.process("velocity у cthulhu — нормально?")
        # Stub schema + DB-side helpers to keep the test hermetic.
        # We only inspect SQL Agent's output dict here.
        from unittest.mock import patch

        from hse_prom_prog.agents import sql_agent as sa

        with (
            patch.object(sa, "get_schema_compact", return_value="DDL"),
            patch.object(sa, "set_db", lambda _db: None),
        ):
            mock_db = MagicMock()
            mock_db.engine = MagicMock()
            agent = SQLAgent(db_connection=mock_db)
            fake_graph = MagicMock()
            fake_graph.invoke.return_value = {
                "final_sql": "SELECT 1",
                "final_result": [{"x": 1}],
                "final_error": "",
            }
            agent._graph = fake_graph  # type: ignore[assignment]
            sql_out = agent.process(sup_out)

        merged = {**sup_out, **sql_out}
        assert merged["original_query"] == "velocity у cthulhu — нормально?"

    def test_rag_consumes_merged_state_after_sql(
        self,
        supervisor: SupervisorAgent,
        mock_llm_client: MagicMock,
        rag_agent_with_recording_retriever: tuple[RAGAgent, MagicMock],
    ) -> None:
        # End-to-end of the hybrid contract: state is built by Supervisor,
        # extended by SQL, then handed to RAG. RAG reads ``original_query``
        # which must still match the user's input.
        _llm_returns(
            mock_llm_client,
            {"intent": "metric", "query_type": "hybrid", "entities": {}},
        )
        sup_out = supervisor.process("какая velocity и это нормально?")
        # Simulate SQL Agent's update (without running the agent).
        sql_update: dict[str, Any] = {
            "original_query": "какая velocity и это нормально?",
            "sql_query": "SELECT velocity FROM metrics LIMIT 10",
            "sql_result": [{"velocity": 42}],
            "error": None,
        }
        merged = {**sup_out, **sql_update}
        agent, retriever = rag_agent_with_recording_retriever
        agent.process(merged)
        # Pin: RAG Agent never sees the SQL payload but receives the
        # untouched user query — no leakage of SQL strings into Qdrant.
        retriever.invoke.assert_called_once_with("какая velocity и это нормально?")


# ===================================================================== #
# Negative — Supervisor's "simple" branch must NOT reach RAG
# ===================================================================== #


@pytest.mark.contract
class TestSimpleBranchSkipsRag:
    """Pin: when Supervisor returns ``query_type=simple`` (greeting,
    chitchat), the workflow must route past RAG. We verify the
    *output shape* does not advertise itself as RAG-eligible."""

    def test_simple_classification_has_no_rag_payload(
        self, supervisor: SupervisorAgent, mock_llm_client: MagicMock
    ) -> None:
        _llm_returns(
            mock_llm_client,
            {"intent": "general", "query_type": "simple", "entities": {}},
        )
        out = supervisor.process("привет")
        assert out["query_type"] == "simple"
        # Pin: Supervisor never preemptively writes rag_response / rag_sources
        # — those are produced *only* by the RAG Agent on the rag/hybrid paths.
        assert "rag_response" not in out
        assert "rag_sources" not in out
