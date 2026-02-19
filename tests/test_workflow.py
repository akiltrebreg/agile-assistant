"""Tests for the multi-agent workflow components.

Tests cover:
- Supervisor: regex fast path, LLM classification, JSON parsing, query_type
- SQL Agent: query building for all intents, validation
- RAG Agent: retrieval, LLM generation, empty results, errors
- Validator Agent: sql-only, rag-only, hybrid, both-failed scenarios
- Response Agent: direct, DB-based, RAG, hybrid, error handling
- Workflow: end-to-end integration with mocked agents (all 4 routes)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.exc import OperationalError

from hse_prom_prog.agents.rag_agent import RAGAgent
from hse_prom_prog.agents.response_agent import ResponseAgent
from hse_prom_prog.agents.sql_agent import SQLAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent
from hse_prom_prog.agents.validator_agent import ValidatorAgent
from hse_prom_prog.graph.workflow import AgileWorkflow

# ────────────────────────────────────────────────────────────────
# Supervisor Agent
# ────────────────────────────────────────────────────────────────


class TestSupervisorAgent:
    """Tests for Supervisor agent."""

    def test_fast_path_regex_extracts_issue_key(self) -> None:
        """Regex fast path: issue key found → skip LLM, intent=task."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Выведи данные по задаче ABC-123")

        assert result["intent"] == "task"
        assert result["entities"] == {"issue_key": "ABC-123"}
        assert result["query_type"] == "sql"
        assert result["route"] == "db_query"
        mock_llm.invoke.assert_not_called()

    def test_fast_path_with_complex_key(self) -> None:
        """Regex handles multi-letter prefixes and long numbers."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Посмотри PROJ-456789")

        assert result["entities"]["issue_key"] == "PROJ-456789"
        assert result["intent"] == "task"
        assert result["query_type"] == "sql"

    def test_fast_path_multiple_keys_returns_first(self) -> None:
        """When multiple keys present, first one is returned."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Сравни ABC-123 и DEF-456")

        assert result["entities"]["issue_key"] == "ABC-123"

    def test_llm_classification_tasks_filter(self) -> None:
        """LLM classifies query as tasks_filter when no issue key."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "tasks_filter", "query_type": "sql",'
            ' "entities": {"team_name": "cthulhu", "sprint_name": null}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Все задачи команды cthulhu")

        assert result["intent"] == "tasks_filter"
        assert result["entities"]["team_name"] == "cthulhu"
        assert result["query_type"] == "sql"
        assert result["route"] == "db_query"

    def test_llm_classification_metric(self) -> None:
        """LLM classifies query as metric."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "metric", "query_type": "sql",'
            ' "entities": {"team_name": "lpop", "metric_name": "done_total"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Done total команды lpop")

        assert result["intent"] == "metric"
        assert result["entities"]["metric_name"] == "done_total"
        assert result["query_type"] == "sql"
        assert result["route"] == "db_query"

    def test_llm_classification_general_simple(self) -> None:
        """LLM classifies query as general with query_type=simple."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "general", "query_type": "simple", "entities": {}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Привет!")

        assert result["intent"] == "general"
        assert result["query_type"] == "simple"
        assert result["route"] == "direct_response"

    def test_llm_classification_rag(self) -> None:
        """LLM classifies query as RAG when asking about practices."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"intent": "general", "query_type": "rag", "entities": {}}'
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Как снизить Scope Drop?")

        assert result["intent"] == "general"
        assert result["query_type"] == "rag"
        assert result["route"] == "db_query"

    def test_llm_classification_hybrid(self) -> None:
        """LLM classifies query as hybrid when needing DB data + recommendations."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "metric", "query_type": "hybrid",'
            ' "entities": {"team_name": "cthulhu", "metric_name": "scope_drop"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Покажи scope drop команды cthulhu и дай рекомендации")

        assert result["intent"] == "metric"
        assert result["query_type"] == "hybrid"
        assert result["entities"]["team_name"] == "cthulhu"
        assert result["route"] == "db_query"

    def test_llm_returns_markdown_json(self) -> None:
        """LLM wraps JSON in markdown code fences — parsed correctly."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '```json\n{"intent": "general", "query_type": "simple", "entities": {}}\n```'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Привет")

        assert result["intent"] == "general"
        assert result["query_type"] == "simple"

    def test_llm_returns_garbage(self) -> None:
        """LLM returns unparsable text → fallback to general/simple."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "I don't understand"
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Что-то непонятное")

        assert result["intent"] == "general"
        assert result["query_type"] == "simple"
        assert result["route"] == "direct_response"

    def test_llm_exception_fallback(self) -> None:
        """LLM raises exception → fallback to general/simple."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("timeout")
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Задачи команды lpop")

        assert result["intent"] == "general"
        assert result["query_type"] == "simple"
        assert result["route"] == "direct_response"

    def test_null_entities_cleaned(self) -> None:
        """Null-valued entities are removed from the dict."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "tasks_filter", "query_type": "sql",'
            ' "entities": {"team_name": "cthulhu", "sprint_name": null,'
            ' "issue_type": null, "status": null}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Задачи cthulhu")

        assert "sprint_name" not in result["entities"]
        assert result["entities"] == {"team_name": "cthulhu"}

    def test_invalid_query_type_falls_back(self) -> None:
        """Invalid query_type falls back to intent-based mapping."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "metric", "query_type": "invalid_type", "entities": {"team_name": "lpop"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Метрики lpop")

        assert result["query_type"] == "sql"  # metric intent → sql

    def test_missing_query_type_falls_back(self) -> None:
        """Missing query_type falls back to intent-based mapping."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "tasks_filter", "entities": {"team_name": "lpop"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Задачи lpop")

        assert result["query_type"] == "sql"  # tasks_filter intent → sql


# ────────────────────────────────────────────────────────────────
# SQL Agent
# ────────────────────────────────────────────────────────────────


class TestSQLAgent:
    """Tests for SQL agent query building."""

    def test_build_task_query(self) -> None:
        """Task intent builds WHERE issue_key = :issue_key."""
        agent = SQLAgent()
        sql, params = agent._build_task_query({"issue_key": "AL-38787"})

        assert "issue_key = :issue_key" in sql
        assert params["issue_key"] == "AL-38787"
        assert "report_agile_dashboard" in sql

    def test_build_tasks_filter_single(self) -> None:
        """Tasks filter with one entity → single WHERE condition."""
        agent = SQLAgent()
        sql, params = agent._build_tasks_filter_query({"team_name": "cthulhu"})

        assert "feature_teams ILIKE :p_team_name" in sql
        assert params["p_team_name"] == "%cthulhu%"
        assert "LIMIT" in sql

    def test_build_tasks_filter_multiple(self) -> None:
        """Tasks filter with multiple entities → AND conditions."""
        agent = SQLAgent()
        sql, _params = agent._build_tasks_filter_query(
            {
                "team_name": "lpop",
                "status": "In Progress",
            }
        )

        assert "feature_teams ILIKE :p_team_name" in sql
        assert "issue_status_act ILIKE :p_status" in sql
        assert "AND" in sql

    def test_build_tasks_filter_empty(self) -> None:
        """Tasks filter with no entities → WHERE TRUE (all rows)."""
        agent = SQLAgent()
        sql, params = agent._build_tasks_filter_query({})

        assert "WHERE TRUE" in sql
        assert params == {}

    def test_build_metric_query_specific(self) -> None:
        """Metric with specific metric_name → selects that column."""
        agent = SQLAgent()
        sql, _params = agent._build_metric_query(
            {
                "team_name": "cthulhu",
                "metric_name": "done_total",
            }
        )

        assert "SELECT feature_teams, sprint_name, sprint_state, done_total" in sql
        assert "report_agile_dashboard_metrics" in sql

    def test_build_metric_query_all(self) -> None:
        """Metric without metric_name → SELECT *."""
        agent = SQLAgent()
        sql, _params = agent._build_metric_query({"team_name": "lpop"})

        assert "SELECT *" in sql

    def test_build_metric_disallowed_column(self) -> None:
        """Disallowed metric_name → falls back to SELECT *."""
        agent = SQLAgent()
        sql, _params = agent._build_metric_query(
            {
                "metric_name": "nonexistent_column",
            }
        )

        assert "SELECT *" in sql

    def test_process_unknown_intent(self) -> None:
        """Unknown intent returns error."""
        mock_db = MagicMock()
        agent = SQLAgent(db_connection=mock_db)
        result = agent.process(
            {
                "intent": "unknown",
                "entities": {},
                "original_query": "test",
            }
        )

        assert result["error"] is not None
        assert "Unknown intent" in result["error"]

    def test_process_db_error(self) -> None:
        """Database error is caught and returned in state."""
        mock_db = MagicMock()
        mock_db.execute_query.side_effect = OperationalError(
            "stmt", "params", Exception("connection refused")
        )
        agent = SQLAgent(db_connection=mock_db)

        result = agent.process(
            {
                "intent": "task",
                "entities": {"issue_key": "AL-123"},
                "original_query": "test",
            }
        )

        assert result["error"] is not None
        assert "Database error" in result["error"]
        assert result["sql_result"] is None

    def test_process_empty_result(self) -> None:
        """Empty DB result returns empty list, no error."""
        mock_db = MagicMock()
        mock_db.execute_query.return_value = []
        agent = SQLAgent(db_connection=mock_db)

        result = agent.process(
            {
                "intent": "task",
                "entities": {"issue_key": "NONEXISTENT-999"},
                "original_query": "test",
            }
        )

        assert result["sql_result"] == []
        assert result["error"] is None

    def test_process_success(self) -> None:
        """Successful query returns results."""
        mock_db = MagicMock()
        mock_db.execute_query.return_value = [{"issue_key": "AL-38787", "summary": "Test task"}]
        agent = SQLAgent(db_connection=mock_db)

        result = agent.process(
            {
                "intent": "task",
                "entities": {"issue_key": "AL-38787"},
                "original_query": "test",
            }
        )

        assert len(result["sql_result"]) == 1
        assert result["sql_result"][0]["issue_key"] == "AL-38787"
        assert result["error"] is None


# ────────────────────────────────────────────────────────────────
# RAG Agent
# ────────────────────────────────────────────────────────────────


class TestRAGAgent:
    """Tests for RAG agent."""

    def _make_mock_doc(self, content: str, source: str = "doc.md", category: str = "agile"):
        """Create a mock LangChain Document."""
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source, "category": category}
        return doc

    def test_process_success(self) -> None:
        """RAG Agent retrieves docs and generates answer."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Scope Drop — это снижение объёма спринта."
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            self._make_mock_doc("Scope Drop definition...", "scope.md", "metrics"),
        ]
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "Что такое Scope Drop?"})

        assert result["rag_response"] == "Scope Drop — это снижение объёма спринта."
        assert "metrics/scope.md" in result["rag_sources"]
        mock_retriever.invoke.assert_called()
        mock_llm.invoke.assert_called()

    def test_process_no_documents(self) -> None:
        """No relevant documents → rag_response is None."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "rewritten query"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "Непонятный запрос"})

        assert result["rag_response"] is None
        assert result["rag_sources"] == []

    def test_process_retrieval_error(self) -> None:
        """All retrieval sub-queries fail → rag_response is None."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "rewritten query"
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = ConnectionError("Qdrant down")
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "test"})

        assert result["rag_response"] is None
        assert result["rag_sources"] == []

    def test_process_llm_error(self) -> None:
        """LLM generation error is caught; sources still returned."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("vLLM down")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            self._make_mock_doc("Some context", "doc.md"),
        ]
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "test"})

        assert result["rag_response"] is None
        assert len(result["rag_sources"]) == 1
        assert "RAG generation error" in result["error"]

    def test_context_truncation(self) -> None:
        """Long documents are truncated to _MAX_CONTEXT_CHARS."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()
        # Create documents that exceed the 4000 char limit
        mock_retriever.invoke.return_value = [
            self._make_mock_doc("A" * 3000, "doc1.md"),
            self._make_mock_doc("B" * 3000, "doc2.md"),
        ]
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Answer"
        # Only one source because second doc exceeds char limit
        assert len(result["rag_sources"]) == 1

    def test_deduplicates_sources(self) -> None:
        """Same source appearing in multiple docs is deduplicated."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            self._make_mock_doc("Chunk 1", "doc.md", "agile"),
            self._make_mock_doc("Chunk 2", "doc.md", "agile"),
        ]
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "test"})

        assert result["rag_sources"] == ["agile/doc.md"]

    # ── Query Rewriter tests ──────────────────────────────────

    def test_rewrite_query_success(self) -> None:
        """Query rewriter transforms vague query into precise search query."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "определение и расчёт метрики Scope Drop в Agile"
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._rewrite_query("как считается эта метрика?")

        assert result == "определение и расчёт метрики Scope Drop в Agile"
        mock_llm.invoke.assert_called_once()

    def test_rewrite_query_fallback_on_error(self) -> None:
        """Query rewriter falls back to original query on LLM error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("LLM down")
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._rewrite_query("test query")

        assert result == "test query"

    def test_rewrite_query_fallback_on_empty(self) -> None:
        """Query rewriter falls back to original if LLM returns empty string."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "   "
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._rewrite_query("test query")

        assert result == "test query"

    # ── Multi-Query tests ─────────────────────────────────────

    def test_generate_multi_queries_success(self) -> None:
        """Multi-query generates 3 alternative formulations."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "Что такое Scope Drop в Agile\n"
            "Как рассчитывается показатель Scope Drop\n"
            "Метрика снижения объёма спринта"
        )
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._generate_multi_queries("Scope Drop")

        assert len(result) == 3
        assert "Scope Drop" in result[0]

    def test_generate_multi_queries_fallback_on_error(self) -> None:
        """Multi-query falls back to [original_query] on LLM error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("LLM down")
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._generate_multi_queries("test query")

        assert result == ["test query"]

    def test_generate_multi_queries_empty_response(self) -> None:
        """Multi-query falls back to [original_query] on empty LLM response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._generate_multi_queries("test query")

        assert result == ["test query"]

    # ── HyDE tests ────────────────────────────────────────────

    def test_generate_hyde_document_success(self) -> None:
        """HyDE generates a hypothetical document fragment."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "Scope Drop — это метрика, отражающая процент задач, "
            "исключённых из спринта после его начала."
        )
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._generate_hyde_document("Что такое Scope Drop?")

        assert "Scope Drop" in result
        assert len(result) > 20

    def test_generate_hyde_document_fallback_on_error(self) -> None:
        """HyDE returns empty string on LLM error."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("LLM down")
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._generate_hyde_document("test query")

        assert result == ""

    # ── Advanced retrieval integration tests ──────────────────

    def test_retrieve_deduplicates_across_sub_queries(self) -> None:
        """Results from multiple sub-queries are deduplicated by page_content."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten query",  # _rewrite_query
            "variant 1\nvariant 2",  # _generate_multi_queries
            "hypothetical document about topic",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()

        shared_doc = self._make_mock_doc("Shared content", "doc.md", "agile")
        unique_doc = self._make_mock_doc("Unique content", "doc2.md", "agile")
        mock_retriever.invoke.side_effect = [
            [shared_doc],  # variant 1
            [shared_doc, unique_doc],  # variant 2
            [shared_doc],  # HyDE
        ]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Final answer"
        assert len(result["rag_sources"]) == 2

    def test_retrieve_partial_sub_query_failure(self) -> None:
        """If some sub-queries fail, results from successful ones are used."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten query",  # _rewrite_query
            "variant 1",  # _generate_multi_queries (1 variant)
            "hypothetical document",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()

        doc = self._make_mock_doc("Some content", "doc.md", "agile")
        mock_retriever.invoke.side_effect = [
            [doc],  # variant 1 succeeds
            ConnectionError("Qdrant timeout"),  # HyDE query fails
        ]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Final answer"
        assert "agile/doc.md" in result["rag_sources"]

    def test_retrieve_all_strategies_fail_gracefully(self) -> None:
        """When all retriever sub-queries fail, result is None (no crash)."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten query",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hypothetical doc",  # _generate_hyde_document
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = ConnectionError("Qdrant down")

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] is None
        assert result["rag_sources"] == []

    # ── RRF tests ─────────────────────────────────────────────

    def test_rrf_merge_combines_ranked_lists(self) -> None:
        """RRF merges two ranked lists; doc in both lists ranks higher."""
        doc_a = self._make_mock_doc("Content A", "a.md")
        doc_b = self._make_mock_doc("Content B", "b.md")
        doc_c = self._make_mock_doc("Content C", "c.md")

        vector_list = [doc_a, doc_b]  # A rank 0, B rank 1
        bm25_list = [doc_b, doc_c]  # B rank 0, C rank 1

        result = RAGAgent._rrf_merge([vector_list, bm25_list])

        contents = [doc.page_content for doc in result]
        # B appears in both lists → highest RRF score
        assert contents[0] == "Content B"
        assert len(result) == 3

    def test_rrf_merge_empty_lists(self) -> None:
        """RRF with empty input returns empty output."""
        result = RAGAgent._rrf_merge([[], []])
        assert result == []

    # ── Metadata filtering tests ──────────────────────────────

    def test_resolve_category_filter_metric(self) -> None:
        """intent=metric → category filter 'metrics'."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._resolve_category_filter({"intent": "metric"})
        assert result == "metrics"

    def test_resolve_category_filter_general(self) -> None:
        """intent=general → no category filter."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent._resolve_category_filter({"intent": "general"})
        assert result is None

    # ── BM25 search tests ─────────────────────────────────────

    def test_bm25_search_returns_results(self) -> None:
        """BM25 search delegates to bm25_index when available."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_bm25 = MagicMock()
        doc = self._make_mock_doc("BM25 result", "doc.md")
        mock_bm25.search.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever, bm25_index=mock_bm25)
        result = agent._bm25_search("scope_drop", k=4)

        assert len(result) == 1
        assert result[0].page_content == "BM25 result"
        mock_bm25.search.assert_called_once_with("scope_drop", k=4, category=None)

    def test_bm25_search_without_index(self) -> None:
        """BM25 search returns empty when index is None."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)  # no bm25_index

        result = agent._bm25_search("test")
        assert result == []

    # ── Hybrid search integration tests ───────────────────────

    def test_hybrid_search_combines_bm25_and_vector(self) -> None:
        """Hybrid search uses both BM25 and vector, merges with RRF."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()

        # Vector search returns doc_a
        mock_vector_store = MagicMock()
        doc_a = self._make_mock_doc("Vector result", "vec.md", "metrics")
        mock_vector_store.similarity_search.return_value = [doc_a]

        # BM25 returns doc_b
        mock_bm25 = MagicMock()
        doc_b = self._make_mock_doc("BM25 result", "bm25.md", "metrics")
        mock_bm25.search.return_value = [doc_b]

        agent = RAGAgent(mock_llm, mock_retriever, mock_vector_store, mock_bm25)
        result = agent.process({"original_query": "scope_drop", "intent": "metric"})

        assert result["rag_response"] == "Final answer"
        # Both sources should be present (from vector and BM25)
        assert len(result["rag_sources"]) >= 1
        mock_vector_store.similarity_search.assert_called()
        mock_bm25.search.assert_called()

    def test_hybrid_search_vector_only_fallback(self) -> None:
        """Without bm25_index, falls back to vector-only search."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()

        mock_vector_store = MagicMock()
        doc = self._make_mock_doc("Vector doc", "v.md")
        mock_vector_store.similarity_search.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever, mock_vector_store)  # no bm25
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Answer"
        mock_vector_store.similarity_search.assert_called()

    def test_hybrid_search_with_category_filter(self) -> None:
        """intent=metric passes category='metrics' to vector search filter."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()

        mock_vector_store = MagicMock()
        doc = self._make_mock_doc("Metric doc", "m.md", "metrics")
        mock_vector_store.similarity_search.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever, mock_vector_store)
        agent.process({"original_query": "scope_drop", "intent": "metric"})

        # Verify filter was passed to similarity_search
        call_kwargs = mock_vector_store.similarity_search.call_args_list[0]
        assert call_kwargs.kwargs.get("filter") is not None

    # ── BM25Index class test ──────────────────────────────────

    def test_bm25_index_search_with_category_filter(self) -> None:
        """BM25Index.search filters by category metadata."""
        from hse_prom_prog.rag.bm25_index import BM25Index

        doc_metrics = MagicMock()
        doc_metrics.page_content = "scope drop definition and calculation"
        doc_metrics.metadata = {"source": "scope.md", "category": "metrics"}

        doc_agile = MagicMock()
        doc_agile.page_content = "general agile practices overview"
        doc_agile.metadata = {"source": "agile.md", "category": "agile"}

        index = BM25Index([doc_metrics, doc_agile])

        # Without filter: both docs are candidates
        results_all = index.search("scope drop", k=4)
        assert len(results_all) >= 1

        # With metrics filter: only metrics doc
        results_filtered = index.search("scope drop", k=4, category="metrics")
        for doc in results_filtered:
            assert doc.metadata["category"] == "metrics"


# ────────────────────────────────────────────────────────────────
# Reranker
# ────────────────────────────────────────────────────────────────


class TestReranker:
    """Tests for the cross-encoder Reranker."""

    def _make_mock_doc(self, content: str, source: str = "doc.md", category: str = "agile"):
        """Create a mock LangChain Document."""
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source, "category": category}
        return doc

    @patch("hse_prom_prog.rag.reranker.CrossEncoder")
    def test_rerank_scores_and_filters(self, mock_ce_cls: MagicMock) -> None:
        """Docs below threshold are filtered out."""
        from hse_prom_prog.rag.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.7, 0.2, 0.5])
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model", threshold=0.3, top_n=5)
        docs = [self._make_mock_doc(f"Doc {i}") for i in range(5)]
        result = reranker.rerank("test query", docs)

        # Scores 0.1, 0.2 are below threshold 0.3 → filtered out
        assert len(result) == 3
        mock_model.predict.assert_called_once()

    @patch("hse_prom_prog.rag.reranker.CrossEncoder")
    def test_rerank_respects_top_n(self, mock_ce_cls: MagicMock) -> None:
        """Only top_n documents returned even if more pass threshold."""
        from hse_prom_prog.rag.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.31])
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model", threshold=0.3, top_n=3)
        docs = [self._make_mock_doc(f"Doc {i}") for i in range(8)]
        result = reranker.rerank("test query", docs)

        assert len(result) == 3

    @patch("hse_prom_prog.rag.reranker.CrossEncoder")
    def test_rerank_empty_input(self, mock_ce_cls: MagicMock) -> None:
        """Empty document list returns empty, predict not called."""
        from hse_prom_prog.rag.reranker import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model", threshold=0.3, top_n=5)
        result = reranker.rerank("test query", [])

        assert result == []
        mock_model.predict.assert_not_called()

    @patch("hse_prom_prog.rag.reranker.CrossEncoder")
    def test_rerank_all_below_threshold(self, mock_ce_cls: MagicMock) -> None:
        """All docs below threshold → empty result."""
        from hse_prom_prog.rag.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.05, 0.2])
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model", threshold=0.3, top_n=5)
        docs = [self._make_mock_doc(f"Doc {i}") for i in range(3)]
        result = reranker.rerank("test query", docs)

        assert result == []

    def test_lost_in_the_middle_reorder(self) -> None:
        """Most relevant docs placed at start and end, weakest in middle."""
        from hse_prom_prog.rag.reranker import Reranker

        docs = [self._make_mock_doc(f"Doc {i}") for i in range(5)]
        scored_docs = [
            (0.9, docs[0]),  # best
            (0.7, docs[1]),  # 2nd
            (0.5, docs[2]),  # 3rd
            (0.3, docs[3]),  # 4th
            (0.1, docs[4]),  # 5th
        ]
        result = Reranker._lost_in_the_middle_reorder(scored_docs)

        # Expected: [best, 3rd, 5th, 4th, 2nd]
        assert result[0].page_content == "Doc 0"  # best at start
        assert result[-1].page_content == "Doc 1"  # 2nd-best at end
        assert result[2].page_content == "Doc 4"  # weakest in middle
        assert len(result) == 5

    # ── RAGAgent reranking integration ─────────────────────────

    def test_rerank_integration_filters_docs(self) -> None:
        """Reranker in RAGAgent filters out low-scoring docs."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()

        doc_a = self._make_mock_doc("Kept doc", "kept.md")
        doc_b = self._make_mock_doc("Dropped doc", "dropped.md")
        mock_retriever.invoke.return_value = [doc_a, doc_b]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [doc_a]  # drops doc_b

        agent = RAGAgent(mock_llm, mock_retriever, reranker=mock_reranker)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Final answer"
        assert "agile/kept.md" in result["rag_sources"]
        mock_reranker.rerank.assert_called()

    def test_rerank_disabled_when_no_reranker(self) -> None:
        """Without reranker, docs pass through unchanged."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Content", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever)  # no reranker
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Answer"
        assert "agile/doc.md" in result["rag_sources"]

    def test_rerank_failure_falls_back(self) -> None:
        """Reranker error → docs returned in original order."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Content", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("model crashed")

        agent = RAGAgent(mock_llm, mock_retriever, reranker=mock_reranker)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Answer"
        assert "agile/doc.md" in result["rag_sources"]

    def test_rerank_returns_empty_gives_none_response(self) -> None:
        """Reranker returns empty → rag_response is None."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",
            "variant 1",
            "hyde doc",
        ]
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Content", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []  # all below threshold

        agent = RAGAgent(mock_llm, mock_retriever, reranker=mock_reranker)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] is None
        assert result["rag_sources"] == []

    def test_retrieve_with_reranker_full_pipeline(self) -> None:
        """Full pipeline: multi-query + hybrid + reranking."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten query",  # _rewrite_query
            "v1\nv2",  # _generate_multi_queries
            "hypothetical answer",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()

        doc1 = self._make_mock_doc("Doc 1", "d1.md")
        doc2 = self._make_mock_doc("Doc 2", "d2.md")
        doc3 = self._make_mock_doc("Doc 3", "d3.md")
        mock_retriever.invoke.return_value = [doc1, doc2, doc3]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [doc1, doc3]  # keeps 2 of 3

        agent = RAGAgent(mock_llm, mock_retriever, reranker=mock_reranker)
        result = agent.process({"original_query": "scope_drop"})

        assert result["rag_response"] == "Final answer"
        sources = result["rag_sources"]
        assert "agile/d1.md" in sources
        assert "agile/d3.md" in sources
        mock_reranker.rerank.assert_called_once()


# ────────────────────────────────────────────────────────────────
# Context Assembly (Parent-Child Chunking, Dedup, Metadata)
# ────────────────────────────────────────────────────────────────


class TestContextAssembly:
    """Tests for parent-child chunking, parent-level dedup, and structured metadata."""

    def _make_mock_doc(self, content: str, source: str = "doc.md", category: str = "agile"):
        """Create a mock LangChain Document."""
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source, "category": category}
        return doc

    # ── Section extraction ───────────────────────────────────

    def test_extract_section_finds_markdown_heading(self) -> None:
        """Section extraction finds the first markdown heading."""
        from hse_prom_prog.rag.ingest import _extract_section

        text = "Some preamble\n## Описание метрики\nContent here"
        assert _extract_section(text) == "Описание метрики"

    def test_extract_section_no_heading(self) -> None:
        """Section extraction returns em-dash when no heading found."""
        from hse_prom_prog.rag.ingest import _extract_section

        text = "Plain text without any headings at all."
        assert _extract_section(text) == "\u2014"

    # ── Parent-child splitting ───────────────────────────────

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_split_parent_child_metadata(self, mock_settings: MagicMock) -> None:
        """Parent-child split adds parent_id, parent_content, section to children."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_parent_child

        mock_settings.parent_chunk_size = 200
        mock_settings.parent_chunk_overlap = 0
        mock_settings.child_chunk_size = 50
        mock_settings.child_chunk_overlap = 0

        doc = Document(
            page_content="# My Section\n" + "A" * 180,
            metadata={"source": "test.md", "category": "agile"},
        )
        children = _split_parent_child([doc])

        assert len(children) > 0
        for child in children:
            assert "parent_id" in child.metadata
            assert "parent_content" in child.metadata
            assert "section" in child.metadata
            assert "chunk_index" in child.metadata
            assert child.metadata["section"] == "My Section"
            assert len(child.metadata["parent_id"]) == 36  # UUID format
        # chunk_index is sequential within parent
        indices = [c.metadata["chunk_index"] for c in children]
        assert indices == list(range(len(children)))

    # ── Parent resolution + dedup ────────────────────────────

    def test_resolve_parents_dedup_by_parent_id(self) -> None:
        """Two children with same parent_id resolve to a single parent."""
        child1 = self._make_mock_doc("child chunk 1")
        child1.metadata["parent_id"] = "pid-1"
        child1.metadata["parent_content"] = "Full parent content here"

        child2 = self._make_mock_doc("child chunk 2")
        child2.metadata["parent_id"] = "pid-1"
        child2.metadata["parent_content"] = "Full parent content here"

        result = RAGAgent._resolve_parents([child1, child2])

        assert len(result) == 1
        assert result[0].page_content == "Full parent content here"

    def test_resolve_parents_different_parents(self) -> None:
        """Children from different parents both survive dedup."""
        child1 = self._make_mock_doc("child chunk 1")
        child1.metadata["parent_id"] = "pid-1"
        child1.metadata["parent_content"] = "Parent 1 content"

        child2 = self._make_mock_doc("child chunk 2")
        child2.metadata["parent_id"] = "pid-2"
        child2.metadata["parent_content"] = "Parent 2 content"

        result = RAGAgent._resolve_parents([child1, child2])

        assert len(result) == 2
        contents = {doc.page_content for doc in result}
        assert "Parent 1 content" in contents
        assert "Parent 2 content" in contents

    def test_resolve_parents_backward_compat(self) -> None:
        """Docs without parent_content/parent_id fall back to page_content."""
        doc1 = self._make_mock_doc("Old chunk A")
        doc2 = self._make_mock_doc("Old chunk B")

        result = RAGAgent._resolve_parents([doc1, doc2])

        assert len(result) == 2
        assert result[0].page_content == "Old chunk A"
        assert result[1].page_content == "Old chunk B"

    def test_resolve_parents_backward_compat_duplicate(self) -> None:
        """Docs without parent_id with identical page_content are deduped."""
        doc1 = self._make_mock_doc("Same content")
        doc2 = self._make_mock_doc("Same content")

        result = RAGAgent._resolve_parents([doc1, doc2])

        assert len(result) == 1

    # ── Structured metadata formatting ───────────────────────

    def test_format_context_block_with_page(self) -> None:
        """Context block includes source, section, and page for PDF docs."""
        doc = self._make_mock_doc("Content text here", "metrics/done_total.pdf")
        doc.metadata["section"] = "Описание метрики"
        doc.metadata["page"] = 3

        result = RAGAgent._format_context_block(doc)

        assert "[Источник: done_total.pdf" in result
        assert "Раздел: Описание метрики" in result
        assert "Стр. 3" in result
        assert "Content text here" in result

    def test_format_context_block_without_page(self) -> None:
        """Context block omits page for Markdown docs."""
        doc = self._make_mock_doc("Content text here", "agile/practices.md")
        doc.metadata["section"] = "Definition of Done"

        result = RAGAgent._format_context_block(doc)

        assert "[Источник: practices.md" in result
        assert "Раздел: Definition of Done" in result
        assert "Стр." not in result
        assert "Content text here" in result

    def test_format_context_block_no_section(self) -> None:
        """Context block uses em-dash for section when metadata is missing."""
        doc = self._make_mock_doc("Content", "doc.md")

        result = RAGAgent._format_context_block(doc)

        assert "\u2014" in result  # em-dash as fallback section


# ────────────────────────────────────────────────────────────────
# Semantic Chunking Pipeline
# ────────────────────────────────────────────────────────────────


class TestTableDetection:
    """Tests for _detect_table_blocks regex-based table detection."""

    def test_detect_pipe_table(self) -> None:
        """Pipe-delimited table (3+ lines) is detected."""
        from hse_prom_prog.rag.ingest import _detect_table_blocks

        text = "Some text\n| A | B | C |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\nMore text"
        regions = _detect_table_blocks(text)
        assert len(regions) == 1
        assert regions[0] == (1, 3)

    def test_detect_whitespace_table(self) -> None:
        """Multi-column whitespace-aligned table is detected."""
        from hse_prom_prog.rag.ingest import _detect_table_blocks

        text = "Header\nCol1  Col2  Col3\nA     B     C\nD     E     F\nEnd"
        regions = _detect_table_blocks(text)
        assert len(regions) == 1

    def test_no_table_detected(self) -> None:
        """Plain text without tables returns empty list."""
        from hse_prom_prog.rag.ingest import _detect_table_blocks

        text = "Just some regular text.\nAnother line.\nThird line."
        regions = _detect_table_blocks(text)
        assert regions == []

    def test_short_run_not_detected(self) -> None:
        """Only 2 consecutive table lines (below threshold of 3) are not detected."""
        from hse_prom_prog.rag.ingest import _detect_table_blocks

        text = "Text\n| A | B |\n| 1 | 2 |\nMore text"
        regions = _detect_table_blocks(text)
        assert regions == []

    def test_multiple_tables(self) -> None:
        """Multiple separate table regions are all detected."""
        from hse_prom_prog.rag.ingest import _detect_table_blocks

        text = (
            "Intro\n| A | B |\n| 1 | 2 |\n| 3 | 4 |\n"
            "Middle text\n| X | Y |\n| 5 | 6 |\n| 7 | 8 |\nEnd"
        )
        regions = _detect_table_blocks(text)
        assert len(regions) == 2


class TestSplitTextPreservingTables:
    """Tests for _split_text_preserving_tables."""

    def test_no_tables_splits_normally(self) -> None:
        """Text without tables is split by the parent splitter."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        from hse_prom_prog.rag.ingest import _split_text_preserving_tables

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "A" * 200
        segments = _split_text_preserving_tables(text, splitter)
        assert all(not is_table for _, is_table in segments)
        assert len(segments) >= 2

    def test_table_preserved_as_atomic(self) -> None:
        """Table region is kept as one atomic chunk."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        from hse_prom_prog.rag.ingest import _split_text_preserving_tables

        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Before\n| A | B |\n| 1 | 2 |\n| 3 | 4 |\nAfter"
        segments = _split_text_preserving_tables(text, splitter)
        table_segments = [(t, f) for t, f in segments if f]
        assert len(table_segments) == 1

    def test_mixed_content_preserves_order(self) -> None:
        """Non-table, table, non-table segments maintain document order."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        from hse_prom_prog.rag.ingest import _split_text_preserving_tables

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        text = "Before text.\n| A | B |\n| 1 | 2 |\n| 3 | 4 |\nAfter text."
        segments = _split_text_preserving_tables(text, splitter)
        flags = [is_table for _, is_table in segments]
        assert flags == [False, True, False]


class TestMarkdownSemantic:
    """Tests for _split_markdown_semantically."""

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_heading_based_split(self, mock_settings: MagicMock) -> None:
        """Markdown is split on headings; section metadata derived from heading."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_markdown_semantically

        mock_settings.parent_chunk_size = 1500
        mock_settings.parent_chunk_overlap = 200

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        doc = Document(
            page_content=("# Title\nIntro\n## Section A\nContent A\n## Section B\nContent B"),
            metadata={"source": "test.md", "category": "agile"},
        )
        parents = _split_markdown_semantically([doc], splitter)
        sections = [p.metadata["section"] for p in parents]
        assert "Section A" in sections
        assert "Section B" in sections

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_oversized_section_falls_back(self, mock_settings: MagicMock) -> None:
        """Section exceeding parent_chunk_size is further split."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_markdown_semantically

        mock_settings.parent_chunk_size = 100
        mock_settings.parent_chunk_overlap = 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        doc = Document(
            page_content="# Big Section\n" + "A " * 200,
            metadata={"source": "test.md", "category": "agile"},
        )
        parents = _split_markdown_semantically([doc], splitter)
        assert len(parents) > 1
        for p in parents:
            assert p.metadata["section"] == "Big Section"


class TestPdfWithTables:
    """Tests for _split_pdf_with_tables."""

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_table_gets_is_table_metadata(self, mock_settings: MagicMock) -> None:
        """PDF table region gets is_table=True metadata."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_pdf_with_tables

        mock_settings.parent_chunk_size = 500
        mock_settings.parent_chunk_overlap = 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        doc = Document(
            page_content="Intro text.\n| A | B |\n| 1 | 2 |\n| 3 | 4 |\nEnd text.",
            metadata={"source": "test.pdf", "category": "metrics", "page": 0},
        )
        parents = _split_pdf_with_tables([doc], splitter)
        table_parents = [p for p in parents if p.metadata.get("is_table")]
        assert len(table_parents) == 1
        assert table_parents[0].metadata["section"] == "Table"

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_non_table_text_gets_section(self, mock_settings: MagicMock) -> None:
        """Non-table PDF text gets section from _extract_section."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_pdf_with_tables

        mock_settings.parent_chunk_size = 500
        mock_settings.parent_chunk_overlap = 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        doc = Document(
            page_content="## My Section\nSome content here.",
            metadata={"source": "test.pdf", "category": "metrics", "page": 0},
        )
        parents = _split_pdf_with_tables([doc], splitter)
        assert parents[0].metadata["section"] == "My Section"
        assert parents[0].metadata.get("is_table") is False


class TestMetadataEnrichment:
    """Tests for metadata enrichment: filename, chunk_index, is_table."""

    def test_enrich_metadata_adds_filename(self) -> None:
        """_enrich_metadata adds filename from source path."""
        from pathlib import Path

        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _enrich_metadata

        doc = Document(
            page_content="test",
            metadata={"source": "/kb/metrics/done_total.pdf"},
        )
        _enrich_metadata(doc, Path("/kb"))
        assert doc.metadata["filename"] == "done_total.pdf"

    def test_enrich_metadata_empty_source(self) -> None:
        """_enrich_metadata handles empty source gracefully."""
        from pathlib import Path

        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _enrich_metadata

        doc = Document(page_content="test", metadata={"source": ""})
        _enrich_metadata(doc, Path("/kb"))
        assert doc.metadata["filename"] == "unknown"

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_split_parent_child_adds_chunk_index(self, mock_settings: MagicMock) -> None:
        """Child chunks get sequential chunk_index (0-based)."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_parent_child

        mock_settings.parent_chunk_size = 200
        mock_settings.parent_chunk_overlap = 0
        mock_settings.child_chunk_size = 50
        mock_settings.child_chunk_overlap = 0

        doc = Document(
            page_content="A" * 180,
            metadata={"source": "/path/to/doc.pdf", "category": "metrics"},
        )
        children = _split_parent_child([doc])
        indices = [c.metadata["chunk_index"] for c in children]
        assert indices == list(range(len(children)))

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_split_parent_child_filename_metadata(self, mock_settings: MagicMock) -> None:
        """Children preserve filename from enriched metadata."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_parent_child

        mock_settings.parent_chunk_size = 200
        mock_settings.parent_chunk_overlap = 0
        mock_settings.child_chunk_size = 50
        mock_settings.child_chunk_overlap = 0

        doc = Document(
            page_content="Content " * 30,
            metadata={
                "source": "/kb/metrics/done_total.pdf",
                "category": "metrics",
                "filename": "done_total.pdf",
            },
        )
        children = _split_parent_child([doc])
        for child in children:
            assert child.metadata["filename"] == "done_total.pdf"

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_split_parent_child_table_is_atomic(self, mock_settings: MagicMock) -> None:
        """PDF table chunks skip child splitting and have is_table=True."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_parent_child

        mock_settings.parent_chunk_size = 500
        mock_settings.parent_chunk_overlap = 0
        mock_settings.child_chunk_size = 50
        mock_settings.child_chunk_overlap = 0

        table_text = "| Col1 | Col2 | Col3 |\n| A | B | C |\n| D | E | F |\n| G | H | I |"
        doc = Document(
            page_content=table_text,
            metadata={"source": "/kb/metrics/done_total.pdf", "category": "metrics"},
        )
        children = _split_parent_child([doc])
        assert len(children) == 1
        assert children[0].metadata.get("is_table") is True
        assert children[0].metadata["chunk_index"] == 0

    @patch("hse_prom_prog.rag.ingest.settings")
    def test_split_parent_child_md_heading_section(self, mock_settings: MagicMock) -> None:
        """Markdown doc gets section from heading via MarkdownHeaderTextSplitter."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_parent_child

        mock_settings.parent_chunk_size = 1500
        mock_settings.parent_chunk_overlap = 0
        mock_settings.child_chunk_size = 50
        mock_settings.child_chunk_overlap = 0

        doc = Document(
            page_content="## My Heading\nContent about this topic " * 10,
            metadata={"source": "test.md", "category": "agile"},
        )
        children = _split_parent_child([doc])
        for child in children:
            assert child.metadata["section"] == "My Heading"


# ────────────────────────────────────────────────────────────────
# Faithfulness Check
# ────────────────────────────────────────────────────────────────


class TestFaithfulness:
    """Tests for faithfulness check, citation instructions, and conflict handling."""

    def _make_mock_doc(self, content: str, source: str = "doc.md", category: str = "agile"):
        """Create a mock LangChain Document."""
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source, "category": category}
        return doc

    # ── Unit tests for _check_faithfulness ────────────────────

    def test_faithfulness_check_passes(self) -> None:
        """LLM returns 'FAITHFUL' → check returns True."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "FAITHFUL"
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        assert agent._check_faithfulness("answer", "context") is True

    def test_faithfulness_check_fails(self) -> None:
        """LLM returns 'UNFAITHFUL' → check returns False."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "UNFAITHFUL"
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        assert agent._check_faithfulness("answer", "context") is False

    def test_faithfulness_check_with_extra_text(self) -> None:
        """Whitespace/case variations handled correctly."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        mock_llm.invoke.return_value = "  faithful  "
        assert agent._check_faithfulness("a", "c") is True

        mock_llm.invoke.return_value = "UNFAITHFUL - some claims not supported"
        assert agent._check_faithfulness("a", "c") is False

    # ── Prompt verification ───────────────────────────────────

    def test_generate_answer_prompt_has_citation_instructions(self) -> None:
        """Generation prompt includes citation format and conflict instructions."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "answer"
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        agent._generate_answer("test query", "test context", ["source.md"])

        prompt = mock_llm.invoke.call_args[0][0]
        assert "[имя_файла, стр. N]" in prompt
        assert "противоречие" in prompt

    def test_check_faithfulness_prompt_structure(self) -> None:
        """Faithfulness prompt contains context, answer, and verdict sections."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "FAITHFUL"
        mock_retriever = MagicMock()
        agent = RAGAgent(mock_llm, mock_retriever)

        agent._check_faithfulness("test answer", "test context")

        prompt = mock_llm.invoke.call_args[0][0]
        assert "КОНТЕКСТ:" in prompt
        assert "ОТВЕТ:" in prompt
        assert "Вердикт:" in prompt

    # ── Integration tests for process() faithfulness loop ─────

    def test_process_faithful_no_warning(self) -> None:
        """Faithful answer → no warning appended."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            "FAITHFUL",  # _check_faithfulness
        ]
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Context", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Final answer"
        assert "⚠️" not in result["rag_response"]

    def test_process_unfaithful_then_faithful_retry(self) -> None:
        """First check unfaithful, retry succeeds → no warning."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Bad answer",  # _generate_answer (1st)
            "UNFAITHFUL",  # _check_faithfulness (1st)
            "Good answer",  # _generate_answer (retry)
            "FAITHFUL",  # _check_faithfulness (retry)
        ]
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Context", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Good answer"
        assert "⚠️" not in result["rag_response"]

    def test_process_unfaithful_both_times_warning(self) -> None:
        """Both checks unfaithful → warning appended to retry answer."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Bad answer",  # _generate_answer (1st)
            "UNFAITHFUL",  # _check_faithfulness (1st)
            "Still bad answer",  # _generate_answer (retry)
            "UNFAITHFUL",  # _check_faithfulness (retry)
        ]
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Context", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert "⚠️" in result["rag_response"]
        assert "Still bad answer" in result["rag_response"]

    def test_process_faithfulness_check_error_skips(self) -> None:
        """Faithfulness check raises → original answer returned, no warning."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Final answer",  # _generate_answer
            ConnectionError("LLM down"),  # _check_faithfulness error
        ]
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Context", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Final answer"
        assert "⚠️" not in result["rag_response"]

    def test_process_faithfulness_retry_error(self) -> None:
        """Unfaithful + retry raises → warning on original answer."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "rewritten",  # _rewrite_query
            "variant 1",  # _generate_multi_queries
            "hyde doc",  # _generate_hyde_document
            "Bad answer",  # _generate_answer (1st)
            "UNFAITHFUL",  # _check_faithfulness (1st)
            ConnectionError("LLM down"),  # _generate_answer retry error
        ]
        mock_retriever = MagicMock()
        doc = self._make_mock_doc("Context", "doc.md")
        mock_retriever.invoke.return_value = [doc]

        agent = RAGAgent(mock_llm, mock_retriever)
        result = agent.process({"original_query": "test"})

        assert "⚠️" in result["rag_response"]
        assert "Bad answer" in result["rag_response"]


# ────────────────────────────────────────────────────────────────
# Validator Agent
# ────────────────────────────────────────────────────────────────


class TestValidatorAgent:
    """Tests for Validator agent."""

    def test_sql_query_with_data(self) -> None:
        """SQL query with valid data → use_sql=True."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "sql",
                "sql_result": [{"issue_key": "AL-123"}],
                "error": None,
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is True
        assert vr["use_rag"] is False
        assert vr["note"] is None

    def test_sql_query_with_error(self) -> None:
        """SQL query with error → use_sql=False, note contains error."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "sql",
                "sql_result": None,
                "error": "Connection refused",
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is False
        assert "Connection refused" in vr["note"]

    def test_sql_query_empty_result(self) -> None:
        """SQL query with empty result → use_sql=False."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "sql",
                "sql_result": [],
                "error": None,
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is False
        assert vr["note"] is not None

    def test_rag_query_with_response(self) -> None:
        """RAG query with response → use_rag=True."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "rag",
                "rag_response": "Scope Drop — это...",
                "rag_sources": ["metrics/scope.md"],
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is False
        assert vr["use_rag"] is True
        assert vr["note"] is None

    def test_rag_query_no_response(self) -> None:
        """RAG query with no response → use_rag=False."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "rag",
                "rag_response": None,
                "rag_sources": [],
            }
        )

        vr = result["validation_result"]
        assert vr["use_rag"] is False
        assert "No relevant documents" in vr["note"]

    def test_hybrid_both_available(self) -> None:
        """Hybrid query with both SQL and RAG data → both True."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "hybrid",
                "sql_result": [{"done_total": 85.0}],
                "error": None,
                "rag_response": "Рекомендации по улучшению...",
                "rag_sources": ["agile/practices.md"],
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is True
        assert vr["use_rag"] is True
        assert vr["note"] is None

    def test_hybrid_only_sql(self) -> None:
        """Hybrid with only SQL data → use_sql=True, use_rag=False."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "hybrid",
                "sql_result": [{"done_total": 85.0}],
                "error": None,
                "rag_response": None,
                "rag_sources": [],
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is True
        assert vr["use_rag"] is False

    def test_hybrid_only_rag(self) -> None:
        """Hybrid with only RAG data → use_sql=False, use_rag=True."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "hybrid",
                "sql_result": [],
                "error": None,
                "rag_response": "Рекомендации...",
                "rag_sources": ["agile/doc.md"],
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is False
        assert vr["use_rag"] is True

    def test_hybrid_both_failed(self) -> None:
        """Hybrid with no data from either agent → note set."""
        agent = ValidatorAgent()
        result = agent.process(
            {
                "query_type": "hybrid",
                "sql_result": [],
                "error": "DB error",
                "rag_response": None,
                "rag_sources": [],
            }
        )

        vr = result["validation_result"]
        assert vr["use_sql"] is False
        assert vr["use_rag"] is False
        assert vr["note"] is not None


# ────────────────────────────────────────────────────────────────
# Response Agent
# ────────────────────────────────────────────────────────────────


class TestResponseAgent:
    """Tests for Response agent."""

    def test_direct_response(self) -> None:
        """Direct response route calls LLM without DB data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Спринт — это итерация в Agile."
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "direct_response",
                "query_type": "simple",
                "intent": "general",
                "original_query": "Что такое спринт?",
            }
        )

        assert "Спринт" in result["final_response"]
        mock_llm.invoke.assert_called_once()

    def test_error_response(self) -> None:
        """SQL error is formatted into final_response."""
        mock_llm = MagicMock()
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "sql",
                "intent": "task",
                "original_query": "test",
                "error": "Connection refused",
                "sql_result": None,
                "validation_result": {"use_sql": False, "use_rag": False, "note": None},
            }
        )

        assert "Connection refused" in result["final_response"]
        mock_llm.invoke.assert_not_called()

    def test_empty_result_task(self) -> None:
        """Empty result for task intent shows issue_key not found."""
        mock_llm = MagicMock()
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "sql",
                "intent": "task",
                "original_query": "test",
                "entities": {"issue_key": "FAKE-999"},
                "sql_result": [],
                "error": None,
                "validation_result": {"use_sql": False, "use_rag": False, "note": None},
            }
        )

        assert "FAKE-999" in result["final_response"]
        assert "не найдена" in result["final_response"]

    def test_empty_result_filter(self) -> None:
        """Empty result for filter intent shows generic message."""
        mock_llm = MagicMock()
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "sql",
                "intent": "tasks_filter",
                "original_query": "test",
                "entities": {"team_name": "nonexistent"},
                "sql_result": [],
                "error": None,
                "validation_result": {"use_sql": False, "use_rag": False, "note": None},
            }
        )

        assert "ничего не найдено" in result["final_response"]

    def test_task_response(self) -> None:
        """Single task result calls LLM with task data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Задача AL-38787 в статусе In Progress."
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "sql",
                "intent": "task",
                "original_query": "Расскажи о задаче AL-38787",
                "entities": {"issue_key": "AL-38787"},
                "sql_result": [{"issue_key": "AL-38787", "issue_status_act": "In Progress"}],
                "error": None,
                "validation_result": {"use_sql": True, "use_rag": False, "note": None},
            }
        )

        assert "AL-38787" in result["final_response"]

    def test_metric_response(self) -> None:
        """Metric result calls LLM with metric data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Done total команды cthulhu: 85%."
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "sql",
                "intent": "metric",
                "original_query": "Done total cthulhu",
                "entities": {"team_name": "cthulhu", "metric_name": "done_total"},
                "sql_result": [
                    {"feature_teams": "cthulhu", "sprint_name": "#1 Q1'26", "done_total": 85.0}
                ],
                "error": None,
                "validation_result": {"use_sql": True, "use_rag": False, "note": None},
            }
        )

        assert "85" in result["final_response"]

    def test_llm_error_fallback(self) -> None:
        """LLM error produces fallback message, not crash."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("vllm down")
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "sql",
                "intent": "task",
                "original_query": "test",
                "entities": {"issue_key": "AL-123"},
                "sql_result": [{"issue_key": "AL-123"}],
                "error": None,
                "validation_result": {"use_sql": True, "use_rag": False, "note": None},
            }
        )

        assert "Не удалось" in result["final_response"]

    def test_rag_response(self) -> None:
        """RAG-only query returns formatted response with sources."""
        mock_llm = MagicMock()
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "rag",
                "intent": "general",
                "original_query": "Что такое Definition of Done?",
                "rag_response": "DoD — это набор критериев.",
                "rag_sources": ["agile/dod.md"],
                "validation_result": {"use_sql": False, "use_rag": True, "note": None},
            }
        )

        assert "DoD — это набор критериев." in result["final_response"]
        assert "agile/dod.md" in result["final_response"]

    def test_rag_response_no_sources(self) -> None:
        """RAG response without sources omits sources section."""
        mock_llm = MagicMock()
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "rag",
                "intent": "general",
                "original_query": "test",
                "rag_response": "Ответ без источников.",
                "rag_sources": [],
                "validation_result": {"use_sql": False, "use_rag": True, "note": None},
            }
        )

        assert "Ответ без источников." in result["final_response"]
        assert "Источники" not in result["final_response"]

    def test_hybrid_response(self) -> None:
        """Hybrid query combines DB data with RAG context."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Scope Drop 15%. Рекомендуем уменьшить WIP."
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "hybrid",
                "intent": "metric",
                "original_query": "Scope Drop cthulhu и рекомендации",
                "sql_result": [{"feature_teams": "cthulhu", "scope_drop": 15.0}],
                "error": None,
                "rag_response": "Для снижения Scope Drop рекомендуется...",
                "rag_sources": ["metrics/scope_drop.md"],
                "validation_result": {"use_sql": True, "use_rag": True, "note": None},
            }
        )

        assert "Scope Drop" in result["final_response"]
        assert "metrics/scope_drop.md" in result["final_response"]
        mock_llm.invoke.assert_called_once()

    def test_both_failed_response(self) -> None:
        """Both SQL and RAG failed → error message from validation note."""
        mock_llm = MagicMock()
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "hybrid",
                "intent": "metric",
                "original_query": "test",
                "sql_result": [],
                "error": "DB error",
                "rag_response": None,
                "rag_sources": [],
                "validation_result": {
                    "use_sql": False,
                    "use_rag": False,
                    "note": "No data from SQL or RAG",
                },
            }
        )

        assert "No data from SQL or RAG" in result["final_response"]

    def test_simple_query_type_triggers_direct(self) -> None:
        """query_type=simple triggers direct response even with db_query route."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Привет! Чем могу помочь?"
        agent = ResponseAgent(mock_llm)

        result = agent.process(
            {
                "route": "db_query",
                "query_type": "simple",
                "intent": "general",
                "original_query": "Привет",
            }
        )

        assert "Привет" in result["final_response"]
        mock_llm.invoke.assert_called_once()


# ────────────────────────────────────────────────────────────────
# Workflow Integration
# ────────────────────────────────────────────────────────────────


class TestAgileWorkflow:
    """Tests for the complete workflow."""

    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ValidatorAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_sql_route(
        self,
        mock_response_cls: MagicMock,
        mock_validator_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
    ) -> None:
        """SQL route: Supervisor → SQL Agent → Validator → Response Agent."""
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "original_query": "Данные по AL-38787",
            "intent": "task",
            "entities": {"issue_key": "AL-38787"},
            "query_type": "sql",
            "route": "db_query",
        }
        mock_supervisor_cls.return_value = mock_supervisor

        mock_sql = MagicMock()
        mock_sql.process.return_value = {
            "sql_query": "SELECT * FROM ...",
            "sql_result": [{"issue_key": "AL-38787"}],
            "error": None,
        }
        mock_sql_cls.return_value = mock_sql

        mock_validator = MagicMock()
        mock_validator.process.return_value = {
            "validation_result": {"use_sql": True, "use_rag": False, "note": None},
        }
        mock_validator_cls.return_value = mock_validator

        mock_response = MagicMock()
        mock_response.process.return_value = {
            "final_response": "Задача AL-38787 найдена.",
        }
        mock_response_cls.return_value = mock_response

        mock_llm = MagicMock()
        workflow = AgileWorkflow(mock_llm)
        result = workflow.run("Данные по AL-38787")

        assert "final_response" in result
        mock_supervisor.process.assert_called_once()
        mock_sql.process.assert_called_once()
        mock_validator.process.assert_called_once()
        mock_response.process.assert_called_once()

    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ValidatorAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_simple_route(
        self,
        mock_response_cls: MagicMock,
        mock_validator_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
    ) -> None:
        """Simple route: Supervisor → Response Agent (skips SQL, Validator)."""
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "original_query": "Что такое спринт?",
            "intent": "general",
            "entities": {},
            "query_type": "simple",
            "route": "direct_response",
        }
        mock_supervisor_cls.return_value = mock_supervisor

        mock_sql = MagicMock()
        mock_sql_cls.return_value = mock_sql

        mock_validator = MagicMock()
        mock_validator_cls.return_value = mock_validator

        mock_response = MagicMock()
        mock_response.process.return_value = {
            "final_response": "Спринт — это итерация.",
        }
        mock_response_cls.return_value = mock_response

        mock_llm = MagicMock()
        workflow = AgileWorkflow(mock_llm)
        result = workflow.run("Что такое спринт?")

        assert "final_response" in result
        mock_supervisor.process.assert_called_once()
        mock_sql.process.assert_not_called()
        mock_validator.process.assert_not_called()
        mock_response.process.assert_called_once()

    @patch("hse_prom_prog.graph.workflow.RAGAgent")
    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ValidatorAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_rag_route(
        self,
        mock_response_cls: MagicMock,
        mock_validator_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
        mock_rag_cls: MagicMock,
    ) -> None:
        """RAG route: Supervisor → RAG Agent → Validator → Response Agent."""
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "original_query": "Что такое Definition of Done?",
            "intent": "general",
            "entities": {},
            "query_type": "rag",
            "route": "db_query",
        }
        mock_supervisor_cls.return_value = mock_supervisor

        mock_sql = MagicMock()
        mock_sql_cls.return_value = mock_sql

        mock_rag = MagicMock()
        mock_rag.process.return_value = {
            "rag_response": "DoD — это набор критериев.",
            "rag_sources": ["agile/dod.md"],
        }
        mock_rag_cls.return_value = mock_rag

        mock_validator = MagicMock()
        mock_validator.process.return_value = {
            "validation_result": {"use_sql": False, "use_rag": True, "note": None},
        }
        mock_validator_cls.return_value = mock_validator

        mock_response = MagicMock()
        mock_response.process.return_value = {
            "final_response": "DoD — это набор критериев.\n\nИсточники:\n- agile/dod.md",
        }
        mock_response_cls.return_value = mock_response

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        workflow = AgileWorkflow(mock_llm, retriever=mock_retriever)
        result = workflow.run("Что такое Definition of Done?")

        assert "final_response" in result
        mock_supervisor.process.assert_called_once()
        mock_sql.process.assert_not_called()
        mock_rag.process.assert_called_once()
        mock_validator.process.assert_called_once()
        mock_response.process.assert_called_once()

    @patch("hse_prom_prog.graph.workflow.RAGAgent")
    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ValidatorAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_hybrid_route(
        self,
        mock_response_cls: MagicMock,
        mock_validator_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
        mock_rag_cls: MagicMock,
    ) -> None:
        """Hybrid route: Supervisor → SQL+RAG → Validator → Response Agent."""
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "original_query": "Scope drop cthulhu и рекомендации",
            "intent": "metric",
            "entities": {"team_name": "cthulhu", "metric_name": "scope_drop"},
            "query_type": "hybrid",
            "route": "db_query",
        }
        mock_supervisor_cls.return_value = mock_supervisor

        mock_sql = MagicMock()
        mock_sql.process.return_value = {
            "sql_query": "SELECT ...",
            "sql_result": [{"scope_drop": 15.0}],
            "error": None,
        }
        mock_sql_cls.return_value = mock_sql

        mock_rag = MagicMock()
        mock_rag.process.return_value = {
            "rag_response": "Рекомендации по снижению...",
            "rag_sources": ["metrics/scope_drop.md"],
        }
        mock_rag_cls.return_value = mock_rag

        mock_validator = MagicMock()
        mock_validator.process.return_value = {
            "validation_result": {"use_sql": True, "use_rag": True, "note": None},
        }
        mock_validator_cls.return_value = mock_validator

        mock_response = MagicMock()
        mock_response.process.return_value = {
            "final_response": "Scope Drop 15%. Рекомендации...",
        }
        mock_response_cls.return_value = mock_response

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        workflow = AgileWorkflow(mock_llm, retriever=mock_retriever)
        result = workflow.run("Scope drop cthulhu и рекомендации")

        assert "final_response" in result
        mock_supervisor.process.assert_called_once()
        mock_sql.process.assert_called_once()
        mock_rag.process.assert_called_once()
        mock_validator.process.assert_called_once()
        mock_response.process.assert_called_once()

    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ValidatorAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_workflow_without_retriever(
        self,
        mock_response_cls: MagicMock,
        mock_validator_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
    ) -> None:
        """Workflow without retriever: RAG agent not created."""
        mock_supervisor_cls.return_value = MagicMock()
        mock_sql_cls.return_value = MagicMock()
        mock_validator_cls.return_value = MagicMock()
        mock_response_cls.return_value = MagicMock()

        mock_llm = MagicMock()
        workflow = AgileWorkflow(mock_llm)

        assert workflow.rag_agent is None


# ────────────────────────────────────────────────────────────────
# Parametrized
# ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,expected_key",
    [
        ("Выведи данные по задаче ABC-123", "ABC-123"),
        ("PROJ-456", "PROJ-456"),
        ("Информация о DEV-999 нужна", "DEV-999"),
        ("Задача XY-1", "XY-1"),
        ("LONG-12345", "LONG-12345"),
    ],
)
def test_issue_key_extraction_parametrized(query: str, expected_key: str) -> None:
    """Parametrized test for issue key extraction via regex fast path."""
    mock_llm = MagicMock()
    agent = SupervisorAgent(mock_llm)

    result = agent.process(query)

    assert result["entities"]["issue_key"] == expected_key
    assert result["intent"] == "task"
    assert result["query_type"] == "sql"
    assert result["route"] == "db_query"


@pytest.mark.parametrize(
    "query",
    [
        "Что такое спринт?",
        "Привет",
        "Как рассчитываются story points?",
    ],
)
def test_general_queries_no_regex(query: str) -> None:
    """General queries without issue keys go to LLM classification."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = '{"intent": "general", "query_type": "simple", "entities": {}}'
    agent = SupervisorAgent(mock_llm)

    result = agent.process(query)

    assert result["intent"] == "general"
    assert result["query_type"] == "simple"
    assert result["route"] == "direct_response"
    mock_llm.invoke.assert_called_once()
