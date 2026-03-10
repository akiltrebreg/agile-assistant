"""Tests for the multi-agent workflow components.

Tests cover:
- Supervisor: regex fast path, LLM classification, JSON parsing, query_type
- SQL Agent: query building for all intents, validation
- RAG Agent: retrieval, LLM generation, empty results, errors
- Ingestion: splitting, metadata enrichment
- Validator Agent: sql-only, rag-only, hybrid, both-failed scenarios
- Response Agent: direct, DB-based, RAG, hybrid, error handling
- Workflow: end-to-end integration with mocked agents (all 4 routes)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

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
    """Tests for simplified RAG agent."""

    def _make_mock_doc(self, content: str, source: str = "doc.md", category: str = "agile"):
        """Create a mock LangChain Document."""
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {"source": source, "category": category}
        return doc

    def test_process_success(self) -> None:
        """RAG Agent retrieves docs and generates answer."""
        from hse_prom_prog.agents.rag_agent import RAGAgent

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
        from hse_prom_prog.agents.rag_agent import RAGAgent

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "Непонятный запрос"})

        assert result["rag_response"] is None
        assert result["rag_sources"] == []

    def test_process_retrieval_error(self) -> None:
        """Retrieval failure → rag_response is None with error."""
        from hse_prom_prog.agents.rag_agent import RAGAgent

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = ConnectionError("Qdrant down")
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "test"})

        assert result["rag_response"] is None
        assert result["rag_sources"] == []
        assert "RAG retrieval error" in result["error"]

    def test_process_llm_error(self) -> None:
        """LLM generation error is caught; sources still returned."""
        from hse_prom_prog.agents.rag_agent import RAGAgent

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

    def test_deduplicates_sources(self) -> None:
        """Same source appearing in multiple docs is deduplicated."""
        from hse_prom_prog.agents.rag_agent import RAGAgent

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

    def test_all_chunks_in_context(self) -> None:
        """All retrieved chunks are included in LLM context (no truncation)."""
        from hse_prom_prog.agents.rag_agent import RAGAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            self._make_mock_doc("A" * 3000, "doc1.md"),
            self._make_mock_doc("B" * 3000, "doc2.md"),
        ]
        agent = RAGAgent(mock_llm, mock_retriever)

        result = agent.process({"original_query": "test"})

        assert result["rag_response"] == "Answer"
        # Both sources included — no truncation
        assert len(result["rag_sources"]) == 2

    def test_prompt_contains_context_and_question(self) -> None:
        """Generated prompt contains context and question sections."""
        from hse_prom_prog.agents.rag_agent import RAGAgent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            self._make_mock_doc("Important context here"),
        ]
        agent = RAGAgent(mock_llm, mock_retriever)

        agent.process({"original_query": "My question?"})

        prompt = mock_llm.invoke.call_args[0][0]
        assert "Important context here" in prompt
        assert "My question?" in prompt


# ────────────────────────────────────────────────────────────────
# Ingestion Pipeline
# ────────────────────────────────────────────────────────────────


class TestIngestion:
    """Tests for simplified ingestion pipeline."""

    def test_split_documents_produces_chunks(self) -> None:
        """_split_documents splits long text into chunks."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _split_documents

        docs = [Document(page_content="word " * 500, metadata={"source": "test.md"})]
        chunks = _split_documents(docs)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 1000 + 50  # small margin for word boundaries

    def test_enrich_metadata_adds_category(self) -> None:
        """_enrich_metadata extracts category from sub-folder name."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _enrich_metadata

        kb_dir = Path("/data/knowledge_base")
        doc = Document(
            page_content="test",
            metadata={"source": "/data/knowledge_base/metrics/scope_drop.md"},
        )
        _enrich_metadata(doc, kb_dir)

        assert doc.metadata["category"] == "metrics"
        assert "ingested_at" in doc.metadata

    def test_enrich_metadata_general_fallback(self) -> None:
        """_enrich_metadata falls back to 'general' for root-level files."""
        from langchain_core.documents import Document

        from hse_prom_prog.rag.ingest import _enrich_metadata

        kb_dir = Path("/data/knowledge_base")
        doc = Document(
            page_content="test",
            metadata={"source": "/data/knowledge_base/readme.md"},
        )
        _enrich_metadata(doc, kb_dir)

        assert doc.metadata["category"] == "general"


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
