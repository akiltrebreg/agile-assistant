"""Tests for the multi-agent workflow components.

Tests cover:
- Supervisor: regex fast path, LLM classification, JSON parsing
- SQL Agent: query building for all intents, validation
- Response Agent: direct responses, DB-based responses, error handling
- Workflow: end-to-end integration with mocked agents
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from hse_prom_prog.agents.response_agent import ResponseAgent
from hse_prom_prog.agents.sql_agent import SQLAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent
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
        assert result["route"] == "db_query"
        mock_llm.invoke.assert_not_called()

    def test_fast_path_with_complex_key(self) -> None:
        """Regex handles multi-letter prefixes and long numbers."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Посмотри PROJ-456789")

        assert result["entities"]["issue_key"] == "PROJ-456789"
        assert result["intent"] == "task"

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
            '{"intent": "tasks_filter", "entities": {"team_name": "cthulhu", "sprint_name": null}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Все задачи команды cthulhu")

        assert result["intent"] == "tasks_filter"
        assert result["entities"]["team_name"] == "cthulhu"
        assert result["route"] == "db_query"

    def test_llm_classification_metric(self) -> None:
        """LLM classifies query as metric."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "metric", "entities": {"team_name": "lpop", "metric_name": "done_total"}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Done total команды lpop")

        assert result["intent"] == "metric"
        assert result["entities"]["metric_name"] == "done_total"
        assert result["route"] == "db_query"

    def test_llm_classification_general(self) -> None:
        """LLM classifies query as general."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"intent": "general", "entities": {}}'
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Что такое спринт?")

        assert result["intent"] == "general"
        assert result["route"] == "direct_response"

    def test_llm_returns_markdown_json(self) -> None:
        """LLM wraps JSON in markdown code fences — parsed correctly."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '```json\n{"intent": "general", "entities": {}}\n```'
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Привет")

        assert result["intent"] == "general"

    def test_llm_returns_garbage(self) -> None:
        """LLM returns unparsable text → fallback to general."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "I don't understand"
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Что-то непонятное")

        assert result["intent"] == "general"
        assert result["route"] == "direct_response"

    def test_llm_exception_fallback(self) -> None:
        """LLM raises exception → fallback to general."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("timeout")
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Задачи команды lpop")

        assert result["intent"] == "general"
        assert result["route"] == "direct_response"

    def test_null_entities_cleaned(self) -> None:
        """Null-valued entities are removed from the dict."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"intent": "tasks_filter", '
            '"entities": {"team_name": "cthulhu", "sprint_name": null, '
            '"issue_type": null, "status": null}}'
        )
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Задачи cthulhu")

        assert "sprint_name" not in result["entities"]
        assert result["entities"] == {"team_name": "cthulhu"}


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
        agent = SQLAgent()
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
                "intent": "task",
                "original_query": "test",
                "error": "Connection refused",
                "sql_result": None,
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
                "intent": "task",
                "original_query": "test",
                "entities": {"issue_key": "FAKE-999"},
                "sql_result": [],
                "error": None,
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
                "intent": "tasks_filter",
                "original_query": "test",
                "entities": {"team_name": "nonexistent"},
                "sql_result": [],
                "error": None,
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
                "intent": "task",
                "original_query": "Расскажи о задаче AL-38787",
                "entities": {"issue_key": "AL-38787"},
                "sql_result": [{"issue_key": "AL-38787", "issue_status_act": "In Progress"}],
                "error": None,
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
                "intent": "metric",
                "original_query": "Done total cthulhu",
                "entities": {"team_name": "cthulhu", "metric_name": "done_total"},
                "sql_result": [
                    {"feature_teams": "cthulhu", "sprint_name": "#1 Q1'26", "done_total": 85.0}
                ],
                "error": None,
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
                "intent": "task",
                "original_query": "test",
                "entities": {"issue_key": "AL-123"},
                "sql_result": [{"issue_key": "AL-123"}],
                "error": None,
            }
        )

        assert "Не удалось" in result["final_response"]


# ────────────────────────────────────────────────────────────────
# Workflow Integration
# ────────────────────────────────────────────────────────────────


class TestAgileWorkflow:
    """Tests for the complete workflow."""

    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_db_query_route(
        self,
        mock_response_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
    ) -> None:
        """DB query route: Supervisor → SQL Agent → Response Agent."""
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "original_query": "Данные по AL-38787",
            "intent": "task",
            "entities": {"issue_key": "AL-38787"},
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
        mock_response.process.assert_called_once()

    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_direct_response_route(
        self,
        mock_response_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
    ) -> None:
        """Direct route: Supervisor → Response Agent (skips SQL Agent)."""
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "original_query": "Что такое спринт?",
            "intent": "general",
            "entities": {},
            "route": "direct_response",
        }
        mock_supervisor_cls.return_value = mock_supervisor

        mock_sql = MagicMock()
        mock_sql_cls.return_value = mock_sql

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
        mock_response.process.assert_called_once()


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
    mock_llm.invoke.return_value = '{"intent": "general", "entities": {}}'
    agent = SupervisorAgent(mock_llm)

    result = agent.process(query)

    assert result["intent"] == "general"
    assert result["route"] == "direct_response"
    mock_llm.invoke.assert_called_once()
