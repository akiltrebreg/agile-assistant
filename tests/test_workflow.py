"""Tests for the LangGraph workflow.

This module contains tests for the multi-agent workflow components.
"""

from unittest.mock import MagicMock, patch

import pytest

from hse_prom_prog.agents.response_agent import ResponseAgent
from hse_prom_prog.agents.sql_agent import SQLAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent
from hse_prom_prog.graph.workflow import AgileWorkflow


class TestSupervisorAgent:
    """Tests for Supervisor agent."""

    def test_extract_issue_key_with_regex_simple(self) -> None:
        """Test extracting issue key using regex with simple query."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Выведи данные по задаче ABC-123")

        assert result["issue_key"] == "ABC-123"
        assert result["original_query"] == "Выведи данные по задаче ABC-123"

    def test_extract_issue_key_with_regex_complex(self) -> None:
        """Test extracting issue key from complex query."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Привет! Мне нужна информация о задаче PROJ-456, срочно!")

        assert result["issue_key"] == "PROJ-456"

    def test_extract_issue_key_fallback_to_llm(self) -> None:
        """Test falling back to LLM when regex fails."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "XYZ-789"
        agent = SupervisorAgent(mock_llm)

        # Query without clear pattern
        agent.process("Покажи задачу экс уай зет семьсот восемьдесят девять")

        # Should still try LLM
        mock_llm.invoke.assert_called_once()

    def test_extract_multiple_keys_returns_first(self) -> None:
        """Test that when multiple keys present, first one is returned."""
        mock_llm = MagicMock()
        agent = SupervisorAgent(mock_llm)

        result = agent.process("Сравни ABC-123 и DEF-456")

        # Should return first match
        assert result["issue_key"] == "ABC-123"


class TestSQLAgent:
    """Tests for SQL agent."""

    def test_process_returns_mock_data(self) -> None:
        """Test that SQL agent returns mock data."""
        agent = SQLAgent()

        state = {
            "issue_key": "TEST-999",
            "original_query": "Test query",
        }

        result = agent.process(state)

        assert result["issue_key"] == "TEST-999"
        assert result["original_query"] == "Test query"
        assert "Привет! Во втором задании" in result["sql_response"]
        assert "TEST-999" in result["sql_response"]

    def test_process_handles_unknown_key(self) -> None:
        """Test SQL agent handles missing issue key."""
        agent = SQLAgent()

        state = {"original_query": "Test query"}

        result = agent.process(state)

        assert result["issue_key"] == "UNKNOWN"
        assert "UNKNOWN" in result["sql_response"]


class TestResponseAgent:
    """Tests for Response agent."""

    def test_format_response(self) -> None:
        """Test response formatting."""
        agent = ResponseAgent()

        state = {
            "issue_key": "ABC-123",
            "original_query": "Test query",
            "sql_response": "Test SQL response data",
        }

        result = agent.process(state)

        assert result["issue_key"] == "ABC-123"
        assert "ABC-123" in result["final_response"]
        assert "Test SQL response data" in result["final_response"]
        assert "##" in result["final_response"]  # Markdown heading

    def test_format_response_with_missing_data(self) -> None:
        """Test response formatting with missing data."""
        agent = ResponseAgent()

        state = {}

        result = agent.process(state)

        assert "UNKNOWN" in result["final_response"]
        assert "No data available" in result["final_response"]


class TestAgileWorkflow:
    """Tests for the complete workflow."""

    @patch("hse_prom_prog.graph.workflow.SupervisorAgent")
    @patch("hse_prom_prog.graph.workflow.SQLAgent")
    @patch("hse_prom_prog.graph.workflow.ResponseAgent")
    def test_workflow_integration(
        self,
        mock_response_cls: MagicMock,
        mock_sql_cls: MagicMock,
        mock_supervisor_cls: MagicMock,
    ) -> None:
        """Test the complete workflow integration."""
        # Setup mocks
        mock_supervisor = MagicMock()
        mock_supervisor.process.return_value = {
            "issue_key": "ABC-123",
            "original_query": "Test query",
        }
        mock_supervisor_cls.return_value = mock_supervisor

        mock_sql = MagicMock()
        mock_sql.process.return_value = {
            "issue_key": "ABC-123",
            "original_query": "Test query",
            "sql_response": "Mock SQL response",
        }
        mock_sql_cls.return_value = mock_sql

        mock_response = MagicMock()
        mock_response.process.return_value = {
            "issue_key": "ABC-123",
            "original_query": "Test query",
            "sql_response": "Mock SQL response",
            "final_response": "Formatted response",
        }
        mock_response_cls.return_value = mock_response

        # Create workflow
        mock_llm = MagicMock()
        workflow = AgileWorkflow(mock_llm)

        # Run workflow
        result = workflow.run("Выведи данные по задаче ABC-123")

        # Verify
        assert "final_response" in result
        mock_supervisor.process.assert_called_once()
        mock_sql.process.assert_called_once()
        mock_response.process.assert_called_once()


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
    """Parametrized test for issue key extraction."""
    mock_llm = MagicMock()
    agent = SupervisorAgent(mock_llm)

    result = agent.process(query)

    assert result["issue_key"] == expected_key
