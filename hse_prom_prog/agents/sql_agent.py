"""SQL agent with mock data functionality.

This agent simulates database queries for Jira issues. In this version,
it returns mock data instead of performing real SQL queries.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SQLAgent:
    """Agent that simulates SQL queries to retrieve Jira issue data.

    This is a mock implementation that returns placeholder data.
    In the next iteration, this will connect to a real Postgres database.

    Attributes:
        None (stateless agent for mock implementation).
    """

    def __init__(self) -> None:
        """Initialize the SQL agent."""
        logger.info("[SQL Agent] Initialized with mock data functionality")

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the request and return mock SQL query results.

        Args:
            state: State dictionary containing 'issue_key' from Supervisor.

        Returns:
            Dictionary containing the mock SQL response.
        """
        issue_key = state.get("issue_key", "UNKNOWN")
        logger.info(f"[SQL Agent] Processing request for issue: {issue_key}")

        # Mock response message
        mock_response = (
            f"Привет! Во втором задании я научусь отправлять запрос в Postgres "
            f"и пришлю данные по задаче {issue_key}!"
        )

        logger.info(f"[SQL Agent] Returning mock response: {mock_response}")

        return {
            "issue_key": issue_key,
            "original_query": state.get("original_query", ""),
            "sql_response": mock_response,
        }
