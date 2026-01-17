"""Response agent for formatting final user responses.

This agent takes the SQL agent's output and formats it into a
user-friendly markdown response.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResponseAgent:
    """Agent that formats final responses for users.

    The Response agent takes data from previous agents and creates
    a well-formatted, user-friendly output message.

    Attributes:
        None (stateless agent).
    """

    def __init__(self) -> None:
        """Initialize the Response agent."""
        logger.info("[Response Agent] Initialized")

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Format the final response for the user.

        Args:
            state: State dictionary containing 'sql_response' from SQL Agent.

        Returns:
            Dictionary containing the formatted final response.
        """
        logger.info("[Response Agent] Formatting final response...")

        sql_response = state.get("sql_response", "No data available")
        issue_key = state.get("issue_key", "UNKNOWN")

        # Format the response in markdown
        formatted_response = f"""
## Результат обработки задачи {issue_key}

{sql_response}

---
*Запрос обработан успешно*
        """.strip()

        logger.info("[Response Agent] Response formatted successfully")

        return {
            "issue_key": issue_key,
            "original_query": state.get("original_query", ""),
            "sql_response": sql_response,
            "final_response": formatted_response,
        }
