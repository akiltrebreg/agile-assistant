"""Supervisor agent for extracting Jira issue keys from user queries.

This agent is responsible for parsing user requests and extracting
Jira issue keys in the format XXX-NNN.
"""

import logging
import re
from typing import Any

from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Agent that extracts Jira issue keys from user queries.

    The Supervisor agent uses an LLM to identify and extract Jira issue
    keys (e.g., ABC-123) from natural language user requests.

    Attributes:
        llm_client: LLM client for generating responses.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the Supervisor agent.

        Args:
            llm_client: LLM client instance for text generation.
        """
        self.llm_client = llm_client

    def _extract_issue_key_with_regex(self, text: str) -> str | None:
        """Extract Jira issue key using regex pattern.

        Args:
            text: Text to search for issue key.

        Returns:
            Extracted issue key or None if not found.
        """
        # Pattern matches: 2+ uppercase letters, hyphen, 1+ digits
        pattern = r"\b([A-Z]{2,}-\d+)\b"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def process(self, user_query: str) -> dict[str, Any]:
        """Process user query and extract Jira issue key.

        This method first attempts regex extraction, then uses LLM if needed.

        Args:
            user_query: The user's natural language query.

        Returns:
            Dictionary containing the extracted issue_key and original query.
        """
        logger.info(f"[Supervisor] Processing query: {user_query}")

        # Try regex extraction first
        issue_key = self._extract_issue_key_with_regex(user_query)

        if issue_key:
            logger.info(f"[Supervisor] Found issue key with regex: {issue_key}")
        else:
            # Fall back to LLM extraction
            logger.info("[Supervisor] Regex failed, using LLM for extraction...")
            prompt = (
                f"Извлеки ключ Jira-задачи из запроса. "
                f"Верни только ключ в формате XXX-NNN. "
                f"Если ключ не найден, верни 'NOT_FOUND'. "
                f"Запрос: {user_query}"
            )

            try:
                llm_response = self.llm_client.invoke(prompt)
                # Extract using regex from LLM response
                issue_key = self._extract_issue_key_with_regex(llm_response)
                if issue_key:
                    logger.info(f"[Supervisor] Found issue key with LLM: {issue_key}")
                else:
                    logger.warning("[Supervisor] Could not extract issue key")
                    issue_key = "NOT_FOUND"
            except Exception as e:
                logger.error(f"[Supervisor] Error during LLM extraction: {e}")
                issue_key = "ERROR"

        return {
            "issue_key": issue_key,
            "original_query": user_query,
        }
