"""SQL agent for querying Jira issues from PostgreSQL.

This agent retrieves Jira issue data from PostgreSQL database
using SQLAlchemy for database operations.
"""

import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.database.connection import DatabaseConnection, get_database

logger = logging.getLogger(__name__)


class SQLAgent:
    """Agent that queries PostgreSQL database for Jira issue data.

    This agent executes SQL queries against the jira_issues table
    and retrieves complete issue information.

    Attributes:
        db: Database connection instance.
    """

    def __init__(self, db_connection: DatabaseConnection | None = None) -> None:
        """Initialize the SQL agent.

        Args:
            db_connection: Optional database connection. If not provided,
                          a new connection will be created.
        """
        self.db = db_connection
        if self.db:
            logger.info("[SQL Agent] Initialized with provided database connection")
        else:
            logger.info("[SQL Agent] Initialized (database connection will be created on demand)")

    def _ensure_db_connection(self) -> None:
        """Ensure database connection exists, create if needed."""
        if not self.db:
            self.db = get_database()
            logger.info("[SQL Agent] Created new database connection")

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the request and query PostgreSQL for issue data.

        Args:
            state: State dictionary containing 'issue_key' from Supervisor.

        Returns:
            Dictionary containing the SQL query results.
        """
        issue_key = state.get("issue_key", "UNKNOWN")
        logger.info(f"[SQL Agent] Processing request for issue: {issue_key}")

        # Ensure database connection
        self._ensure_db_connection()

        try:
            # Query the database
            query = """
                SELECT
                    issue_key,
                    jirasprint_id,
                    sprint_name,
                    start_date,
                    end_date,
                    sprint_state,
                    issue_project,
                    issue_type,
                    feature_teams,
                    reporter,
                    assignee_name,
                    issue_status_end_of_sprint,
                    storypoints_start_of_sprint,
                    storypoints_end_of_sprint,
                    time_h_in_progress,
                    time_h_not_fixed,
                    create_time,
                    resolution,
                    labels,
                    cluster,
                    unit
                FROM jira_issues
                WHERE issue_key = :issue_key
            """

            results = self.db.execute_query(query, {"issue_key": issue_key})

            if not results:
                logger.warning(f"[SQL Agent] No data found for issue: {issue_key}")
                return {
                    "issue_key": issue_key,
                    "original_query": state.get("original_query", ""),
                    "sql_response": None,
                    "error": f"Issue {issue_key} not found in database",
                }

            logger.info(f"[SQL Agent] Successfully retrieved data for issue: {issue_key}")

            return {
                "issue_key": issue_key,
                "original_query": state.get("original_query", ""),
                "sql_response": results[0],  # Return first result (issue_key is unique)
                "error": None,
            }

        except SQLAlchemyError as e:
            logger.error(f"[SQL Agent] Database error while querying issue {issue_key}: {e}")
            return {
                "issue_key": issue_key,
                "original_query": state.get("original_query", ""),
                "sql_response": None,
                "error": f"Database error: {e!s}",
            }
        except Exception as e:
            logger.error(f"[SQL Agent] Unexpected error while processing issue {issue_key}: {e}")
            return {
                "issue_key": issue_key,
                "original_query": state.get("original_query", ""),
                "sql_response": None,
                "error": f"Unexpected error: {e!s}",
            }
