"""SQL agent for querying Jira data from PostgreSQL.

This agent builds SQL queries from structured intent + entities
(provided by the Supervisor) using safe, template-based construction.
No raw LLM-generated SQL is executed — all queries are assembled in code.
"""

import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.agents.schema_description import SQL_MAX_ROWS
from hse_prom_prog.database.connection import DatabaseConnection, get_database

logger = logging.getLogger(__name__)


class SQLAgent:
    """Agent that builds and executes SQL queries based on intent + entities.

    Supports three intent types:
    - task: lookup a single task by issue_key
    - tasks_filter: search tasks by filters (team, sprint, type, status, ...)
    - metric: retrieve aggregated metrics from the metrics table

    Attributes:
        db: Database connection instance.
    """

    def __init__(self, db_connection: DatabaseConnection | None = None) -> None:
        self.db = db_connection
        logger.info("[SQL Agent] Initialized")

    def _ensure_db(self) -> None:
        if not self.db:
            self.db = get_database()
            logger.info("[SQL Agent] Created new database connection")

    # ------------------------------------------------------------------
    # Query builders — each returns (sql_string, params_dict)
    # ------------------------------------------------------------------

    def _build_task_query(self, entities: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """SELECT * FROM report_agile_dashboard WHERE issue_key = :key."""
        issue_key = entities.get("issue_key", "UNKNOWN")
        sql = (
            "SELECT * FROM report_agile_dashboard "
            "WHERE issue_key = :issue_key "
            f"LIMIT {SQL_MAX_ROWS}"
        )
        return sql, {"issue_key": issue_key}

    def _build_tasks_filter_query(self, entities: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build a filtered SELECT on report_agile_dashboard."""
        conditions: list[str] = []
        params: dict[str, Any] = {}

        entity_to_column = {
            "team_name": "feature_teams",
            "sprint_name": "sprint_name",
            "issue_type": "issue_type",
            "status": "issue_status_act",
            "assignee": "assignee_name",
            "cluster": "cluster",
        }

        for entity_key, column in entity_to_column.items():
            value = entities.get(entity_key)
            if value:
                # Use ILIKE for fuzzy text matching
                conditions.append(f"{column} ILIKE :p_{entity_key}")
                params[f"p_{entity_key}"] = f"%{value}%"

        where = " AND ".join(conditions) if conditions else "TRUE"
        sql = (
            f"SELECT * FROM report_agile_dashboard WHERE {where} "
            f"ORDER BY create_time DESC LIMIT {SQL_MAX_ROWS}"
        )
        return sql, params

    def _build_metric_query(self, entities: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build a SELECT on report_agile_dashboard_metrics."""
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if entities.get("team_name"):
            conditions.append("feature_teams ILIKE :p_team")
            params["p_team"] = f"%{entities['team_name']}%"

        if entities.get("sprint_name"):
            conditions.append("sprint_name ILIKE :p_sprint")
            params["p_sprint"] = f"%{entities['sprint_name']}%"

        if entities.get("cluster"):
            conditions.append("cluster_name ILIKE :p_cluster")
            params["p_cluster"] = f"%{entities['cluster']}%"

        where = " AND ".join(conditions) if conditions else "TRUE"

        # Select specific metric column if requested, otherwise all
        metric_name = entities.get("metric_name")
        allowed_metrics = {
            "initial_commitment_sp",
            "added_work_sp",
            "final_commitment_sp",
            "undone_sp",
            "complete_sp",
            "dev_potential_sp",
            "scope_drop",
            "done_total",
            "sprint_goal",
            "complete_initial_sp",
            "complete_count_sg",
            "count_sg",
            "cancel_rate",
            "done_total_issues",
            "scope_drop_issues",
        }
        if metric_name and metric_name in allowed_metrics:
            select = f"feature_teams, sprint_name, sprint_state, {metric_name}"
        else:
            select = "*"

        sql = (
            f"SELECT {select} FROM report_agile_dashboard_metrics "
            f"WHERE {where} "
            f"ORDER BY activation_date DESC LIMIT {SQL_MAX_ROWS}"
        )
        return sql, params

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build SQL from intent+entities, execute, return results.

        Args:
            state: Workflow state with 'intent', 'entities', 'original_query'.

        Returns:
            State update with sql_query, sql_result, error.
        """
        intent = state.get("intent", "task")
        entities = state.get("entities", {})
        original_query = state.get("original_query", "")

        logger.info(f"[SQL Agent] Processing intent={intent}, entities={entities}")

        self._ensure_db()

        # Build query
        try:
            if intent == "task":
                sql, params = self._build_task_query(entities)
            elif intent == "tasks_filter":
                sql, params = self._build_tasks_filter_query(entities)
            elif intent == "metric":
                sql, params = self._build_metric_query(entities)
            else:
                return {
                    "original_query": original_query,
                    "sql_query": None,
                    "sql_result": None,
                    "error": f"Unknown intent: {intent}",
                }
        except Exception as e:
            logger.error(f"[SQL Agent] Error building query: {e}")
            return {
                "original_query": original_query,
                "sql_query": None,
                "sql_result": None,
                "error": f"Error building query: {e}",
            }

        logger.info(f"[SQL Agent] SQL: {sql} | params: {params}")

        # Execute query
        try:
            results = self.db.execute_query(sql, params)
        except SQLAlchemyError as e:
            logger.error(f"[SQL Agent] Database error: {e}")
            return {
                "original_query": original_query,
                "sql_query": sql,
                "sql_result": None,
                "error": f"Database error: {e!s}",
            }

        if not results:
            logger.warning("[SQL Agent] Query returned no results")
            return {
                "original_query": original_query,
                "sql_query": sql,
                "sql_result": [],
                "error": None,
            }

        logger.info(f"[SQL Agent] Returned {len(results)} row(s)")
        return {
            "original_query": original_query,
            "sql_query": sql,
            "sql_result": results,
            "error": None,
        }
