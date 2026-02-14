"""LangGraph workflow definition for multi-agent processing.

This module defines the state graph that coordinates the three agents:
Supervisor -> (conditional) -> SQL Agent -> Response Agent
                            -> Response Agent (direct)
"""

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hse_prom_prog.agents.response_agent import ResponseAgent
from hse_prom_prog.agents.sql_agent import SQLAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent
from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow.

    Attributes:
        messages: List of conversation messages (for LangGraph compatibility).
        original_query: The original user query.
        intent: Classified intent (task, tasks_filter, metric, general).
        entities: Extracted entities dict from Supervisor.
        route: Routing decision (db_query or direct_response).
        sql_query: Generated SQL query string (for debugging).
        sql_result: Query results as list of dicts.
        error: Error message if any step failed.
        final_response: Formatted final response.
    """

    messages: Annotated[list, add_messages]
    original_query: str
    intent: str
    entities: dict[str, Any]
    route: str
    sql_query: str
    sql_result: list[dict[str, Any]]
    error: str
    final_response: str


class AgileWorkflow:
    """LangGraph workflow for processing Jira queries through multiple agents.

    This class builds and manages a state graph that processes user queries
    through Supervisor, SQL Agent, and Response Agent.

    Attributes:
        supervisor: Supervisor agent instance.
        sql_agent: SQL agent instance.
        response_agent: Response agent instance.
        db: Database connection instance.
        graph: Compiled LangGraph state graph.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db_connection: DatabaseConnection | None = None,
    ) -> None:
        logger.info("[Workflow] Initializing AgileWorkflow...")

        self.db = db_connection

        self.supervisor = SupervisorAgent(llm_client)
        self.sql_agent = SQLAgent(db_connection=self.db)
        self.response_agent = ResponseAgent(llm_client)

        self.graph = self._build_graph()
        logger.info("[Workflow] Workflow graph built successfully")

    def _supervisor_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering Supervisor node")
        return self.supervisor.process(state["original_query"])

    def _sql_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering SQL Agent node")
        return self.sql_agent.process(state)

    def _response_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering Response Agent node")
        return self.response_agent.process(state)

    def _route_after_supervisor(self, state: WorkflowState) -> str:
        """Route based on Supervisor's decision."""
        route = state.get("route", "db_query")
        logger.info(f"[Workflow] Routing decision: {route}")
        return "sql_agent" if route == "db_query" else "response_agent"

    def _build_graph(self) -> Any:
        workflow = StateGraph(WorkflowState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("sql_agent", self._sql_agent_node)
        workflow.add_node("response_agent", self._response_agent_node)

        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {"sql_agent": "sql_agent", "response_agent": "response_agent"},
        )
        workflow.add_edge("sql_agent", "response_agent")
        workflow.add_edge("response_agent", END)

        return workflow.compile()

    def run(self, user_query: str) -> dict[str, Any]:
        """Execute the workflow with a user query.

        Args:
            user_query: The user's natural language query.

        Returns:
            Final state after processing through all agents.
        """
        logger.info(f"[Workflow] Starting workflow with query: {user_query}")

        initial_state: WorkflowState = {
            "messages": [],
            "original_query": user_query,
            "intent": "",
            "entities": {},
            "route": "",
            "sql_query": "",
            "sql_result": [],
            "error": "",
            "final_response": "",
        }

        try:
            result = self.graph.invoke(initial_state)
            logger.info("[Workflow] Workflow completed successfully")
            return result
        except Exception as e:
            logger.error(f"[Workflow] Error during workflow execution: {e}")
            raise
