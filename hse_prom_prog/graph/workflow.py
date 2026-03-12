"""LangGraph workflow definition for multi-agent processing.

Graph topology:

  Supervisor ──► (conditional routing by query_type)
    ├─ sql     ──► SQL Agent ────────────────────► Validator ──► Response Agent
    ├─ rag     ──► RAG Agent ────────────────────► Validator ──► Response Agent
    ├─ hybrid  ──► SQL Agent ──┐
    │              RAG Agent ──┴─► Validator ──► Response Agent
    └─ simple  ──► Response Agent (direct)
"""

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hse_prom_prog.agents.rag_agent import RAGAgent
from hse_prom_prog.agents.response_agent import ResponseAgent
from hse_prom_prog.agents.sql_agent import SQLAgent
from hse_prom_prog.agents.supervisor import SupervisorAgent
from hse_prom_prog.agents.validator_agent import ValidatorAgent
from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow."""

    messages: Annotated[list, add_messages]
    original_query: str
    intent: str
    entities: dict[str, Any]
    query_type: str
    route: str
    sql_query: str
    sql_result: list[dict[str, Any]]
    rag_response: str
    rag_sources: list[str]
    error: str
    validation_result: dict[str, Any]
    final_response: str


class AgileWorkflow:
    """LangGraph workflow for processing Jira queries through multiple agents.

    Attributes:
        supervisor: Supervisor agent instance.
        sql_agent: SQL agent instance.
        rag_agent: RAG agent instance (None if retriever unavailable).
        validator: Validator agent instance.
        response_agent: Response agent instance.
        db: Database connection instance.
        graph: Compiled LangGraph state graph.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db_connection: DatabaseConnection | None = None,
        retriever: Any | None = None,
    ) -> None:
        logger.info("[Workflow] Initializing AgileWorkflow...")

        self.db = db_connection

        self.supervisor = SupervisorAgent(llm_client)
        self.sql_agent = SQLAgent(db_connection=self.db)
        self.rag_agent = RAGAgent(llm_client, retriever) if retriever else None
        self.validator = ValidatorAgent()
        self.response_agent = ResponseAgent(llm_client)

        self.graph = self._build_graph()
        logger.info("[Workflow] Workflow graph built successfully")

    # ------------------------------------------------------------------
    # Node wrappers
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering Supervisor node")
        return self.supervisor.process(state["original_query"])

    def _sql_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering SQL Agent node")
        return self.sql_agent.process(state)

    def _rag_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering RAG Agent node")
        if self.rag_agent is None:
            logger.warning("[Workflow] RAG Agent not available (no retriever)")
            return {"rag_response": None, "rag_sources": []}
        return self.rag_agent.process(state)

    def _sql_and_rag_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run SQL Agent and RAG Agent sequentially for hybrid queries."""
        logger.info("[Workflow] Entering SQL+RAG node (hybrid)")
        sql_result = self.sql_agent.process(state)

        merged = {**state, **sql_result}
        rag_result = (
            self.rag_agent.process(merged)
            if self.rag_agent
            else {"rag_response": None, "rag_sources": []}
        )

        return {**sql_result, **rag_result}

    def _validator_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering Validator node")
        return self.validator.process(state)

    def _response_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        logger.info("[Workflow] Entering Response Agent node")
        return self.response_agent.process(state)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_supervisor(self, state: WorkflowState) -> str:
        """Route based on Supervisor's query_type."""
        query_type = state.get("query_type", "sql")
        logger.info("[Workflow] Routing decision: query_type=%s", query_type)

        if query_type == "simple":
            return "response_agent"
        if query_type == "rag":
            return "rag_agent"
        if query_type == "hybrid":
            return "sql_and_rag"
        # Default: sql
        return "sql_agent"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(WorkflowState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("sql_agent", self._sql_agent_node)
        workflow.add_node("rag_agent", self._rag_agent_node)
        workflow.add_node("sql_and_rag", self._sql_and_rag_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("response_agent", self._response_agent_node)

        workflow.set_entry_point("supervisor")

        # Conditional routing after Supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "sql_agent": "sql_agent",
                "rag_agent": "rag_agent",
                "sql_and_rag": "sql_and_rag",
                "response_agent": "response_agent",
            },
        )

        # sql / rag / hybrid all go through Validator -> Response Agent
        workflow.add_edge("sql_agent", "validator")
        workflow.add_edge("rag_agent", "validator")
        workflow.add_edge("sql_and_rag", "validator")
        workflow.add_edge("validator", "response_agent")

        # simple goes directly to Response Agent
        workflow.add_edge("response_agent", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_query: str) -> dict[str, Any]:
        """Execute the workflow with a user query.

        Args:
            user_query: The user's natural language query.

        Returns:
            Final state after processing through all agents.
        """
        logger.info("[Workflow] Starting workflow with query: %s", user_query)

        initial_state: WorkflowState = {
            "messages": [],
            "original_query": user_query,
            "intent": "",
            "entities": {},
            "query_type": "",
            "route": "",
            "sql_query": "",
            "sql_result": [],
            "rag_response": "",
            "rag_sources": [],
            "error": "",
            "validation_result": {},
            "final_response": "",
        }

        try:
            result = self.graph.invoke(initial_state)
            logger.info("[Workflow] Workflow completed successfully")
            return result
        except Exception as e:
            logger.error("[Workflow] Error during workflow execution: %s", e)
            raise
