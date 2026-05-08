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
from typing import Any

from langgraph.graph import END, StateGraph

from agile_assistant.agents.guardrails import (
    BLOCKED_RESPONSE,
    OFF_TOPIC_RESPONSE,
    ResponseGuard,
    TopicGuard,
)
from agile_assistant.agents.rag_agent import RAGAgent
from agile_assistant.agents.response_agent import ResponseAgent
from agile_assistant.agents.sql_agent import SQLAgent
from agile_assistant.agents.supervisor import SupervisorAgent
from agile_assistant.agents.validator_agent import ValidatorAgent
from agile_assistant.config import settings
from agile_assistant.database.connection import DatabaseConnection
from agile_assistant.graph.state import WorkflowState
from agile_assistant.llm.client import LLMClient
from agile_assistant.models.memory import ConversationContext
from agile_assistant.tracing import make_langgraph_callback

logger = logging.getLogger(__name__)


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
        """Initialize agents, guardrails, and compile the LangGraph graph.

        Args:
            llm_client: Shared LLM client passed to every agent that
                needs generation.
            db_connection: Optional database connection for the SQL
                agent and supervisor; when ``None`` the SQL path is
                still wired but executes against an empty engine.
            retriever: Optional vector retriever; when ``None`` the RAG
                agent is disabled and hybrid queries degrade to SQL-only.
        """
        logger.info("[Workflow] Initializing AgileWorkflow...")

        self.db = db_connection

        db_engine = self.db.engine if self.db is not None else None
        self.supervisor = SupervisorAgent(llm_client, db_engine=db_engine)
        self.sql_agent = SQLAgent(db_connection=self.db)
        self.rag_agent = self._build_rag_agent(llm_client, retriever)
        self.validator = ValidatorAgent()
        self.response_agent = ResponseAgent(llm_client)
        self.topic_guard = self._build_topic_guard()
        self.response_guard = ResponseGuard()

        self.graph = self._build_graph()
        logger.info("[Workflow] Workflow graph built successfully")

    @staticmethod
    def _build_rag_agent(
        llm_client: LLMClient,
        retriever: Any | None,
    ) -> RAGAgent | None:
        """Return a ``RAGAgent`` bound to ``retriever``, or ``None``."""
        if retriever is None:
            return None
        return RAGAgent(llm_client, retriever)

    @staticmethod
    def _build_topic_guard() -> TopicGuard | None:
        """Build the regex-only ``TopicGuard``, or ``None`` when disabled.

        Returns:
            ``TopicGuard()`` when ``settings.guardrail_enabled`` is true,
            otherwise ``None`` so the input-guardrail node short-circuits.
        """
        if not settings.guardrail_enabled:
            logger.info("[Workflow] TopicGuard disabled via settings")
            return None
        return TopicGuard()

    # ------------------------------------------------------------------
    # Node wrappers
    # ------------------------------------------------------------------

    def _input_guardrail_node(self, state: WorkflowState) -> dict[str, Any]:
        """Level 1 Input Guardrail: prompt-injection filter + whitelist.

        Off-topic classification is handled by Supervisor — this node only
        hard-blocks prompt-injection attempts and logs whitelist fast-paths.
        """
        logger.info("[Workflow] Entering Input Guardrail node")
        if self.topic_guard is None:
            return {"blocked": False, "guard_result": {"passed": True, "reason": "disabled"}}

        result = self.topic_guard.check(state["original_query"])
        guard_payload = {"passed": result.passed, "reason": result.reason}
        if not result.passed:
            return {
                "blocked": True,
                "guard_result": guard_payload,
                "final_response": OFF_TOPIC_RESPONSE,
            }
        return {"blocked": False, "guard_result": guard_payload}

    def _off_topic_node(self, state: WorkflowState) -> dict[str, Any]:
        """Fixed response for off-topic queries classified by Supervisor."""
        logger.info(
            "[Workflow] Supervisor classified as off_topic: %r",
            state.get("original_query", "")[:100],
        )
        return {"final_response": OFF_TOPIC_RESPONSE}

    def _supervisor_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run the Supervisor agent to classify the query and extract entities."""
        logger.info("[Workflow] Entering Supervisor node")
        return self.supervisor.process(
            state["original_query"],
            conversation_context=state.get("conversation_context"),
            user_profile=state.get("user_profile"),
        )

    def _sql_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run the SQL agent to translate the query and execute it."""
        logger.info("[Workflow] Entering SQL Agent node")
        return self.sql_agent.process(state)

    def _rag_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run the RAG agent, or return an empty result when disabled."""
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
        """Run the Validator agent over the agent outputs in ``state``."""
        logger.info("[Workflow] Entering Validator node")
        return self.validator.process(state)

    def _response_agent_node(self, state: WorkflowState) -> dict[str, Any]:
        """Run the Response agent to compose the final user-facing reply."""
        logger.info("[Workflow] Entering Response Agent node")
        return self.response_agent.process(state)

    def _output_guardrail_node(self, state: WorkflowState) -> dict[str, Any]:
        """Level 3 Output Guardrail: sanitize / block the final response."""
        logger.info("[Workflow] Entering Output Guardrail node")
        response = state.get("final_response", "")
        if not response:
            return {}
        # Skip guard for off-topic placeholder — it's already safe content.
        if response == OFF_TOPIC_RESPONSE:
            return {}

        result = self.response_guard.check(
            response=response,
            query_type=state.get("query_type", ""),
            context_urls=state.get("rag_sources", []),
        )

        if result.blocked:
            failed = [c.name for c in result.checks if not c.passed]
            logger.warning("[Workflow] OutputGuard BLOCKED: %s", failed)
            return {"final_response": BLOCKED_RESPONSE}

        if not result.passed:
            failed = [c.name for c in result.checks if not c.passed]
            logger.info("[Workflow] OutputGuard SANITIZED: %s", failed)
            return {"final_response": result.sanitized_response}

        return {}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_guardrail(self, state: WorkflowState) -> str:
        """If guardrail blocked the query — skip to END; otherwise Supervisor."""
        if state.get("blocked"):
            logger.info(
                "[Workflow] Input blocked by guardrail: %s",
                state.get("guard_result", {}).get("reason"),
            )
            return "end"
        return "supervisor"

    def _route_after_supervisor(self, state: WorkflowState) -> str:
        """Route based on Supervisor's query_type."""
        query_type = state.get("query_type", "sql")
        logger.info("[Workflow] Routing decision: query_type=%s", query_type)

        if query_type == "off_topic":
            return "off_topic"
        if query_type in ("simple", "error"):
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
        """Wire the LangGraph nodes and edges and return the compiled graph."""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("input_guardrail", self._input_guardrail_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("off_topic", self._off_topic_node)
        workflow.add_node("sql_agent", self._sql_agent_node)
        workflow.add_node("rag_agent", self._rag_agent_node)
        workflow.add_node("sql_and_rag", self._sql_and_rag_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("response_agent", self._response_agent_node)
        workflow.add_node("output_guardrail", self._output_guardrail_node)

        workflow.set_entry_point("input_guardrail")

        # Level 1 guardrail: block off-topic before Supervisor
        workflow.add_conditional_edges(
            "input_guardrail",
            self._route_after_guardrail,
            {"end": END, "supervisor": "supervisor"},
        )

        # Conditional routing after Supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "sql_agent": "sql_agent",
                "rag_agent": "rag_agent",
                "sql_and_rag": "sql_and_rag",
                "response_agent": "response_agent",
                "off_topic": "off_topic",
            },
        )

        # Off-topic skips Response Agent entirely — fixed text, straight to END
        workflow.add_edge("off_topic", END)

        # sql / rag / hybrid all go through Validator -> Response Agent
        workflow.add_edge("sql_agent", "validator")
        workflow.add_edge("rag_agent", "validator")
        workflow.add_edge("sql_and_rag", "validator")
        workflow.add_edge("validator", "response_agent")

        # Response Agent → Output Guardrail → END
        workflow.add_edge("response_agent", "output_guardrail")
        workflow.add_edge("output_guardrail", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0913 — keyword-only DTO-style entrypoint
        self,
        user_query: str,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        conversation_context: ConversationContext | None = None,
        user_profile: dict[str, Any] | None = None,
        trace_user_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        trace_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the workflow with a user query.

        The memory-layer kwargs are optional. Callers that don't use the
        memory layer (tests, direct invocations, pre-memory deployments)
        get the original stateless behaviour.

        Tracing kwargs (``trace_*``) are also optional. When the
        Langfuse SDK is initialised, this method attaches a callback
        handler to ``graph.invoke`` so every node and LLM call lands
        inside a single trace tree — replacing the previous imperative
        ``langfuse_client.trace(...)`` pattern, which did not propagate
        across LangGraph's Pregel runtime.

        Args:
            user_query: The user's natural language query.
            conversation_id: Short-term memory conversation id (if any).
            user_id: Long-term memory user id (if any).
            conversation_context: Pre-assembled history to inject.
            user_profile: Pre-loaded preferences to inject.
            trace_user_id: External user identifier surfaced in Langfuse
                (cookie/SSO id from the API layer). ``user_id`` above is
                an *internal* memory uuid and would not be useful in the
                tracing UI.
            trace_metadata: Free-form key/value bag attached to the
                trace (task_id, celery_retry, etc.). Avoid PII.
            trace_tags: Optional tag list for filtering in Langfuse UI.

        Returns:
            Final state after processing through all agents. When
            tracing is active the dict additionally carries
            ``_langfuse_trace_id`` so downstream consumers (e.g. the
            judge task) can attach scores to the same trace.
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
            "blocked": False,
            "guard_result": {},
            "conversation_id": conversation_id,
            "user_id": user_id,
            "conversation_context": conversation_context,
            "user_profile": user_profile,
        }

        # Build the Langfuse callback per-invocation so user_id /
        # session_id / metadata land on the right trace. The handler is
        # stateful (one trace per call), so reusing a single instance
        # across requests would interleave events.
        callback = make_langgraph_callback(
            user_id=trace_user_id,
            session_id=conversation_id,
            metadata=trace_metadata,
            tags=trace_tags,
        )
        config: dict[str, Any] = {"callbacks": [callback]} if callback else {}

        try:
            result = self.graph.invoke(initial_state, config=config or None)
            logger.info("[Workflow] Workflow completed successfully")
            # Surface the trace id to the caller so downstream async
            # work (judge scoring) can attach to the same trace.
            # ``trace_id`` becomes available on the handler after the
            # first node runs; reading it here is safe and idempotent.
            if callback is not None:
                trace_id = getattr(callback, "get_trace_id", lambda: "")()
                if trace_id:
                    result["_langfuse_trace_id"] = trace_id
            return result
        except Exception as e:
            logger.error("[Workflow] Error during workflow execution: %s", e)
            raise
