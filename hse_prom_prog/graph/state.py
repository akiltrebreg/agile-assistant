"""LangGraph state schema shared across the workflow and agents.

Extracted from ``workflow.py`` so that nodes, repositories, and the Celery
task layer can depend on the state shape without pulling in the full
``AgileWorkflow`` object (and its heavy LLM / DB initialisation).
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from hse_prom_prog.models.memory import ConversationContext


class WorkflowState(TypedDict, total=False):
    """State schema for the LangGraph workflow.

    Marked ``total=False`` so memory-layer fields (``conversation_id``,
    ``user_id``, ``conversation_context``, ``user_profile``) stay optional
    — existing callers that don't populate them continue to work unchanged.
    The core workflow fields are still always populated by ``run()``.
    """

    # --- Core workflow fields (always populated by AgileWorkflow.run) ---
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
    blocked: bool
    guard_result: dict[str, Any]

    # --- Memory-layer fields (all optional; None when memory is disabled) ---
    conversation_id: str | None
    user_id: str | None
    conversation_context: ConversationContext | None
    user_profile: dict[str, Any] | None
