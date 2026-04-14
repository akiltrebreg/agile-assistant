"""SQL Agent: LangGraph-based text-to-SQL with tool calling.

Architecture (based on https://docs.langchain.com/oss/python/langgraph/sql-agent):
  START → model (LLM with tools) ↔ tools → extract → END

The LLM uses tool calls to:
1. list_tables — see available tables
2. get_schema — get DDL for selected tables
3. run_query — execute SQL (with retry on error, max 3 attempts)

Uses Qwen3-8B-AWQ served by vLLM with --enable-auto-tool-choice.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from hse_prom_prog.agents.sql_tools import SQL_TOOLS, set_db
from hse_prom_prog.config import settings
from hse_prom_prog.database.connection import DatabaseConnection, get_database

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_MIN_LINES_FOR_TABLE = 2

_SYSTEM_PROMPT = """\
You are a PostgreSQL SQL expert. You have access to tools to explore the database \
and run queries.

Workflow:
1. Call list_tables to see available tables
2. Call get_schema with the relevant table names to see their columns
3. Write a SELECT query and call run_query to execute it
4. If you get an error, fix the query and retry (up to 3 times)

Rules:
- Only use SELECT queries
- Use ILIKE '%value%' for text filters (PostgreSQL)
- Always check the schema before writing a query
- If the question asks for a specific team, sprint, or entity, filter by it
- Return the final SQL and results to the user"""


# ── State ────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Internal state for the LangGraph SQL agent."""

    messages: Annotated[list, add_messages]
    retry_count: int
    final_sql: str
    final_result: list[dict[str, Any]]
    final_error: str


# ── Graph node functions (module-level) ──────────────────────

_llm_with_tools: Any = None


def _init_llm() -> Any:
    """Initialize LLM with tools (lazy, once)."""
    global _llm_with_tools  # noqa: PLW0603
    if _llm_with_tools is None:
        llm = ChatOpenAI(
            base_url=settings.sql_vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.sql_vllm_model,
            temperature=0.0,
            max_tokens=512,
        )
        _llm_with_tools = llm.bind_tools(SQL_TOOLS)
    return _llm_with_tools


def _call_model(state: AgentState) -> dict:
    """Call the LLM with current messages."""
    llm = _init_llm()
    response = llm.invoke(state["messages"])
    tool_calls = getattr(response, "tool_calls", None) or []
    content_preview = str(response.content)[:300] if response.content else ""
    logger.info(
        "[SQL Agent] LLM response: tool_calls=%d, content=%r",
        len(tool_calls),
        content_preview,
    )
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    """Route: if last message has tool_calls → tools, else → extract."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "extract"


def _check_retry(state: AgentState) -> dict:
    """Increment retry counter after tool execution if there was an error."""
    last = state["messages"][-1]
    retry = state.get("retry_count", 0)
    if isinstance(last, ToolMessage) and last.name == "run_query":
        content = last.content
        if content.startswith("SQL Error:") or content.startswith("ERROR:"):
            retry += 1
            if retry >= _MAX_RETRIES:
                logger.warning("[SQL Agent] Max retries reached (%d)", retry)
    return {"retry_count": retry}


def _after_tools(state: AgentState) -> str:
    """After tool execution: if max retries → extract, else → model."""
    retry = state.get("retry_count", 0)
    if retry >= _MAX_RETRIES:
        return "extract"
    return "model"


def _extract_results(state: AgentState) -> dict:
    """Extract SQL and results from the conversation history."""
    sql = ""
    result: list[dict[str, Any]] = []
    error = ""

    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "run_query":
                    sql = tc["args"].get("query", "")
        if isinstance(msg, ToolMessage) and msg.name == "run_query":
            content = msg.content
            if content.startswith("SQL Error:") or content.startswith("ERROR:"):
                error = content
                result = []
            else:
                error = ""
                result = _parse_tool_result(content)

    return {"final_sql": sql, "final_result": result, "final_error": error}


# ── Graph builder ────────────────────────────────────────────


def _build_graph() -> Any:
    """Build the LangGraph state graph."""
    tool_node = ToolNode(SQL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("model", _call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("check_retry", _check_retry)
    graph.add_node("extract", _extract_results)

    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model",
        _should_continue,
        {
            "tools": "tools",
            "extract": "extract",
        },
    )
    graph.add_edge("tools", "check_retry")
    graph.add_conditional_edges(
        "check_retry",
        _after_tools,
        {
            "model": "model",
            "extract": "extract",
        },
    )
    graph.add_edge("extract", END)

    return graph.compile()


# ── Agent class ──────────────────────────────────────────────


class SQLAgent:
    """LangGraph SQL Agent with tool calling.

    Drop-in replacement: same process() signature as the old agent.
    """

    def __init__(self, db_connection: DatabaseConnection | None = None) -> None:
        self.db = db_connection
        self._graph = _build_graph()
        logger.info(
            "[SQL Agent] Initialized LangGraph agent (model=%s, url=%s)",
            settings.sql_vllm_model,
            settings.sql_vllm_base_url,
        )

    def _ensure_db(self) -> None:
        if not self.db:
            self.db = get_database()

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process a user query through the LangGraph SQL agent.

        Args:
            state: Workflow state with 'original_query'.

        Returns:
            dict with sql_query, sql_result, error, original_query.
        """
        original_query = state.get("original_query", "")
        logger.info("[SQL Agent] Processing: %s", original_query[:80])

        self._ensure_db()
        set_db(self.db)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=original_query),
        ]

        try:
            result = self._graph.invoke(
                {
                    "messages": messages,
                    "retry_count": 0,
                    "final_sql": "",
                    "final_result": [],
                    "final_error": "",
                }
            )
        except Exception as e:
            logger.error("[SQL Agent] Graph execution failed: %s", e)
            return {
                "original_query": original_query,
                "sql_query": None,
                "sql_result": None,
                "error": f"Agent error: {e}",
            }

        sql = result.get("final_sql", "")
        sql_result = result.get("final_result")
        error = result.get("final_error", "")

        if sql:
            logger.info("[SQL Agent] Final SQL: %s", sql[:200])
        if error:
            logger.warning("[SQL Agent] Error: %s", error[:200])
        logger.info(
            "[SQL Agent] Returned %d row(s)",
            len(sql_result) if sql_result else 0,
        )

        return {
            "original_query": original_query,
            "sql_query": sql or None,
            "sql_result": sql_result or None,
            "error": error or None,
        }


# ── Utilities ────────────────────────────────────────────────


def _parse_tool_result(content: str) -> list[dict[str, Any]]:
    """Parse the pipe-delimited text output of run_query into list[dict]."""
    lines = content.strip().split("\n")
    if len(lines) < _MIN_LINES_FOR_TABLE:
        return []

    data_lines = [line for line in lines if not line.startswith("...")]
    if len(data_lines) < _MIN_LINES_FOR_TABLE:
        return []

    headers = [h.strip() for h in data_lines[0].split("|")]
    rows: list[dict[str, Any]] = []
    for line in data_lines[1:]:
        values = [v.strip() for v in line.split("|")]
        if len(values) == len(headers):
            row = dict(zip(headers, values, strict=False))
            _convert_numeric_values(row)
            rows.append(row)
    return rows


def _convert_numeric_values(row: dict[str, Any]) -> None:
    """Convert string values to int/float where possible, in-place."""
    for k, v in row.items():
        if not isinstance(v, str):
            continue
        with contextlib.suppress(ValueError, TypeError):
            row[k] = float(v) if "." in v else int(v)
