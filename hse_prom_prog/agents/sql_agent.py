"""SQL Agent: LangGraph-based text-to-SQL with tool calling.

Architecture (based on https://docs.langchain.com/oss/python/langgraph/sql-agent):
  START → model (LLM with tools) ↔ tools → extract → END

The LLM uses tool calls to:
1. run_query — execute SQL (with retry on error, max 3 attempts)

Schema is loaded from DB at runtime and injected into the system prompt.

Uses Qwen3-8B-AWQ served by vLLM with --enable-auto-tool-choice.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from hse_prom_prog.agents.schema_loader import get_schema_compact
from hse_prom_prog.agents.sql_tools import SQL_TOOLS, set_db
from hse_prom_prog.config import settings
from hse_prom_prog.database.connection import DatabaseConnection, get_database

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

_SYSTEM_PROMPT_TEMPLATE = """\
You are a PostgreSQL SQL expert. Call run_query immediately.

## Schema
{schema}

## CRITICAL RULES (violations = wrong answer)
1. NEVER add WHERE sprint_state = 'closed'. Include ALL sprints.
2. ALWAYS use ILIKE '%%value%%' for team names. \
NEVER use = for teams. NEVER filter by cluster_name.
3. GROUP BY sprint_name, never by jirasprint_id.
4. For tasks: SELECT * FROM report_agile_dashboard WHERE ...
5. "количество задач" = COUNT(*) of ALL tasks, not only Done.
6. MIN/MAX: SELECT feature_teams, sprint_name, <col> \
FROM ... ORDER BY <col> DESC/ASC LIMIT 1. No GROUP BY.
7. No WHERE <metric> > 0 or IS NOT NULL — changes AVG.
8. COUNT tasks → report_agile_dashboard (1 row per task). \
NEVER use metrics (1 row per team×sprint) for task counts.
9. "story points" (storypoints_act) exist for ALL issue_type. \
NEVER add WHERE issue_type = 'Story' for total SP.

## Metric queries (3 types — choose carefully)
A) No word "среднее/средний/avg" → rows per sprint, NO AVG:
   SELECT feature_teams, sprint_name, <metric> \
FROM report_agile_dashboard_metrics \
WHERE feature_teams ILIKE '%%team%%'
   Examples: "X команды Y", "Какой X у команды Y"

B) Word "среднее/средний/средняя/avg" → use AVG():
   SELECT feature_teams, AVG(<metric>) \
FROM report_agile_dashboard_metrics \
WHERE feature_teams ILIKE '%%team%%' \
GROUP BY feature_teams

C) Compare teams ("у какой команды", "топ") → AVG per team:
   SELECT feature_teams, AVG(<metric>) \
FROM report_agile_dashboard_metrics \
GROUP BY feature_teams ORDER BY ...

## Table guide
- Tasks (issue_key, status, assignee, type, bugs) \
→ report_agile_dashboard
- Metrics (velocity=complete_sp, done_total, scope_drop, \
sprint_goal, cancel_rate) → report_agile_dashboard_metrics
- Story points sum → SUM(storypoints_act) \
FROM report_agile_dashboard

## Examples
Q: scope drop команды cthulhu
SQL: SELECT feature_teams, sprint_name, scope_drop \
FROM report_agile_dashboard_metrics \
WHERE feature_teams ILIKE '%%cthulhu%%'

Q: Среднее done total команды marketplace
SQL: SELECT feature_teams, AVG(done_total) \
FROM report_agile_dashboard_metrics \
WHERE feature_teams ILIKE '%%marketplace%%' \
GROUP BY feature_teams

Q: Все баги
SQL: SELECT * FROM report_agile_dashboard \
WHERE issue_type = 'Bug'"""


# ── State ────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Internal state for the LangGraph SQL agent."""

    messages: Annotated[list, add_messages]
    retry_count: int
    final_sql: str
    final_result: list[dict[str, Any]]
    final_error: str


# ── Graph node functions (module-level) ──────────────────────

_llm_free: Any = None
_llm_forced: Any = None


def _init_llms() -> tuple[Any, Any]:
    """Initialize both LLM variants (lazy, once).

    Returns (llm_forced, llm_free):
      - llm_forced: tool_choice="any" — guarantees a tool call
      - llm_free: no constraint — model can respond with text
    """
    global _llm_free, _llm_forced  # noqa: PLW0603
    if _llm_free is None:
        llm = ChatOpenAI(
            base_url=settings.sql_vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.sql_vllm_model,
            temperature=0.0,
            max_tokens=512,
        )
        _llm_free = llm.bind_tools(SQL_TOOLS)
        _llm_forced = llm.bind_tools(SQL_TOOLS, tool_choice="any")
    return _llm_forced, _llm_free


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", flags=re.DOTALL)


def _strip_think(messages: list) -> list:
    """Remove <think>…</think> blocks from prior AI messages to save tokens."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            new_content = _THINK_RE.sub("", msg.content).strip()
            cleaned.append(
                AIMessage(
                    content=new_content,
                    tool_calls=msg.tool_calls,
                    id=msg.id,
                )
            )
        else:
            cleaned.append(msg)
    return cleaned


def _has_query_result(messages: list) -> bool:
    """Check if run_query has already returned a successful result."""
    for msg in messages:
        if (
            isinstance(msg, ToolMessage)
            and msg.name == "run_query"
            and not msg.content.startswith("SQL Error:")
            and not msg.content.startswith("ERROR:")
        ):
            return True
    return False


def _call_model(state: AgentState) -> dict:
    """Call the LLM with current messages.

    Uses tool_choice="any" (forced) until the first successful
    run_query result, then switches to free mode so the model
    can finish with a text response.
    """
    llm_forced, llm_free = _init_llms()
    messages = _strip_think(state["messages"])

    llm = llm_free if _has_query_result(messages) else llm_forced

    response = llm.invoke(messages)
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


def _get_last_sqls(messages: list) -> list[str]:
    """Extract all run_query SQL calls from message history."""
    sqls: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "run_query":
                    sqls.append(tc["args"].get("query", ""))
    return sqls


def _check_retry(state: AgentState) -> dict:
    """Increment retry counter; add hint if model repeats same SQL."""
    last = state["messages"][-1]
    retry = state.get("retry_count", 0)
    extra_messages: list = []

    if isinstance(last, ToolMessage) and last.name == "run_query":
        content = last.content
        if content.startswith("SQL Error:") or content.startswith("ERROR:"):
            retry += 1
            if retry >= _MAX_RETRIES:
                logger.warning("[SQL Agent] Max retries reached (%d)", retry)
            else:
                sqls = _get_last_sqls(state["messages"])
                min_for_dup = 2
                if len(sqls) >= min_for_dup and sqls[-1] == sqls[-2]:
                    hint = (
                        "Your SQL is identical to the previous attempt. "
                        "Read the error carefully and change the query. "
                        "Common fixes: add missing columns to GROUP BY, "
                        "remove GROUP BY, or use a different column."
                    )
                    extra_messages.append(
                        HumanMessage(content=hint),
                    )
                    logger.info("[SQL Agent] Added retry hint")

    result: dict[str, Any] = {"retry_count": retry}
    if extra_messages:
        result["messages"] = extra_messages
    return result


def _after_tools(state: AgentState) -> str:
    """After tool execution: if max retries → extract, else → model."""
    retry = state.get("retry_count", 0)
    if retry >= _MAX_RETRIES:
        return "extract"
    return "model"


def _extract_results(state: AgentState) -> dict:
    """Extract SQL from conversation and re-execute for full results.

    The tool returns only a sample to the LLM (to stay under context limit),
    so we re-run the last successful SQL here to get complete results.
    """
    last_sql = ""
    last_ok_sql = ""
    error = ""

    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "run_query":
                    last_sql = tc["args"].get("query", "")
        if isinstance(msg, ToolMessage) and msg.name == "run_query":
            content = msg.content
            if content.startswith("SQL Error:") or content.startswith("ERROR:"):
                error = content
            else:
                last_ok_sql = last_sql
                error = ""

    sql = last_ok_sql or last_sql

    # Re-execute the last successful SQL to get full results
    result: list[dict[str, Any]] = []
    if last_ok_sql:
        from hse_prom_prog.agents.sql_tools import _get_db  # noqa: PLC0415

        try:
            db = _get_db()
            result = db.execute_query(sql)
        except Exception as e:
            logger.warning("[SQL Agent] Re-execute failed: %s", e)
            error = f"Re-execute error: {e}"

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

        schema_ddl = get_schema_compact(self.db.engine)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema=schema_ddl)

        messages = [
            SystemMessage(content=system_prompt),
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
    """Parse the JSON output of run_query into list[dict]."""
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        payload = json.loads(content)
        if isinstance(payload, dict) and "rows" in payload:
            return payload["rows"]
    return []
