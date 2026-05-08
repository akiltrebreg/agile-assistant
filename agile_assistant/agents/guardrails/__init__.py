"""Guardrails for input / SQL / output layers.

Level 1 (TopicGuard): regex-only pre-filter (prompt injection + whitelist);
    off-topic classification is delegated to Supervisor.
Level 2 (SQLGuard): three-layer defense for run_query (regex + AST + limits).
Level 3 (ResponseGuard): rule-based post-processing of the final response.
"""

from agile_assistant.agents.guardrails.response_guard import (
    BLOCKED_RESPONSE,
    OutputCheckResult,
    OutputGuardResult,
    ResponseGuard,
)
from agile_assistant.agents.guardrails.sql_guard import SQLGuardResult, check_sql
from agile_assistant.agents.guardrails.topic_guard import (
    OFF_TOPIC_RESPONSE,
    GuardResult,
    TopicGuard,
)

__all__ = [
    "BLOCKED_RESPONSE",
    "OFF_TOPIC_RESPONSE",
    "GuardResult",
    "OutputCheckResult",
    "OutputGuardResult",
    "ResponseGuard",
    "SQLGuardResult",
    "TopicGuard",
    "check_sql",
]
