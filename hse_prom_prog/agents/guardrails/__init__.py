"""Guardrails for input / SQL / output layers.

Level 1 (TopicGuard): off-topic filter via embedding cosine similarity.
Level 2 (SQLGuard): three-layer defense for run_query (regex + AST + limits).
Level 3 (ResponseGuard): rule-based post-processing of the final response.
"""

from hse_prom_prog.agents.guardrails.response_guard import (
    BLOCKED_RESPONSE,
    OutputCheckResult,
    OutputGuardResult,
    ResponseGuard,
)
from hse_prom_prog.agents.guardrails.sql_guard import SQLGuardResult, check_sql
from hse_prom_prog.agents.guardrails.topic_guard import (
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
