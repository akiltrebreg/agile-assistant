"""Guardrails for input / SQL / output layers.

Level 1 (TopicGuard): off-topic filter via embedding cosine similarity.
"""

from hse_prom_prog.agents.guardrails.topic_guard import (
    OFF_TOPIC_RESPONSE,
    GuardResult,
    TopicGuard,
)

__all__ = ["OFF_TOPIC_RESPONSE", "GuardResult", "TopicGuard"]
