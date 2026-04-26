"""Level 1: Input Guardrail — regex-only pre-filter before Supervisor.

Responsibilities (intentionally minimal):
  * Block prompt-injection / role-hijacking attempts before they reach any
    LLM — pure security concern, handled by regex.
  * Fast-path whitelist for trivially-safe queries (issue keys, greetings,
    meta-questions like "что ты умеешь") so they skip unnecessary work.

Topic / off-topic classification is handled **by Supervisor** (it already
runs an LLM call and can distinguish «вопрос про Agile» from «рецепт
борща» via its prompt). Duplicating that in a separate embedding /
NLI model proved fragile (narrow gap on Russian cosine similarity,
miscalibrated NLI probabilities), so it was removed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from hse_prom_prog.metrics import GUARDRAIL_L1_RESULTS

logger = logging.getLogger(__name__)

# Map internal reason strings to a small, stable Prometheus label set —
# wider cardinality would clutter the dashboard with one-off variants.
_L1_REASON_LABELS = {
    "blocked:prompt_injection": "injection_blocked",
    "whitelist": "whitelist_fast_path",
    "pass": "pass",
}


_ALWAYS_PASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[A-Z]{2,10}-\d+", re.IGNORECASE),
    re.compile(r"^\s*(привет|здравствуй|добрый|hi|hello)", re.IGNORECASE),
    re.compile(r"(что (ты )?(умеешь|можешь)|\bhelp\b|помо[гщ])", re.IGNORECASE),
]

_HARD_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(ignore|forget|disregard).{0,30}(instruction|prompt|system|role)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(ты теперь|act as|you are now|pretend).{0,30}"
        r"(не ассистент|not an assistant|другая роль|different role)",
        re.IGNORECASE,
    ),
]


@dataclass
class GuardResult:
    """Result of an input guardrail check.

    Outcomes:
      * `blocked:prompt_injection` → hard-deny regex fired
      * `whitelist`                → known-safe pattern (issue key, greeting)
      * `pass`                     → regex nothing to say, Supervisor decides
    """

    passed: bool
    reason: str = ""


class TopicGuard:
    """Regex-only input guardrail.

    No LLM, no embeddings, no thresholds — this layer exists only for:
      1. Security (prompt injection block)
      2. Fast-path (trivial whitelist)

    Everything else (including off-topic detection) is delegated to the
    Supervisor agent, which is an LLM call anyway.
    """

    def check(self, query: str) -> GuardResult:
        """Classify the query via two regex stages.

        Order:
          1. Hard-deny → block with reason='blocked:prompt_injection'
          2. Whitelist → pass with reason='whitelist'
          3. Default   → pass with reason='pass' (Supervisor will classify)
        """
        for pattern in _HARD_DENY_PATTERNS:
            if pattern.search(query):
                logger.warning("[TopicGuard] HARD DENY: %r", query[:100])
                GUARDRAIL_L1_RESULTS.labels(
                    reason=_L1_REASON_LABELS["blocked:prompt_injection"]
                ).inc()
                return GuardResult(passed=False, reason="blocked:prompt_injection")

        for pattern in _ALWAYS_PASS_PATTERNS:
            if pattern.search(query):
                GUARDRAIL_L1_RESULTS.labels(reason=_L1_REASON_LABELS["whitelist"]).inc()
                return GuardResult(passed=True, reason="whitelist")

        GUARDRAIL_L1_RESULTS.labels(reason=_L1_REASON_LABELS["pass"]).inc()
        return GuardResult(passed=True, reason="pass")


OFF_TOPIC_RESPONSE = (
    "К сожалению, этот вопрос выходит за рамки моих компетенций. "
    "Я — ассистент для анализа Jira-задач и Agile-метрик.\n\n"
    "Я могу помочь с:\n"
    "• Agile-практиками (Sprint Goal, Velocity, Backlog Grooming)\n"
    "• Метриками команд (Done Total, Scope Drop, Cancel Rate)\n"
    "• Деталях о задачах Jira (текущий статус, описание)\n\n"
    "Попробуйте задать вопрос по одной из этих тем!"
)
