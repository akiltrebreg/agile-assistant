"""Level 3: Output Guardrail — rule-based post-processing финального ответа.

Sanitizes and validates the response produced by ``ResponseAgent`` before it
reaches the user. Runs in <1 ms — no LLM calls. LLM-as-judge is reserved for
offline eval (``eval/metrics.py``).

Two modes per check:
  * **BLOCK** — critical violation (empty response, traceback) → whole response
    is replaced with ``BLOCKED_RESPONSE``
  * **SANITIZE** — suspicious fragments are removed in place (SQL leak, URL
    hallucination, internal-detail leak)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import ClassVar

from hse_prom_prog.metrics import GUARDRAIL_L3_CHECKS_FAILED, GUARDRAIL_L3_RESULTS

logger = logging.getLogger(__name__)


@dataclass
class OutputCheckResult:
    """Result of a single output check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class OutputGuardResult:
    """Aggregated result of all output checks."""

    passed: bool
    checks: list[OutputCheckResult] = field(default_factory=list)
    sanitized_response: str = ""
    blocked: bool = False


class ResponseGuard:
    """Rule-based guardrail for the final response.

    Args:
        max_response_length: Hard upper bound on response length.
        min_response_length: Anything shorter is treated as empty.
        min_cyrillic_ratio: Min fraction of Cyrillic letters to consider the
            response Russian (lowered from 0.3 used in eval because user
            messages may legitimately quote English terms like "In Progress").
    """

    # Hallucinated URLs / emails
    _URL_PATTERN = re.compile(r"https?://[^\s\]\)\"'<>]{10,}", re.IGNORECASE)
    _EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    # Raw SQL leak: SQL-keyword + one of our whitelist / system tables.
    # Blocks `SELECT ... FROM report_agile_dashboard WHERE ...`
    # but NOT natural-language "необходимо select подходящий вариант from ...".
    _SQL_LEAK_PATTERN = re.compile(
        r"\b(?:SELECT|INSERT|UPDATE|DELETE|WHERE)\b"
        r"[^\n]{0,200}?\b"
        r"(report_agile_dashboard(?:_metrics)?|pg_\w+)\b",
        re.IGNORECASE,
    )

    # Python traceback fragments
    _TRACEBACK_PATTERN = re.compile(
        r"(Traceback \(most recent|"
        r"File \".+\", line \d+|"
        r"raise \w+Error|"
        r"Exception:|"
        r"Error:.*\n\s+at )",
        re.IGNORECASE,
    )

    # Internal system details we never want to leak.
    # Technology names (qdrant/redis/vllm/celery) are FINE as words — model may
    # legitimately explain "how RAG works". Block only connection-string context
    # (URI schemes, host:port, env vars) which is always a real leak.
    _INTERNAL_LEAK_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # Our table names — always a leak
        re.compile(r"\breport_agile_dashboard\w*\b", re.IGNORECASE),
        # PostgreSQL system tables
        re.compile(r"\bpg_\w+\b"),
        # Prompt / instructions leak
        re.compile(r"\b(system prompt|system message|instructions?:)\b", re.IGNORECASE),
        # LLM params leak
        re.compile(r"temperature\s*=\s*[\d.]+"),
        # Connection strings: URI schemes (qdrant://, redis://, postgres://)
        re.compile(r"\b(qdrant|redis|postgresql?|vllm)s?://", re.IGNORECASE),
        # host:port patterns (qdrant:6333, redis:6379)
        re.compile(r"\b(qdrant|redis|postgres|vllm)[-\w]*:\d{2,5}\b", re.IGNORECASE),
        # Environment variables (VLLM_BASE_URL, QDRANT_URL, REDIS_HOST, etc.)
        re.compile(r"\b(QDRANT_\w+|REDIS_\w+|POSTGRES_\w+|VLLM_\w+|VSELLM_\w+|CELERY_\w+)\b"),
    ]

    # Whitelist URL prefixes (extend if RAG ever surfaces http-links)
    _ALLOWED_URL_PREFIXES: ClassVar[list[str]] = []

    # Checks that trigger full replacement instead of sanitization
    _CRITICAL_CHECKS: ClassVar[frozenset[str]] = frozenset({"traceback", "length_empty"})

    def __init__(
        self,
        max_response_length: int = 5000,
        min_response_length: int = 10,
        min_cyrillic_ratio: float = 0.25,
    ) -> None:
        self.max_response_length = max_response_length
        self.min_response_length = min_response_length
        self.min_cyrillic_ratio = min_cyrillic_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        response: str,
        query_type: str = "",
        context_urls: list[str] | None = None,
    ) -> OutputGuardResult:
        """Run all output checks and produce a single aggregated result."""
        checks: list[OutputCheckResult] = []
        sanitized = response

        checks.append(self._check_length(response))
        checks.append(self._check_language(response))

        sql_check, sanitized = self._check_sql_leak(sanitized)
        checks.append(sql_check)

        trace_check, sanitized = self._check_traceback(sanitized)
        checks.append(trace_check)

        url_check, sanitized = self._check_hallucinated_urls(sanitized, context_urls or [])
        checks.append(url_check)

        internal_check, sanitized = self._check_internal_leak(sanitized)
        checks.append(internal_check)

        failed = [c for c in checks if not c.passed]
        blocked = any(c.name in self._CRITICAL_CHECKS for c in failed)

        passed = not failed
        GUARDRAIL_L3_RESULTS.labels(
            passed=str(passed).lower(),
            blocked=str(blocked).lower(),
        ).inc()
        for failed_check in failed:
            GUARDRAIL_L3_CHECKS_FAILED.labels(check_name=failed_check.name).inc()

        return OutputGuardResult(
            passed=passed,
            checks=checks,
            sanitized_response=sanitized,
            blocked=blocked,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_length(self, response: str) -> OutputCheckResult:
        length = len(response.strip())
        if length < self.min_response_length:
            return OutputCheckResult("length_empty", False, f"len={length}")
        if length > self.max_response_length:
            return OutputCheckResult("length_overflow", False, f"len={length}")
        return OutputCheckResult("length", True, f"len={length}")

    def _check_language(self, response: str) -> OutputCheckResult:
        letters = [c for c in response if c.isalpha()]
        if not letters:
            return OutputCheckResult("language", True, "no_letters")
        cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
        ratio = cyrillic / len(letters)
        if ratio < self.min_cyrillic_ratio:
            return OutputCheckResult("language", False, f"cyrillic_ratio={ratio:.2f}")
        return OutputCheckResult("language", True, f"cyrillic_ratio={ratio:.2f}")

    def _check_sql_leak(self, response: str) -> tuple[OutputCheckResult, str]:
        matches = self._SQL_LEAK_PATTERN.findall(response)
        if matches:
            sanitized = self._SQL_LEAK_PATTERN.sub("[SQL запрос скрыт]", response)
            return (
                OutputCheckResult("sql_leak", False, f"found={len(matches)}"),
                sanitized,
            )
        return OutputCheckResult("sql_leak", True), response

    def _check_traceback(self, response: str) -> tuple[OutputCheckResult, str]:
        if self._TRACEBACK_PATTERN.search(response):
            sanitized = re.sub(
                r"Traceback.*?(?=\n[^\s]|\Z)",
                "[техническая информация скрыта]",
                response,
                flags=re.DOTALL,
            )
            return OutputCheckResult("traceback", False, "found"), sanitized
        return OutputCheckResult("traceback", True), response

    def _check_hallucinated_urls(
        self, response: str, context_urls: list[str]
    ) -> tuple[OutputCheckResult, str]:
        found_urls = self._URL_PATTERN.findall(response)
        found_emails = self._EMAIL_PATTERN.findall(response)

        hallucinated: list[str] = []
        sanitized = response

        for url in found_urls:
            is_allowed = any(url.startswith(prefix) for prefix in self._ALLOWED_URL_PREFIXES)
            if not is_allowed and url not in context_urls:
                hallucinated.append(url)
                sanitized = sanitized.replace(url, "[ссылка удалена]")

        for email in found_emails:
            hallucinated.append(email)
            sanitized = sanitized.replace(email, "[email скрыт]")

        if hallucinated:
            return (
                OutputCheckResult("hallucinated_urls", False, f"removed={len(hallucinated)}"),
                sanitized,
            )
        return OutputCheckResult("hallucinated_urls", True), response

    def _check_internal_leak(self, response: str) -> tuple[OutputCheckResult, str]:
        found: list[str] = []
        sanitized = response
        for pattern in self._INTERNAL_LEAK_PATTERNS:
            matches = pattern.findall(sanitized)
            if matches:
                found.extend(matches)
                sanitized = pattern.sub("[внутренняя информация]", sanitized)

        if found:
            return (
                OutputCheckResult("internal_leak", False, f"found={found[:3]}"),
                sanitized,
            )
        return OutputCheckResult("internal_leak", True), response


BLOCKED_RESPONSE = (
    "Извините, не удалось сформировать корректный ответ. "
    "Пожалуйста, попробуйте переформулировать запрос или обратитесь позже."
)
