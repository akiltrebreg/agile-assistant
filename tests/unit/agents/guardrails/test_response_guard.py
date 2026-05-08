"""Unit tests for L3 output guardrail (``ResponseGuard``).

Six checks, two outcome modes:

* **BLOCK** (whole response replaced by ``BLOCKED_RESPONSE`` upstream)
  → ``traceback``, ``length_empty``
* **SANITIZE** (response kept, fragments redacted in place)
  → ``sql_leak``, ``hallucinated_urls``, ``internal_leak``,
    ``length_overflow``, ``language``

The ``check()`` method always runs every check (so a single response
with multiple problems gets all redactions applied at once) and the
overall ``blocked`` flag is True only when at least one CRITICAL check
fails. These tests assert (1) outcome, (2) which check fired, and
(3) the post-sanitization text.
"""

from __future__ import annotations

import pytest

from agile_assistant.agents.guardrails.response_guard import (
    BLOCKED_RESPONSE,
    OutputGuardResult,
    ResponseGuard,
)


@pytest.fixture
def guard() -> ResponseGuard:
    return ResponseGuard()


def _check_named(result: OutputGuardResult, name: str):
    """Pull a single named check out of the aggregated result for assertion."""
    matches = [c for c in result.checks if c.name == name]
    assert len(matches) == 1, f"expected one check named {name!r}, got {len(matches)}"
    return matches[0]


# ===================================================================== #
# Length check
# ===================================================================== #


@pytest.mark.unit
class TestLength:
    @pytest.mark.parametrize("response", ["", "   ", "abc", "коротко"])
    def test_short_response_is_critical(self, guard: ResponseGuard, response: str) -> None:
        result = guard.check(response)
        assert _check_named(result, "length_empty").passed is False
        assert result.blocked is True
        assert result.passed is False

    def test_response_in_normal_range_passes_length(self, guard: ResponseGuard) -> None:
        result = guard.check("Это нормальный ответ длиннее десяти символов.")
        assert all(c.passed for c in result.checks if c.name == "length")
        assert result.blocked is False

    def test_overflow_fails_but_does_not_block(self, guard: ResponseGuard) -> None:
        # length_overflow is intentionally NOT in _CRITICAL_CHECKS — it
        # marks the response as failing the guard, but does not trigger
        # the BLOCKED_RESPONSE replacement upstream. Pin this so a
        # well-meaning refactor doesn't accidentally promote it.
        oversized = "А" * 6000
        result = guard.check(oversized)
        assert _check_named(result, "length_overflow").passed is False
        assert result.passed is False
        assert result.blocked is False


# ===================================================================== #
# Language check
# ===================================================================== #


@pytest.mark.unit
class TestLanguage:
    def test_pure_english_response_fails(self, guard: ResponseGuard) -> None:
        result = guard.check("This is a perfectly fine English sentence please.")
        assert _check_named(result, "language").passed is False

    def test_pure_russian_response_passes(self, guard: ResponseGuard) -> None:
        result = guard.check("Это полностью русский ответ без латиницы.")
        assert _check_named(result, "language").passed is True

    def test_mixed_with_threshold_cyrillic_passes(self, guard: ResponseGuard) -> None:
        # ~30% Cyrillic letters — over the 0.25 default threshold.
        result = guard.check(
            "Velocity команды растёт стабильно — продолжайте Sprint Goal practice."
        )
        assert _check_named(result, "language").passed is True

    def test_no_letters_passes_via_short_circuit(self, guard: ResponseGuard) -> None:
        # The check has an explicit no_letters fast-path so a numeric /
        # symbol-only response doesn't divide by zero.
        result = guard.check("123456789012345 !!!!!!!")
        assert _check_named(result, "language").passed is True


# ===================================================================== #
# SQL leak check
# ===================================================================== #


@pytest.mark.unit
class TestSqlLeak:
    def test_select_with_whitelist_table_is_redacted(self, guard: ResponseGuard) -> None:
        response = (
            "По вашему запросу выполнен SELECT issue_key, summary FROM "
            "report_agile_dashboard WHERE issue_key='AL-1' и найдена задача."
        )
        result = guard.check(response)
        assert _check_named(result, "sql_leak").passed is False
        assert "[SQL запрос скрыт]" in result.sanitized_response
        # Original SQL fragment is gone from the sanitized text.
        assert "report_agile_dashboard WHERE" not in result.sanitized_response

    def test_natural_language_select_word_does_not_trigger(self, guard: ResponseGuard) -> None:
        # The pattern requires a whitelist *table* near the keyword —
        # casual mentions of "выберите подходящий вариант" must pass.
        result = guard.check("Необходимо select подходящий вариант из бэклога for sprint.")
        assert _check_named(result, "sql_leak").passed is True


# ===================================================================== #
# Traceback check
# ===================================================================== #


@pytest.mark.unit
class TestTraceback:
    def test_python_traceback_is_critical_and_redacted(self, guard: ResponseGuard) -> None:
        response = (
            "Не удалось получить данные.\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 42, in foo\n'
            '    raise ValueError("boom")\n'
            "ValueError: boom"
        )
        result = guard.check(response)
        assert _check_named(result, "traceback").passed is False
        assert result.blocked is True
        # Sanitized text replaces the traceback block with a placeholder.
        assert "[техническая информация скрыта]" in result.sanitized_response
        assert "Traceback" not in result.sanitized_response

    def test_clean_response_passes_traceback_check(self, guard: ResponseGuard) -> None:
        result = guard.check("Задача AL-1 в статусе In Progress. Назначен Иванов.")
        assert _check_named(result, "traceback").passed is True
        assert result.blocked is False


# ===================================================================== #
# Hallucinated URLs
# ===================================================================== #


@pytest.mark.unit
class TestHallucinatedUrls:
    def test_url_not_in_context_is_removed(self, guard: ResponseGuard) -> None:
        result = guard.check(
            "Подробнее на https://random-blog.example.com/article-42",
            context_urls=[],
        )
        assert _check_named(result, "hallucinated_urls").passed is False
        assert "https://random-blog" not in result.sanitized_response
        assert "[ссылка удалена]" in result.sanitized_response

    def test_url_present_in_context_is_kept(self, guard: ResponseGuard) -> None:
        url = "https://docs.example.com/agile-guide"
        result = guard.check(
            f"Подробнее в гайде {url}",
            context_urls=[url],
        )
        assert _check_named(result, "hallucinated_urls").passed is True
        assert url in result.sanitized_response

    def test_email_is_always_removed(self, guard: ResponseGuard) -> None:
        # Emails have no allow-list — any address is treated as a leak.
        result = guard.check("Свяжитесь с владельцем продукта owner@example.com")
        assert _check_named(result, "hallucinated_urls").passed is False
        assert "owner@example.com" not in result.sanitized_response
        assert "[email скрыт]" in result.sanitized_response

    def test_no_urls_passes(self, guard: ResponseGuard) -> None:
        result = guard.check("Это обычный ответ без каких-либо ссылок.")
        assert _check_named(result, "hallucinated_urls").passed is True


# ===================================================================== #
# Internal-leak check
# ===================================================================== #


@pytest.mark.unit
class TestInternalLeak:
    @pytest.mark.parametrize(
        "secret",
        [
            "qdrant://qdrant:6333",
            "redis://redis:6379/0",
            "postgresql://user:pw@postgres:5432/db",
            "qdrant:6333",
            "QDRANT_URL=http://qdrant:6333",
            "REDIS_HOST=redis",
            "POSTGRES_PASSWORD=secret",
            "VLLM_BASE_URL=http://vllm:8000",
            "report_agile_dashboard",
            "report_agile_dashboard_metrics",
            "pg_catalog",
            "system prompt",
            "temperature=0.7",
        ],
    )
    def test_known_internal_leak_is_redacted(self, guard: ResponseGuard, secret: str) -> None:
        # A response that mentions the secret in passing — the matching
        # fragment is replaced with the canonical placeholder, the rest
        # of the text survives.
        response = f"Ответ ассистента: {secret}, остальное содержание корректно."
        result = guard.check(response)
        assert _check_named(result, "internal_leak").passed is False
        assert "[внутренняя информация]" in result.sanitized_response

    def test_plain_technology_word_does_not_leak(self, guard: ResponseGuard) -> None:
        # Bare technology names without a connection-string context are OK —
        # the model may legitimately explain "RAG uses Qdrant for vectors".
        result = guard.check("RAG работает поверх векторного хранилища, в нашем случае это Qdrant.")
        assert _check_named(result, "internal_leak").passed is True


# ===================================================================== #
# Aggregation & shape
# ===================================================================== #


@pytest.mark.unit
class TestAggregation:
    def test_clean_response_passes_all_checks(self, guard: ResponseGuard) -> None:
        result = guard.check(
            "Задача AL-12345 находится в статусе In Progress. "
            "Назначена на Иванова. Срок — конец спринта."
        )
        assert result.passed is True
        assert result.blocked is False
        assert all(c.passed for c in result.checks)
        # Sanitized text equals the original when nothing fired.
        assert result.sanitized_response.startswith("Задача AL-12345")

    def test_multiple_failures_aggregate(self, guard: ResponseGuard) -> None:
        # SQL leak + internal leak + URL hallucination together.
        response = (
            "SELECT * FROM report_agile_dashboard WHERE 1=1; "
            "REDIS_HOST=redis. Подробнее: https://invented.example.com/x"
        )
        result = guard.check(response)
        failed_names = {c.name for c in result.checks if not c.passed}
        # Each non-language failure is recorded.
        assert "sql_leak" in failed_names
        assert "internal_leak" in failed_names
        assert "hallucinated_urls" in failed_names
        # No critical check fired → not blocked, but overall failed.
        assert result.passed is False
        assert result.blocked is False

    def test_critical_check_sets_blocked(self, guard: ResponseGuard) -> None:
        result = guard.check("")
        # length_empty IS critical — guarantees blocked=True.
        assert result.blocked is True

    def test_returns_outputguardresult(self, guard: ResponseGuard) -> None:
        result = guard.check("Ответ нормальной длины на русском языке.")
        assert isinstance(result, OutputGuardResult)
        assert hasattr(result, "passed")
        assert hasattr(result, "blocked")
        assert hasattr(result, "sanitized_response")
        assert isinstance(result.checks, list)

    def test_blocked_response_constant_shape(self) -> None:
        # The constant is what upstream code substitutes when blocked=True.
        assert isinstance(BLOCKED_RESPONSE, str)
        assert "Извините" in BLOCKED_RESPONSE
