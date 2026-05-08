"""Unit tests for the small pure helpers in ``agile_assistant.memory``.

Three modules here, all stateless and dependency-free:
  * ``token_estimator.estimate_tokens`` — char-based upper-bound
  * ``truncator.truncate_message``      — token-budget-aware cut
  * ``formatter.format_history``        — XML rendering of ConversationContext

Coverage on these is cheap and the bugs they would harbour are nasty
(prompt-budget overflow → vLLM rejects the request, formatting drift →
agent prompts silently degrade), so we cover edge cases broadly.
"""

from __future__ import annotations

from typing import Any

import pytest

from agile_assistant.memory.formatter import format_history
from agile_assistant.memory.token_estimator import CHARS_PER_TOKEN, estimate_tokens
from agile_assistant.memory.truncator import ELLIPSIS, truncate_message

# ===================================================================== #
# TokenEstimator
# ===================================================================== #


@pytest.mark.unit
class TestEstimateTokens:
    @pytest.mark.parametrize("value", ["", None])
    def test_falsy_input_returns_zero(self, value: str | None) -> None:
        # The estimator is called on optional fields (e.g. summary) — None
        # and "" must return 0, never raise.
        assert estimate_tokens(value or "") == 0

    @pytest.mark.parametrize(
        ("text_len", "expected"),
        [
            (1, 1),  # min-1 floor (max(1, 0))
            (2, 1),
            (3, 1),  # 3 // 3 == 1
            (9, 3),
            (30, 10),
            (300, 100),
            (1000, 333),
        ],
    )
    def test_length_to_token_count(self, text_len: int, expected: int) -> None:
        assert estimate_tokens("a" * text_len) == expected

    def test_chars_per_token_constant_is_three(self) -> None:
        # Pinning the constant — bump it and the prompt-budget arithmetic
        # in supervisor/workflow_task must be re-checked.
        assert CHARS_PER_TOKEN == 3

    def test_unicode_text_counted_by_codepoints_not_bytes(self) -> None:
        # Cyrillic is 2 bytes/char in UTF-8 but the estimator uses
        # `len(text)` which is codepoints — must stay codepoint-based
        # so Russian and English get the same heuristic.
        assert estimate_tokens("ABC") == estimate_tokens("АБВ")


# ===================================================================== #
# Truncator
# ===================================================================== #


@pytest.mark.unit
class TestTruncateMessage:
    def test_max_tokens_zero_returns_empty(self) -> None:
        # Defensive guard — calling with a depleted budget must not
        # produce a spurious ellipsis-only string.
        assert truncate_message("anything", 0) == ""

    def test_negative_budget_treated_as_zero(self) -> None:
        assert truncate_message("anything", -5) == ""

    def test_empty_input_passes_through(self) -> None:
        assert truncate_message("", 100) == ""

    def test_text_within_budget_unchanged(self) -> None:
        text = "Короткий ответ"
        # Budget 100 tokens * 3 chars/token = 300 char budget; "Короткий"
        # is well under that.
        assert truncate_message(text, 100) == text

    def test_oversized_text_is_cut_with_ellipsis(self) -> None:
        text = "слово " * 100  # 600 chars, budget 30 tokens = 90 chars
        result = truncate_message(text, 30)
        assert result.endswith(ELLIPSIS)
        # Total length stays inside the budget.
        assert len(result) <= 30 * CHARS_PER_TOKEN

    def test_cut_snaps_to_last_word_boundary(self) -> None:
        # Budget 9 tokens * 3 chars = 27 chars; less ELLIPSIS (6) leaves
        # 21 chars for the head. The cut must land on whitespace so the
        # final word in the head is a complete word from the original.
        text = "первое второе третье четвёртое пятое шестое"
        result = truncate_message(text, 9)
        assert result.endswith(ELLIPSIS)
        assert len(result) <= 9 * CHARS_PER_TOKEN

        head = result[: -len(ELLIPSIS)].rstrip()
        original_words = set(text.split())
        # Every space-separated token in the head must be a complete word
        # from the source — proves no mid-word cut happened.
        assert head, "head must not be empty for this budget"
        for word in head.split():
            assert word in original_words

    def test_budget_too_small_for_ellipsis_returns_marker_only(self) -> None:
        # 1 token * 3 chars = 3 char budget; ELLIPSIS itself is 6 chars.
        # cut_at goes negative → function returns ellipsis stripped.
        result = truncate_message("очень длинный текст", 1)
        assert result == ELLIPSIS.strip()


# ===================================================================== #
# Formatter
# ===================================================================== #


def _ctx(
    *,
    summary: str = "",
    recent: list[dict[str, Any]] | None = None,
    history_token_count: int = 0,
    needs_summarization: bool = False,
) -> dict[str, Any]:
    """Concise ConversationContext factory for these tests."""
    return {
        "summary": summary,
        "recent_turns": recent or [],
        "history_token_count": history_token_count,
        "needs_summarization": needs_summarization,
    }


@pytest.mark.unit
class TestFormatHistory:
    def test_none_context_returns_empty_string(self) -> None:
        assert format_history(None) == ""

    def test_empty_recent_turns_returns_empty_string(self) -> None:
        # Even with a populated summary, no turns means no block — the
        # downstream prompt slot stays clean.
        assert format_history(_ctx(summary="prev")) == ""

    def test_only_recent_turns_renders_block_without_summary(self) -> None:
        ctx = _ctx(
            recent=[
                {"role": "user", "content": "что у cthulhu"},
                {"role": "assistant", "content": "у cthulhu 5 задач"},
            ]
        )
        out = format_history(ctx)
        assert out.startswith("<conversation_history>")
        assert out.endswith("</conversation_history>")
        assert "<summary>" not in out
        assert "<recent>" in out
        assert "User: что у cthulhu" in out
        assert "Assistant: у cthulhu 5 задач" in out

    def test_summary_and_recent_render_both_blocks(self) -> None:
        ctx = _ctx(
            summary="прошлая сессия про метрики",
            recent=[{"role": "user", "content": "ещё"}],
        )
        out = format_history(ctx)
        assert "<summary>прошлая сессия про метрики</summary>" in out
        assert "<recent>" in out
        # Summary precedes the recent block.
        assert out.index("<summary>") < out.index("<recent>")

    def test_unknown_role_passes_through(self) -> None:
        # Defensive: an unexpected role label (e.g. "system") must not
        # crash — the raw role is used as the prefix.
        ctx = _ctx(recent=[{"role": "system", "content": "hint"}])
        out = format_history(ctx)
        assert "system: hint" in out

    def test_missing_role_or_content_does_not_crash(self) -> None:
        # `.get()` defaults: a turn with neither key still produces a line
        # rather than raising KeyError.
        ctx = _ctx(recent=[{}])
        out = format_history(ctx)
        # The line is empty-ish (": ") but the wrapper tags are intact.
        assert "<recent>" in out and "</recent>" in out

    def test_role_labels_are_capitalized(self) -> None:
        # The user-visible label is "User" / "Assistant" (capitalised),
        # not the raw lowercase role string from the DB.
        ctx = _ctx(recent=[{"role": "user", "content": "x"}])
        assert "User: x" in format_history(ctx)
        assert "user: x" not in format_history(ctx)

    def test_turn_order_preserved(self) -> None:
        # Order matters for the LLM — chronological history reads as a
        # dialogue, reversed history reads as nonsense.
        ctx = _ctx(
            recent=[
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        )
        out = format_history(ctx)
        positions = [out.index(s) for s in ("Q1", "A1", "Q2", "A2")]
        assert positions == sorted(positions)
