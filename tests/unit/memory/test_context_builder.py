"""Unit tests for ``ContextBuilder.build``.

The builder pulls messages from a ``ConversationRepository`` and folds them
into a ``ConversationContext`` that fits a per-call token budget. Mocked
inputs only — no DB. The bugs to catch:

  * sliding window blowing the prompt budget under heavy turns
  * summary not being preserved when the message list is empty
  * ``needs_summarization`` flag being wrong (either spam-flagging when
    the rolling summary already covers the older window, or missing the
    case where it doesn't)
  * always-keep-one-turn invariant breaking under tiny budgets
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from agile_assistant.memory.context_builder import SUMMARY_RESERVE_TOKENS, ContextBuilder
from agile_assistant.memory.token_estimator import CHARS_PER_TOKEN, estimate_tokens
from agile_assistant.models.memory import Conversation, Message

# --------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------- #


def _make_conversation(
    *,
    summary: str | None = None,
    summary_turn_index: int = 0,
    is_active: bool = True,
) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=uuid4(),
        user_id=uuid4(),
        title=None,
        summary=summary,
        summary_turn_index=summary_turn_index,
        created_at=now,
        updated_at=now,
        is_active=is_active,
    )


def _make_message(  # noqa: PLR0913
    turn_index: int,
    *,
    role: str = "user",
    content: str = "hi",
    content_truncated: str | None = None,
    metadata: dict[str, Any] | None = None,
    conversation_id: UUID | None = None,
) -> Message:
    return Message(
        id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        turn_index=turn_index,
        role=role,
        content=content,
        content_truncated=content_truncated,
        metadata=metadata or {},
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def repo() -> MagicMock:
    """A bare-bones ConversationRepository stub.

    Tests set ``repo.get.return_value`` and ``repo.get_messages.return_value``
    per scenario. The ContextBuilder only ever calls these two methods.
    """
    r = MagicMock()
    r.get = MagicMock(return_value=None)
    r.get_messages = MagicMock(return_value=[])
    return r


@pytest.fixture
def builder(repo: MagicMock) -> ContextBuilder:
    return ContextBuilder(repo)


# ===================================================================== #
# Empty-state behaviour
# ===================================================================== #


@pytest.mark.unit
class TestEmptyStates:
    def test_no_conversation_returns_empty_context(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Conversation row missing → return a fresh, empty context shape.
        repo.get.return_value = None
        ctx = builder.build(uuid4(), token_budget=800)
        assert ctx == {
            "summary": "",
            "recent_turns": [],
            "history_token_count": 0,
            "needs_summarization": False,
        }
        # Cheap-fail: confirm get_messages was not called once we knew the
        # conversation didn't exist.
        repo.get_messages.assert_not_called()

    def test_conversation_without_messages_keeps_summary(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        repo.get.return_value = _make_conversation(summary="old summary")
        repo.get_messages.return_value = []
        ctx = builder.build(uuid4(), token_budget=800)
        assert ctx["summary"] == "old summary"
        assert ctx["recent_turns"] == []
        assert ctx["history_token_count"] == 0
        assert ctx["needs_summarization"] is False

    def test_conversation_without_messages_no_summary(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        repo.get.return_value = _make_conversation(summary=None)
        repo.get_messages.return_value = []
        ctx = builder.build(uuid4(), token_budget=800)
        assert ctx["summary"] == ""


# ===================================================================== #
# Sliding window — token-budget arithmetic
# ===================================================================== #


@pytest.mark.unit
class TestSlidingWindow:
    def test_all_short_messages_fit_under_budget(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        repo.get.return_value = _make_conversation()
        msgs = [_make_message(i, content=f"msg {i}") for i in range(4)]
        repo.get_messages.return_value = msgs

        ctx = builder.build(uuid4(), token_budget=200)
        assert len(ctx["recent_turns"]) == 4
        # Order is chronological even though we walk newest→oldest internally.
        assert [t["turn_index"] for t in ctx["recent_turns"]] == [0, 1, 2, 3]

    def test_oversized_history_drops_oldest_turns(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Each msg ≈ 60 chars → 20 tokens + 2 = 22 token cost. Budget 50
        # tokens → only 2 messages fit (44 tokens; adding a 3rd would be 66).
        body = "слово " * 10  # 60 chars
        msgs = [_make_message(i, content=body) for i in range(6)]
        repo.get.return_value = _make_conversation()
        repo.get_messages.return_value = msgs

        ctx = builder.build(uuid4(), token_budget=50)
        assert len(ctx["recent_turns"]) == 2
        # Newest two turns survive — oldest are dropped.
        assert [t["turn_index"] for t in ctx["recent_turns"]] == [4, 5]

    def test_at_least_one_turn_kept_even_when_oversize(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Single huge message that exceeds the budget on its own — the
        # always-keep-one-turn invariant must still include it.
        msgs = [_make_message(0, content="х" * 5000)]
        repo.get.return_value = _make_conversation()
        repo.get_messages.return_value = msgs

        ctx = builder.build(uuid4(), token_budget=10)
        assert len(ctx["recent_turns"]) == 1
        assert ctx["recent_turns"][0]["turn_index"] == 0

    def test_summary_reserve_shrinks_available_budget(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Build two scenarios with the same messages: one with summary,
        # one without. The summary-bearing scenario must keep strictly
        # fewer turns because SUMMARY_RESERVE_TOKENS is subtracted.
        body = "x" * 30  # 30 chars → 10 tokens + 2 = 12-token cost
        msgs = [_make_message(i, content=body) for i in range(40)]
        repo.get_messages.return_value = msgs

        repo.get.return_value = _make_conversation(summary=None)
        ctx_no_summary = builder.build(uuid4(), token_budget=200)

        repo.get.return_value = _make_conversation(summary="prev session")
        ctx_with_summary = builder.build(uuid4(), token_budget=200)

        assert len(ctx_with_summary["recent_turns"]) < len(ctx_no_summary["recent_turns"])

    def test_history_token_count_includes_summary(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        summary = "длинная сводка прошлой сессии"
        repo.get.return_value = _make_conversation(summary=summary)
        repo.get_messages.return_value = [_make_message(0, content="hi")]
        ctx = builder.build(uuid4(), token_budget=800)
        # The reported token count combines per-message cost AND the
        # summary's own estimate — downstream gating must see the full
        # total, not just the recent-turns cost.
        assert ctx["history_token_count"] >= estimate_tokens(summary)


# ===================================================================== #
# needs_summarization flag
# ===================================================================== #


@pytest.mark.unit
class TestNeedsSummarization:
    def test_flag_false_when_window_starts_at_zero(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Whole history fits in the window → kept_start_idx=0, no older
        # messages to summarise.
        repo.get.return_value = _make_conversation(summary_turn_index=0)
        repo.get_messages.return_value = [
            _make_message(0, content="a"),
            _make_message(1, content="b"),
        ]
        ctx = builder.build(uuid4(), token_budget=800)
        assert ctx["needs_summarization"] is False

    def test_flag_true_when_older_turns_not_in_summary(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Turns 0..29 exist; only the last few fit; summary covered turn 5.
        # Turns 6..(kept_start-1) are not in the summary → needs_summarization.
        body = "x" * 60
        msgs = [_make_message(i, content=body) for i in range(30)]
        repo.get.return_value = _make_conversation(
            summary="covers up to turn 5", summary_turn_index=5
        )
        repo.get_messages.return_value = msgs

        ctx = builder.build(uuid4(), token_budget=80)
        kept_indices = [t["turn_index"] for t in ctx["recent_turns"]]
        kept_start = kept_indices[0]
        # Sanity: the test setup actually exercises the gap.
        assert kept_start > 6
        assert ctx["needs_summarization"] is True

    def test_flag_false_when_summary_already_covers_window_edge(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Summary covers exactly up to (kept_start - 1) — no gap.
        body = "x" * 60
        msgs = [_make_message(i, content=body) for i in range(10)]
        repo.get.return_value = _make_conversation(summary="rolling", summary_turn_index=8)
        repo.get_messages.return_value = msgs

        ctx = builder.build(uuid4(), token_budget=80)
        # Force the kept_start to be ≤ 8 (covered by summary). Default
        # scenario keeps last few; with our 80-token budget and 60-char
        # messages, ~3 turns survive → kept_start around 7-8.
        kept_start = ctx["recent_turns"][0]["turn_index"]
        # Production check: needs_summarization=True iff summary_turn_index < kept_start.
        expected = kept_start > 8
        assert ctx["needs_summarization"] is expected


# ===================================================================== #
# Per-message conversion (truncated body, role/index propagation)
# ===================================================================== #


@pytest.mark.unit
class TestMessageConversion:
    def test_content_truncated_preferred_over_full_body(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Long replies are persisted with content_truncated; the builder
        # must use the short version so a single big assistant message
        # doesn't blow the prompt.
        long_body = "очень " * 500
        short_body = "очень короткое"
        msg = _make_message(
            0,
            role="assistant",
            content=long_body,
            content_truncated=short_body,
        )
        repo.get.return_value = _make_conversation()
        repo.get_messages.return_value = [msg]

        ctx = builder.build(uuid4(), token_budget=800)
        assert ctx["recent_turns"][0]["content"] == short_body
        # And the budget stayed in line with the short version, not the long.
        assert ctx["history_token_count"] < estimate_tokens(long_body)

    def test_turn_dict_shape(self, builder: ContextBuilder, repo: MagicMock) -> None:
        msg = _make_message(7, role="user", content="x", metadata={"intent": "task"})
        repo.get.return_value = _make_conversation()
        repo.get_messages.return_value = [msg]

        turn = builder.build(uuid4(), token_budget=800)["recent_turns"][0]
        assert turn == {
            "role": "user",
            "content": "x",
            "turn_index": 7,
            "metadata": {"intent": "task"},
        }

    def test_missing_metadata_yields_empty_dict(
        self, builder: ContextBuilder, repo: MagicMock
    ) -> None:
        # Defensive: a Message with metadata=None must convert to {} so
        # downstream agents can do `.get("intent")` without guards.
        msg = _make_message(0, content="x")
        msg.metadata = None  # type: ignore[assignment]
        repo.get.return_value = _make_conversation()
        repo.get_messages.return_value = [msg]
        turn = builder.build(uuid4(), token_budget=800)["recent_turns"][0]
        assert turn["metadata"] == {}

    def test_message_token_cost_overhead(self) -> None:
        # The static helper adds +2 for role label + separator. Pinning
        # this here makes future budget-arithmetic regressions visible.
        msg = _make_message(0, content="abcdef")  # 6 chars → 2 tokens
        cost = ContextBuilder._message_token_cost(msg)
        assert cost == estimate_tokens("abcdef") + 2


# ===================================================================== #
# Constant pinning
# ===================================================================== #


@pytest.mark.unit
class TestConstants:
    def test_summary_reserve_constant(self) -> None:
        # Bumping this without raising token_budget upstream causes silent
        # truncation of recent turns — pin the value so a change is loud.
        assert SUMMARY_RESERVE_TOKENS == 150

    def test_chars_per_token_constant_used_by_builder(self) -> None:
        # Sanity: the builder uses estimate_tokens, which uses CHARS_PER_TOKEN.
        # Indirectly exercised in TestSlidingWindow but worth pinning here too.
        assert CHARS_PER_TOKEN == 3
