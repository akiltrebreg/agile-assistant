"""Unit tests for memory-layer Celery tasks.

The Celery decorators wrap plain functions — we call those functions
directly (no broker, no real DB). The two side effects worth pinning
down:

  * ``summarize_session`` always returns ``None`` on any exception
    (memory maintenance must never break the user flow).
  * ``update_profile_async`` is the cheap rule-based per-turn refresh
    — must NOT call the LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agile_assistant.models.memory import ConversationSummary, Message
from agile_assistant.tasks import memory_tasks as mt

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _msg(
    turn: int, role: str = "user", content: str = "x", meta: dict[str, Any] | None = None
) -> Message:
    return Message(
        id=uuid4(),
        conversation_id=uuid4(),
        turn_index=turn,
        role=role,
        content=content,
        content_truncated=None,
        metadata=meta or {},
        created_at=datetime.now(UTC),
    )


def _summary(text: str, when: datetime | None = None) -> ConversationSummary:
    return ConversationSummary(
        id=uuid4(),
        conversation_id=uuid4(),
        user_id=uuid4(),
        summary=text,
        topics=[],
        turn_count=4,
        created_at=when or datetime.now(UTC),
    )


@pytest.fixture
def memory() -> MagicMock:
    mm = MagicMock()
    mm.conversation_repo.get_messages.return_value = []
    mm.summary_repo.get_recent.return_value = []
    mm.update_profile.return_value = {}
    return mm


@pytest.fixture
def patched_runtime(monkeypatch: pytest.MonkeyPatch, memory: MagicMock):
    """Patch ``get_database`` and ``get_llm_client`` + ``MemoryManager``
    constructor so the task body runs entirely against mocks."""
    db = MagicMock()
    llm = MagicMock()
    monkeypatch.setattr(mt, "get_database", lambda: db)
    monkeypatch.setattr(mt, "get_llm_client", lambda: llm)
    monkeypatch.setattr(mt, "MemoryManager", lambda _db: memory)
    return {"db": db, "llm": llm, "memory": memory}


# ===================================================================== #
# summarize_session
# ===================================================================== #


@pytest.mark.unit
class TestSummarizeSession:
    def test_skips_short_conversations(self, patched_runtime: dict[str, Any]) -> None:
        # Only one message → below _MIN_TURNS_FOR_SUMMARY → no-op + None.
        patched_runtime["memory"].conversation_repo.get_messages.return_value = [
            _msg(0, content="hi")
        ]
        out = mt.summarize_session(str(uuid4()), str(uuid4()))
        assert out is None
        patched_runtime["llm"].invoke.assert_not_called()
        patched_runtime["memory"].summary_repo.create.assert_not_called()

    def test_full_pipeline_persists_and_returns_payload(
        self, patched_runtime: dict[str, Any]
    ) -> None:
        memory = patched_runtime["memory"]
        llm = patched_runtime["llm"]
        memory.conversation_repo.get_messages.return_value = [
            _msg(
                0,
                role="user",
                content="что у cthulhu",
                meta={"entities": {"team_name": "cthulhu", "metric_name": "velocity"}},
            ),
            _msg(1, role="assistant", content="velocity растёт"),
            _msg(
                2, role="user", content="а в спринте?", meta={"entities": {"sprint_name": "Q1.1"}}
            ),
            _msg(3, role="assistant", content="ok"),
        ]
        llm.invoke.return_value = "  Summary text.  "
        memory.summary_repo.get_recent.return_value = []  # rolling-summary path no-ops

        conv_id = uuid4()
        user_id = uuid4()
        out = mt.summarize_session(str(conv_id), str(user_id))

        assert out is not None
        assert out["conversation_id"] == str(conv_id)
        assert out["summary"] == "Summary text."  # stripped
        assert out["turn_count"] == 4
        # Topics include unique team / metric / sprint values in order of first sight.
        assert out["topics"] == ["cthulhu", "velocity", "Q1.1"]
        # Persisted into conversation_summaries via the summary_repo.
        memory.summary_repo.create.assert_called_once()
        # Preferences refreshed against the same conversation.
        memory.update_profile.assert_called_with(user_id, conv_id)

    def test_exception_swallowed_returns_none(self, patched_runtime: dict[str, Any]) -> None:
        # If the LLM call fails, the task must not crash the worker —
        # return None and log. Pin so a future refactor doesn't propagate.
        memory = patched_runtime["memory"]
        memory.conversation_repo.get_messages.return_value = [_msg(i) for i in range(4)]
        patched_runtime["llm"].invoke.side_effect = TimeoutError("vllm down")
        assert mt.summarize_session(str(uuid4()), str(uuid4())) is None

    def test_db_close_called_in_finally(self, patched_runtime: dict[str, Any]) -> None:
        # Even on early-return paths (too few messages), the DB connection
        # must be closed — leaks would build up under heavy close traffic.
        patched_runtime["memory"].conversation_repo.get_messages.return_value = [_msg(0)]
        mt.summarize_session(str(uuid4()), str(uuid4()))
        patched_runtime["db"].close.assert_called_once()


# ===================================================================== #
# update_profile_async
# ===================================================================== #


@pytest.mark.unit
class TestUpdateProfileAsync:
    def test_returns_extracted_preferences(self, patched_runtime: dict[str, Any]) -> None:
        patched_runtime["memory"].update_profile.return_value = {
            "default_team": "cthulhu",
            "frequent_metrics": ["velocity"],
        }
        out = mt.update_profile_async(str(uuid4()), str(uuid4()))
        assert out == {"default_team": "cthulhu", "frequent_metrics": ["velocity"]}

    def test_does_not_invoke_llm(self, patched_runtime: dict[str, Any]) -> None:
        # Per-turn path is intentionally LLM-free — pin it.
        mt.update_profile_async(str(uuid4()), str(uuid4()))
        patched_runtime["llm"].invoke.assert_not_called()

    def test_exception_returns_none(self, patched_runtime: dict[str, Any]) -> None:
        patched_runtime["memory"].update_profile.side_effect = RuntimeError("boom")
        assert mt.update_profile_async(str(uuid4()), str(uuid4())) is None

    def test_db_close_called_in_finally(self, patched_runtime: dict[str, Any]) -> None:
        mt.update_profile_async(str(uuid4()), str(uuid4()))
        patched_runtime["db"].close.assert_called_once()


# ===================================================================== #
# _format_dialog_compact
# ===================================================================== #


@pytest.mark.unit
class TestFormatDialogCompact:
    def test_short_dialog_returned_as_is(self) -> None:
        msgs = [
            _msg(0, role="user", content="hi"),
            _msg(1, role="assistant", content="hello"),
        ]
        out = mt._format_dialog_compact(msgs, max_chars=200)
        assert "User: hi" in out
        assert "Assistant: hello" in out
        assert "..." not in out

    def test_long_dialog_truncates_middle(self) -> None:
        # Head + tail preserved, marker between. Pinning the middle-cut
        # strategy because head-only or tail-only would lose user intent
        # OR resolution.
        msgs = [_msg(i, content="X" * 200) for i in range(20)]
        out = mt._format_dialog_compact(msgs, max_chars=400)
        assert "...[...]..." in out
        assert len(out) <= 500  # head/2 + marker + tail/2 + small margin


# ===================================================================== #
# _extract_topics
# ===================================================================== #


@pytest.mark.unit
class TestExtractTopics:
    def test_unique_values_in_order_of_first_occurrence(self) -> None:
        msgs = [
            _msg(0, meta={"entities": {"team_name": "cthulhu"}}),
            _msg(1, meta={"entities": {"team_name": "cthulhu"}}),  # dup
            _msg(2, meta={"entities": {"metric_name": "velocity"}}),
            _msg(3, meta={"entities": {"sprint_name": "Q1.1"}}),
            _msg(4, meta={"entities": {"team_name": "cthulhu"}}),  # dup
        ]
        assert mt._extract_topics(msgs) == ["cthulhu", "velocity", "Q1.1"]

    def test_skips_messages_without_entities(self) -> None:
        msgs = [
            _msg(0, meta={}),
            _msg(1, meta={"entities": {"team_name": "cthulhu"}}),
        ]
        assert mt._extract_topics(msgs) == ["cthulhu"]

    def test_skips_non_dict_entities(self) -> None:
        # Robustness against legacy messages whose entities was written
        # as a list — must not crash.
        msgs = [_msg(0, meta={"entities": ["broken"]})]
        assert mt._extract_topics(msgs) == []


# ===================================================================== #
# _refresh_rolling_summary
# ===================================================================== #


@pytest.mark.unit
class TestRefreshRollingSummary:
    def test_no_summaries_is_noop(self) -> None:
        memory = MagicMock()
        memory.summary_repo.get_recent.return_value = []
        llm = MagicMock()
        mt._refresh_rolling_summary(memory, uuid4(), llm)
        memory.profile_repo.update_context_summary.assert_not_called()
        llm.invoke.assert_not_called()

    def test_few_summaries_concatenated_without_llm(self) -> None:
        # Below _MIN_SUMMARIES_FOR_META (3) → just join the strings; an
        # LLM call would be wasteful for so little signal.
        memory = MagicMock()
        memory.summary_repo.get_recent.return_value = [
            _summary("a"),
            _summary("b"),
        ]
        llm = MagicMock()
        user_id = uuid4()
        mt._refresh_rolling_summary(memory, user_id, llm)
        memory.profile_repo.update_context_summary.assert_called_once_with(user_id, "a b")
        llm.invoke.assert_not_called()

    def test_many_summaries_call_llm_for_meta(self) -> None:
        memory = MagicMock()
        memory.summary_repo.get_recent.return_value = [_summary(f"summary {i}") for i in range(5)]
        llm = MagicMock()
        llm.invoke.return_value = "  meta summary  "
        user_id = uuid4()
        mt._refresh_rolling_summary(memory, user_id, llm)
        # LLM called with bulleted list and meta-summary persisted (stripped).
        llm.invoke.assert_called_once()
        memory.profile_repo.update_context_summary.assert_called_once_with(user_id, "meta summary")

    def test_exception_swallowed(self) -> None:
        memory = MagicMock()
        memory.summary_repo.get_recent.side_effect = RuntimeError("db blip")
        # Must not raise — best-effort path.
        mt._refresh_rolling_summary(memory, uuid4(), MagicMock())
