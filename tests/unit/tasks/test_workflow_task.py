"""Unit tests for ``workflow_task`` helpers.

The Celery task itself (``execute_workflow``) drags in DB, MemoryManager,
AgileWorkflow, retriever, LLM client, Langfuse, the judge enqueue, and
Celery's runtime. End-to-end coverage of that surface belongs in
contract / integration tests. Here we cover the pure helpers that hold
the actual decision logic:

  * ``_mark_processing_if_first`` — PENDING → PROCESSING transition
                                    happens once, retries don't reset
  * ``_resolve_memory_ids``       — external_id → internal_uuid +
                                    conversation lookup + rotation hand-off
  * ``_maybe_rotate_stale_conversation`` — 4 guards (anon, inactive,
                                    no messages, recent) and full rotate
  * ``_load_memory_payloads``     — context + profile loaders, with
                                    short-circuits for missing ids
  * ``_persist_turn_safe``        — turn metadata composition + error
                                    swallow (must NOT hide answer on save fail)
  * ``_enqueue_profile_refresh``  — lazy-imported Celery enqueue
  * ``_get_retriever_safe``       — Qdrant degradation path
  * ``_update_task_failed``       — last-resort FAILED writer
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from agile_assistant.models.memory import Conversation
from agile_assistant.models.task import TaskStatus
from agile_assistant.tasks import workflow_task as wt

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _conv(
    *,
    is_active: bool = True,
    updated_at: datetime | None = None,
    user_id: UUID | None = None,
) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=uuid4(),
        user_id=user_id,
        title=None,
        summary=None,
        summary_turn_index=0,
        created_at=now,
        updated_at=updated_at or now,
        is_active=is_active,
    )


def _celery_task_mock(retries: int = 0) -> MagicMock:
    """Build a stand-in for the bound Celery `self` arg."""
    task = MagicMock()
    task.request.retries = retries
    task.max_retries = 3
    return task


# ===================================================================== #
# _mark_processing_if_first
# ===================================================================== #


@pytest.mark.unit
class TestMarkProcessingIfFirst:
    def test_first_attempt_marks_processing(self) -> None:
        repo = MagicMock()
        wt._mark_processing_if_first(_celery_task_mock(retries=0), repo, uuid4(), "task-1")
        repo.update_task_status.assert_called_once()
        # Status arg must be PROCESSING — the API contract for in-flight tasks.
        assert repo.update_task_status.call_args.args[1] == TaskStatus.PROCESSING

    def test_retry_does_not_reset_status(self) -> None:
        # On retry the task is already PROCESSING — must not flip back.
        repo = MagicMock()
        wt._mark_processing_if_first(_celery_task_mock(retries=2), repo, uuid4(), "task-1")
        repo.update_task_status.assert_not_called()


# ===================================================================== #
# _resolve_memory_ids
# ===================================================================== #


@pytest.mark.unit
class TestResolveMemoryIds:
    def test_anonymous_no_conversation(self) -> None:
        # No external_id, no conversation_id → both Nones returned;
        # memory layer is not consulted for the user side.
        memory = MagicMock()
        conv_uuid, user_uuid = wt._resolve_memory_ids(memory, None, None)
        assert (conv_uuid, user_uuid) == (None, None)
        memory.profile_repo.get_or_create.assert_not_called()
        memory.get_or_create_conversation.assert_not_called()

    def test_external_id_only_resolves_user(self) -> None:
        memory = MagicMock()
        internal = uuid4()
        memory.profile_repo.get_or_create.return_value = MagicMock(id=internal)

        conv_uuid, user_uuid = wt._resolve_memory_ids(memory, None, "ext-1")
        assert conv_uuid is None
        assert user_uuid == internal

    def test_conversation_id_resolves_with_rotation_check(self) -> None:
        # Conversation exists, recently updated → not rotated, returned as-is.
        memory = MagicMock()
        internal = uuid4()
        conv = _conv(
            user_id=internal,
            updated_at=datetime.now(UTC),  # fresh
        )
        memory.profile_repo.get_or_create.return_value = MagicMock(id=internal)
        memory.get_or_create_conversation.return_value = conv

        conv_uuid, user_uuid = wt._resolve_memory_ids(memory, str(conv.id), "ext-1")
        assert conv_uuid == conv.id
        assert user_uuid == internal


# ===================================================================== #
# _maybe_rotate_stale_conversation — 4 guards + happy rotate
# ===================================================================== #


@pytest.mark.unit
class TestMaybeRotateStaleConversation:
    def test_anonymous_session_never_rotates(self) -> None:
        # user_uuid=None → anon; rotation would orphan the closed conv.
        memory = MagicMock()
        conv = _conv(updated_at=datetime.now(UTC) - timedelta(hours=1))
        out = wt._maybe_rotate_stale_conversation(memory, conv, None)
        assert out is conv
        memory.conversation_repo.close.assert_not_called()

    def test_inactive_conversation_does_not_rotate(self) -> None:
        # already closed → leave it alone (the user reopened intentionally)
        memory = MagicMock()
        conv = _conv(is_active=False, updated_at=datetime.now(UTC) - timedelta(hours=1))
        out = wt._maybe_rotate_stale_conversation(memory, conv, uuid4())
        assert out is conv

    def test_recent_conversation_does_not_rotate(self) -> None:
        # idle < threshold → keep current
        memory = MagicMock()
        conv = _conv(updated_at=datetime.now(UTC) - timedelta(minutes=5))
        out = wt._maybe_rotate_stale_conversation(memory, conv, uuid4())
        assert out is conv

    def test_empty_conversation_does_not_rotate(self) -> None:
        # idle BUT no messages — rotating would just waste a row.
        memory = MagicMock()
        memory.conversation_repo.count_messages.return_value = 0
        conv = _conv(updated_at=datetime.now(UTC) - timedelta(hours=1))
        out = wt._maybe_rotate_stale_conversation(memory, conv, uuid4())
        assert out is conv

    def test_stale_conversation_rotated_and_summary_enqueued(self) -> None:
        memory = MagicMock()
        memory.conversation_repo.count_messages.return_value = 3
        new_conv = _conv()
        memory.conversation_repo.create.return_value = new_conv

        old = _conv(updated_at=datetime.now(UTC) - timedelta(hours=2))
        user_uuid = uuid4()

        with patch("agile_assistant.tasks.memory_tasks.summarize_session") as fake_sum:
            out = wt._maybe_rotate_stale_conversation(memory, old, user_uuid)

        # Old closed, new created, summariser enqueued with both ids.
        assert out is new_conv
        memory.conversation_repo.close.assert_called_once_with(old.id)
        memory.conversation_repo.create.assert_called_once_with(user_id=user_uuid)
        fake_sum.apply_async.assert_called_once_with(args=[str(old.id), str(user_uuid)])

    def test_audit_repoint_when_task_repo_provided(self) -> None:
        # When task_uuid + task_repo are passed, the audit row is repointed
        # to the new conversation so debugging stays honest.
        memory = MagicMock()
        memory.conversation_repo.count_messages.return_value = 1
        new_conv = _conv()
        memory.conversation_repo.create.return_value = new_conv
        task_repo = MagicMock()
        task_uuid = uuid4()

        with patch("agile_assistant.tasks.memory_tasks.summarize_session"):
            wt._maybe_rotate_stale_conversation(
                memory,
                _conv(updated_at=datetime.now(UTC) - timedelta(hours=2)),
                uuid4(),
                task_uuid=task_uuid,
                task_repo=task_repo,
            )

        task_repo.update_conversation_id.assert_called_once_with(task_uuid, new_conv.id)

    def test_close_failure_does_not_block_rotation(self) -> None:
        # If close() raises, the rotation must still produce a fresh conv —
        # the user's CURRENT request can't be held up by housekeeping.
        memory = MagicMock()
        memory.conversation_repo.count_messages.return_value = 3
        memory.conversation_repo.close.side_effect = RuntimeError("db blip")
        new_conv = _conv()
        memory.conversation_repo.create.return_value = new_conv

        with patch("agile_assistant.tasks.memory_tasks.summarize_session"):
            out = wt._maybe_rotate_stale_conversation(
                memory,
                _conv(updated_at=datetime.now(UTC) - timedelta(hours=2)),
                uuid4(),
            )
        assert out is new_conv


# ===================================================================== #
# _load_memory_payloads
# ===================================================================== #


@pytest.mark.unit
class TestLoadMemoryPayloads:
    def test_full_payload(self) -> None:
        memory = MagicMock()
        memory.get_context.return_value = {
            "summary": "",
            "recent_turns": [],
            "history_token_count": 0,
            "needs_summarization": False,
        }
        memory.get_profile.return_value = {"preferences": {"default_team": "cthulhu"}}

        conv_uuid, user_uuid = uuid4(), uuid4()
        ctx, profile = wt._load_memory_payloads(memory, conv_uuid, user_uuid)
        memory.get_context.assert_called_once_with(conv_uuid, token_budget=wt.HISTORY_TOKEN_BUDGET)
        memory.get_profile.assert_called_once_with(user_uuid)
        assert ctx is not None
        assert profile == {"preferences": {"default_team": "cthulhu"}}

    def test_no_conversation_no_user_skips_both_loads(self) -> None:
        memory = MagicMock()
        ctx, profile = wt._load_memory_payloads(memory, None, None)
        assert ctx is None
        assert profile is None
        memory.get_context.assert_not_called()
        memory.get_profile.assert_not_called()

    def test_anonymous_with_conversation_loads_only_context(self) -> None:
        memory = MagicMock()
        memory.get_context.return_value = MagicMock()
        ctx, profile = wt._load_memory_payloads(memory, uuid4(), None)
        assert ctx is not None
        assert profile is None
        memory.get_profile.assert_not_called()


# ===================================================================== #
# _persist_turn_safe
# ===================================================================== #


@pytest.mark.unit
class TestPersistTurnSafe:
    def test_metadata_composed_from_workflow_result(self) -> None:
        memory = MagicMock()
        result: dict[str, Any] = {
            "query_type": "sql",
            "intent": "task",
            "entities": {"issue_key": "AL-1"},
            "sql_query": "SELECT * FROM report_agile_dashboard",
            "final_response": "Найдено",
        }
        wt._persist_turn_safe(memory, uuid4(), "вопрос", result, "task-1")

        memory.save_turn.assert_called_once()
        meta = memory.save_turn.call_args.kwargs["metadata"]
        assert meta == {
            "query_type": "sql",
            "intent": "task",
            "entities": {"issue_key": "AL-1"},
            "last_sql": "SELECT * FROM report_agile_dashboard",
        }

    def test_save_failure_is_swallowed(self) -> None:
        # Persistence failures must NOT propagate — the user has already
        # seen the response and a save error is purely housekeeping.
        memory = MagicMock()
        memory.save_turn.side_effect = RuntimeError("postgres down")
        # Must not raise.
        wt._persist_turn_safe(memory, uuid4(), "q", {"final_response": "reply"}, "task-1")

    def test_missing_sql_query_omits_last_sql_key(self) -> None:
        memory = MagicMock()
        wt._persist_turn_safe(
            memory,
            uuid4(),
            "q",
            {"query_type": "rag", "intent": "general", "final_response": "x"},
            "task-1",
        )
        meta = memory.save_turn.call_args.kwargs["metadata"]
        assert "last_sql" not in meta


# ===================================================================== #
# _enqueue_profile_refresh
# ===================================================================== #


@pytest.mark.unit
class TestEnqueueProfileRefresh:
    def test_enqueues_with_string_uuids(self) -> None:
        user, conv = uuid4(), uuid4()
        with patch("agile_assistant.tasks.memory_tasks.update_profile_async") as fake:
            wt._enqueue_profile_refresh(user, conv)
        fake.apply_async.assert_called_once_with(args=[str(user), str(conv)])

    def test_enqueue_failure_is_swallowed(self) -> None:
        # If the import or apply_async fails, the workflow must not crash
        # (the response is already saved at this point).
        with patch("agile_assistant.tasks.memory_tasks.update_profile_async") as fake:
            fake.apply_async.side_effect = RuntimeError("broker down")
            wt._enqueue_profile_refresh(uuid4(), uuid4())  # must not raise


# ===================================================================== #
# _get_retriever_safe
# ===================================================================== #


@pytest.mark.unit
class TestGetRetrieverSafe:
    def test_returns_retriever_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        monkeypatch.setattr(wt, "get_retriever", lambda: sentinel)
        assert wt._get_retriever_safe() is sentinel

    def test_returns_none_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Qdrant unreachable → return None so the workflow's RAG branch
        # degrades to "no context" instead of crashing.
        def _boom() -> Any:
            raise ConnectionError("qdrant down")

        monkeypatch.setattr(wt, "get_retriever", _boom)
        assert wt._get_retriever_safe() is None


# ===================================================================== #
# _update_task_failed
# ===================================================================== #


@pytest.mark.unit
class TestUpdateTaskFailed:
    def test_updates_status_via_existing_db(self) -> None:
        db = MagicMock()
        with patch.object(wt, "TaskRepository") as fake_repo_cls:
            repo = fake_repo_cls.return_value
            wt._update_task_failed(db, uuid4(), "task-1", "boom")
        # Constructed against the supplied db (not a fresh get_database()).
        fake_repo_cls.assert_called_once_with(db)
        # Status is FAILED with the error string passed through verbatim.
        args = repo.update_task_status.call_args
        assert args.args[1] == TaskStatus.FAILED
        assert args.kwargs["error"] == "boom"

    def test_falls_back_to_get_database_when_db_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If db handle is None (early failure before DB acquired), the
        # writer opens a fresh connection so the FAILED row still lands.
        fresh_db = MagicMock()
        monkeypatch.setattr(wt, "get_database", lambda: fresh_db)
        with patch.object(wt, "TaskRepository") as fake_repo_cls:
            wt._update_task_failed(None, uuid4(), "task-1", "early fail")
        fake_repo_cls.assert_called_once_with(fresh_db)

    def test_swallows_secondary_failures(self) -> None:
        # The "mark as FAILED" path is a last resort — a second exception
        # here would be lost forever. Swallow + log so the user sees the
        # ORIGINAL error rather than a cascading "could not mark failed".
        db = MagicMock()
        with patch.object(wt, "TaskRepository") as fake_repo_cls:
            fake_repo_cls.return_value.update_task_status.side_effect = RuntimeError("double fault")
            wt._update_task_failed(db, uuid4(), "task-1", "boom")  # must not raise


# ===================================================================== #
# Constants
# ===================================================================== #


@pytest.mark.unit
class TestConstants:
    def test_history_token_budget_pinned(self) -> None:
        # Bumping this requires raising vLLM max_model_len — pin so a
        # well-meaning increase to "give the assistant more memory" hits
        # this assertion before it hits a 400 from vLLM in production.
        assert wt.HISTORY_TOKEN_BUDGET == 800

    def test_inactivity_threshold_pinned(self) -> None:
        # 30 min — relaxing this without checking the sidebar UX is a
        # silent product change.
        assert timedelta(minutes=30) == wt.INACTIVITY_THRESHOLD
