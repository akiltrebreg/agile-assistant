"""API + workflow_task integration tests for the memory layer.

These tests use FastAPI's ``TestClient`` with dependency overrides and
mocked repositories — no real PostgreSQL or Celery broker is required.
They validate the wiring added in Part 5:

  * POST /tasks with conversation_id → task bound to that conversation
  * POST /tasks without conversation_id → new conversation created and
    returned in the 202 response
  * GET /conversations?user_id=... → returns the user's list
  * workflow_task._persist_turn_safe → save_turn called with the full
    metadata expected by downstream memory layers
  * workflow_task._maybe_rotate_stale_conversation → idle rotation (Part 8)
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from hse_prom_prog.api.app import app
from hse_prom_prog.api.dependencies import get_db, get_memory_manager
from hse_prom_prog.models.memory import Conversation, Message, UserProfile
from hse_prom_prog.models.task import Task, TaskStatus
from hse_prom_prog.tasks.workflow_task import (
    INACTIVITY_THRESHOLD,
    _maybe_rotate_stale_conversation,
    _persist_turn_safe,
)

# ────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ────────────────────────────────────────────────────────────────


def _profile(external_id: str = "ext-1", internal_id: UUID | None = None) -> UserProfile:
    return UserProfile(
        id=internal_id or uuid4(),
        external_id=external_id,
        display_name=None,
        preferences={},
        context_summary=None,
        total_conversations=0,
        total_messages=0,
        created_at=datetime(2026, 4, 24, 10, 0, 0),
        updated_at=datetime(2026, 4, 24, 10, 0, 0),
    )


def _conv(
    conv_id: UUID | None = None,
    user_id: UUID | None = None,
    title: str | None = "new chat",
) -> Conversation:
    return Conversation(
        id=conv_id or uuid4(),
        user_id=user_id,
        title=title,
        summary=None,
        summary_turn_index=0,
        created_at=datetime(2026, 4, 24, 10, 0, 0),
        updated_at=datetime(2026, 4, 24, 10, 0, 0),
        is_active=True,
    )


def _task(task_id: UUID, conv_id: UUID | None = None) -> Task:
    return Task(
        task_id=task_id,
        query="dummy",
        status=TaskStatus.PENDING,
        created_at=datetime(2026, 4, 24, 10, 0, 0),
        conversation_id=conv_id,
    )


def _mock_memory_manager(profile: UserProfile, conv: Conversation) -> MagicMock:
    memory = MagicMock()
    memory.profile_repo.get_or_create.return_value = profile
    memory.get_or_create_conversation.return_value = conv
    memory.conversation_repo.get.return_value = conv
    memory.conversation_repo.list_by_user.return_value = [conv]
    memory.conversation_repo.count_messages.return_value = 4
    memory.conversation_repo.get_messages.return_value = []
    return memory


def _install_overrides(memory: MagicMock, task_repo_factory: MagicMock) -> None:
    """Point FastAPI dependencies at our mocks."""
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[get_memory_manager] = lambda: memory
    # tasks.router constructs TaskRepository(db) inline — patch import site.
    import hse_prom_prog.api.routers.tasks as tasks_router

    tasks_router.TaskRepository = task_repo_factory  # type: ignore[attr-defined]


def _clear_overrides() -> None:
    app.dependency_overrides.clear()


# ────────────────────────────────────────────────────────────────
# POST /tasks — conversation wiring
# ────────────────────────────────────────────────────────────────


class TestCreateTaskWithMemory:
    def test_with_existing_conversation_id_reuses_it(self, monkeypatch) -> None:
        profile = _profile(external_id="user-A")
        existing_conv_id = uuid4()
        conv = _conv(conv_id=existing_conv_id, user_id=profile.id, title="Existing")
        memory = _mock_memory_manager(profile, conv)

        task_repo = MagicMock()
        task_repo.create_task.return_value = _task(uuid4(), existing_conv_id)
        task_repo_factory = MagicMock(return_value=task_repo)

        _install_overrides(memory, task_repo_factory)

        # Don't actually enqueue Celery — patch the dispatch.
        import hse_prom_prog.api.routers.tasks as tasks_router

        monkeypatch.setattr(
            tasks_router.execute_workflow,
            "apply_async",
            MagicMock(return_value=MagicMock(id="celery-1")),
        )

        try:
            client = TestClient(app)
            resp = client.post(
                "/tasks",
                json={
                    "query": "А что по velocity у этой команды?",
                    "conversation_id": str(existing_conv_id),
                    "user_id": "user-A",
                },
            )
        finally:
            _clear_overrides()

        assert resp.status_code == 202
        body = resp.json()
        assert body["conversation_id"] == str(existing_conv_id)
        assert body["status"] == "PENDING"
        # Memory manager was asked to resolve the exact id.
        memory.get_or_create_conversation.assert_called_once()
        called_conv_id, called_user_uuid = memory.get_or_create_conversation.call_args.args
        assert called_conv_id == existing_conv_id
        assert called_user_uuid == profile.id
        # Task row carries the conversation_id too.
        task_repo.create_task.assert_called_once()
        assert task_repo.create_task.call_args.kwargs["conversation_id"] == existing_conv_id

    def test_without_conversation_id_creates_new_and_returns_it(self, monkeypatch) -> None:
        profile = _profile(external_id="user-B")
        # Conversation auto-created by MemoryManager with a fresh UUID.
        fresh_conv = _conv(conv_id=uuid4(), user_id=profile.id, title=None)
        memory = _mock_memory_manager(profile, fresh_conv)

        task_repo = MagicMock()
        task_repo.create_task.return_value = _task(uuid4(), fresh_conv.id)
        task_repo_factory = MagicMock(return_value=task_repo)

        _install_overrides(memory, task_repo_factory)

        import hse_prom_prog.api.routers.tasks as tasks_router

        monkeypatch.setattr(
            tasks_router.execute_workflow,
            "apply_async",
            MagicMock(return_value=MagicMock(id="celery-2")),
        )

        try:
            client = TestClient(app)
            resp = client.post(
                "/tasks",
                json={"query": "Какой velocity у cthulhu?", "user_id": "user-B"},
            )
        finally:
            _clear_overrides()

        assert resp.status_code == 202
        body = resp.json()
        assert body["conversation_id"] == str(fresh_conv.id)
        # Called with None → manager created a fresh conversation.
        called_conv_id, _ = memory.get_or_create_conversation.call_args.args
        assert called_conv_id is None


# ────────────────────────────────────────────────────────────────
# GET /conversations
# ────────────────────────────────────────────────────────────────


class TestListConversations:
    def test_returns_list_for_known_user(self) -> None:
        profile = _profile(external_id="user-C")
        conv = _conv(conv_id=uuid4(), user_id=profile.id, title="Chat #1")
        memory = _mock_memory_manager(profile, conv)

        _install_overrides(memory, MagicMock())

        try:
            client = TestClient(app)
            resp = client.get("/conversations", params={"user_id": "user-C"})
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["id"] == str(conv.id)
        assert items[0]["title"] == "Chat #1"
        assert items[0]["message_count"] == 4
        assert items[0]["is_active"] is True
        # Correct user was resolved.
        memory.profile_repo.get_or_create.assert_called_once_with("user-C")
        memory.conversation_repo.list_by_user.assert_called_once()


# ────────────────────────────────────────────────────────────────
# workflow_task._persist_turn_safe
# ────────────────────────────────────────────────────────────────


class TestPersistTurnSafe:
    def test_save_turn_receives_correct_metadata(self) -> None:
        memory = MagicMock()
        conv_uuid = uuid4()
        workflow_result: dict = {
            "final_response": "Velocity: 42 SP.",
            "query_type": "sql",
            "intent": "metric",
            "entities": {"team_name": "cthulhu", "metric_name": "velocity"},
            "sql_query": "SELECT value FROM report_agile_dashboard_metrics WHERE ...",
        }

        _persist_turn_safe(memory, conv_uuid, "Какой velocity у cthulhu?", workflow_result, "t-1")

        memory.save_turn.assert_called_once()
        call = memory.save_turn.call_args
        assert call.kwargs["conversation_id"] == conv_uuid
        assert call.kwargs["user_message"] == "Какой velocity у cthulhu?"
        assert call.kwargs["bot_message"] == "Velocity: 42 SP."
        metadata = call.kwargs["metadata"]
        assert metadata["query_type"] == "sql"
        assert metadata["intent"] == "metric"
        assert metadata["entities"] == {"team_name": "cthulhu", "metric_name": "velocity"}
        assert metadata["last_sql"].startswith("SELECT value")

    def test_save_turn_errors_are_swallowed_and_logged(self) -> None:
        """A DB failure must not propagate — user's answer is already generated."""
        memory = MagicMock()
        memory.save_turn.side_effect = RuntimeError("db down")

        # Should not raise.
        _persist_turn_safe(
            memory,
            uuid4(),
            "query",
            {"final_response": "answer", "query_type": "simple"},
            "t-2",
        )
        memory.save_turn.assert_called_once()


# ────────────────────────────────────────────────────────────────
# workflow_task._maybe_rotate_stale_conversation (Part 8)
# ────────────────────────────────────────────────────────────────


def _stale_conv(
    *,
    user_id: UUID,
    idle: timedelta,
    is_active: bool = True,
) -> Conversation:
    """Conversation whose updated_at is ``idle`` behind now."""
    now = datetime.now(UTC)
    return Conversation(
        id=uuid4(),
        user_id=user_id,
        title="existing",
        summary=None,
        summary_turn_index=0,
        created_at=now - idle,
        updated_at=now - idle,
        is_active=is_active,
    )


class TestMaybeRotateStaleConversation:
    def test_rotates_when_idle_past_threshold_and_messages_exist(self) -> None:
        user_id = uuid4()
        stale = _stale_conv(user_id=user_id, idle=INACTIVITY_THRESHOLD + timedelta(minutes=1))
        fresh = _stale_conv(user_id=user_id, idle=timedelta(seconds=0))

        memory = MagicMock()
        memory.conversation_repo.count_messages.return_value = 4
        memory.conversation_repo.create.return_value = fresh

        result = _maybe_rotate_stale_conversation(memory, stale, user_id)

        assert result.id == fresh.id
        memory.conversation_repo.close.assert_called_once_with(stale.id)
        memory.conversation_repo.create.assert_called_once()

    def test_keeps_conversation_when_recently_updated(self) -> None:
        user_id = uuid4()
        fresh = _stale_conv(user_id=user_id, idle=timedelta(minutes=5))

        memory = MagicMock()
        result = _maybe_rotate_stale_conversation(memory, fresh, user_id)

        assert result is fresh
        memory.conversation_repo.close.assert_not_called()
        memory.conversation_repo.create.assert_not_called()

    def test_skips_rotation_for_empty_stale_conversation(self) -> None:
        """No messages → nothing worth summarising, reuse the empty shell."""
        user_id = uuid4()
        stale = _stale_conv(user_id=user_id, idle=timedelta(hours=2))

        memory = MagicMock()
        memory.conversation_repo.count_messages.return_value = 0

        result = _maybe_rotate_stale_conversation(memory, stale, user_id)

        assert result is stale
        memory.conversation_repo.close.assert_not_called()
        memory.conversation_repo.create.assert_not_called()

    def test_skips_rotation_for_anonymous_user(self) -> None:
        """Anon conversations can't be summarised (no profile) — leave alone."""
        stale = _stale_conv(user_id=uuid4(), idle=timedelta(hours=2))

        memory = MagicMock()
        result = _maybe_rotate_stale_conversation(memory, stale, user_uuid=None)

        assert result is stale
        memory.conversation_repo.close.assert_not_called()


# ────────────────────────────────────────────────────────────────
# GET /conversations/{id}/messages — round-trip
# ────────────────────────────────────────────────────────────────


class TestConversationMessages:
    def test_returns_messages_in_turn_order(self) -> None:
        profile = _profile(external_id="user-D")
        conv = _conv(conv_id=uuid4(), user_id=profile.id, title="Transcript")
        memory = _mock_memory_manager(profile, conv)

        messages = [
            Message(
                id=uuid4(),
                conversation_id=conv.id,
                turn_index=0,
                role="user",
                content="Velocity cthulhu?",
                content_truncated=None,
                metadata={"query_type": "sql"},
                created_at=datetime(2026, 4, 24, 10, 0, 0),
            ),
            Message(
                id=uuid4(),
                conversation_id=conv.id,
                turn_index=1,
                role="assistant",
                content="Velocity: 42 SP.",
                content_truncated=None,
                metadata={"query_type": "sql"},
                created_at=datetime(2026, 4, 24, 10, 0, 30),
            ),
        ]
        memory.conversation_repo.get_messages.return_value = messages
        _install_overrides(memory, MagicMock())

        try:
            client = TestClient(app)
            resp = client.get(f"/conversations/{conv.id}/messages")
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload) == 2
        assert payload[0]["role"] == "user"
        assert payload[0]["turn_index"] == 0
        assert payload[1]["role"] == "assistant"
        assert payload[1]["content"] == "Velocity: 42 SP."
