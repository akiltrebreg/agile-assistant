"""Unit tests for ``api.routers.conversations``.

Endpoints under test:

  * GET  /conversations                — list newest-first
  * GET  /conversations/{id}/messages  — full transcript
  * POST /conversations/{id}/close     — mark closed + enqueue summariser

The router talks to ``MemoryManager`` and (for close) to the
``summarize_session`` Celery task. Both are mocked. ``get_db`` is
overridden to a noop because the manager mock doesn't actually
query through it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from agile_assistant.api.app import app
from agile_assistant.api.dependencies import get_db, get_memory_manager
from agile_assistant.models.memory import Conversation, Message

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _conv(
    *,
    title: str | None = "тест",
    user_id: UUID | None = None,
    is_active: bool = True,
) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=uuid4(),
        user_id=user_id or uuid4(),
        title=title,
        summary=None,
        summary_turn_index=0,
        created_at=now,
        updated_at=now,
        is_active=is_active,
    )


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


@pytest.fixture
def memory() -> MagicMock:
    mm = MagicMock()
    profile = MagicMock(id=uuid4())
    mm.profile_repo.get_or_create.return_value = profile
    mm.conversation_repo.list_by_user.return_value = []
    mm.conversation_repo.count_messages.return_value = 0
    mm.conversation_repo.get.return_value = None
    mm.conversation_repo.get_messages.return_value = []
    return mm


@pytest.fixture
def client(memory: MagicMock) -> TestClient:
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[get_memory_manager] = lambda: memory
    yield TestClient(app)
    app.dependency_overrides.clear()


# ===================================================================== #
# GET /conversations
# ===================================================================== #


@pytest.mark.unit
class TestListConversations:
    def test_returns_user_conversations(self, client: TestClient, memory: MagicMock) -> None:
        conv1 = _conv(title="первый")
        conv2 = _conv(title="второй", is_active=False)
        memory.conversation_repo.list_by_user.return_value = [conv1, conv2]
        memory.conversation_repo.count_messages.side_effect = [4, 2]

        resp = client.get("/conversations", params={"user_id": "u1"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["title"] == "первый"
        assert body[0]["message_count"] == 4
        assert body[0]["is_active"] is True
        assert body[1]["is_active"] is False

    def test_user_id_query_param_required(self, client: TestClient) -> None:
        resp = client.get("/conversations")
        assert resp.status_code == 422

    def test_pagination_passes_through(self, client: TestClient, memory: MagicMock) -> None:
        memory.conversation_repo.list_by_user.return_value = []
        client.get("/conversations", params={"user_id": "u1", "limit": 10, "offset": 30})
        kwargs = memory.conversation_repo.list_by_user.call_args.kwargs
        assert kwargs == {"limit": 10, "offset": 30}

    @pytest.mark.parametrize(
        "params",
        [
            {"user_id": "u1", "limit": 0},  # below ge=1
            {"user_id": "u1", "limit": 200},  # above le=100
            {"user_id": "u1", "offset": -1},  # below ge=0
        ],
    )
    def test_invalid_pagination_returns_422(
        self, client: TestClient, params: dict[str, Any]
    ) -> None:
        resp = client.get("/conversations", params=params)
        assert resp.status_code == 422

    def test_profile_lookup_failure_returns_500(
        self, client: TestClient, memory: MagicMock
    ) -> None:
        memory.profile_repo.get_or_create.side_effect = RuntimeError("db down")
        resp = client.get("/conversations", params={"user_id": "u1"})
        assert resp.status_code == 500
        assert "Profile lookup failed" in resp.json()["detail"]


# ===================================================================== #
# GET /conversations/{id}/messages
# ===================================================================== #


@pytest.mark.unit
class TestGetMessages:
    def test_existing_conversation_returns_transcript(
        self, client: TestClient, memory: MagicMock
    ) -> None:
        conv = _conv()
        memory.conversation_repo.get.return_value = conv
        memory.conversation_repo.get_messages.return_value = [
            _msg(0, role="user", content="вопрос"),
            _msg(1, role="assistant", content="ответ", meta={"intent": "task"}),
        ]
        resp = client.get(f"/conversations/{conv.id}/messages")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["role"] == "user"
        assert body[0]["content"] == "вопрос"
        assert body[0]["turn_index"] == 0
        assert body[1]["metadata"]["intent"] == "task"

    def test_missing_conversation_returns_404(self, client: TestClient, memory: MagicMock) -> None:
        memory.conversation_repo.get.return_value = None
        resp = client.get(f"/conversations/{uuid4()}/messages")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_invalid_uuid_path_returns_422(self, client: TestClient) -> None:
        resp = client.get("/conversations/not-a-uuid/messages")
        assert resp.status_code == 422

    def test_limit_param_passed_to_repo(self, client: TestClient, memory: MagicMock) -> None:
        conv = _conv()
        memory.conversation_repo.get.return_value = conv
        client.get(f"/conversations/{conv.id}/messages", params={"limit": 50})
        memory.conversation_repo.get_messages.assert_called_once_with(conv.id, limit=50)

    def test_metadata_defaults_to_empty_dict(self, client: TestClient, memory: MagicMock) -> None:
        # A persisted message with metadata=None must round-trip as {}
        # (the schema's default_factory) so the UI can read keys safely.
        conv = _conv()
        memory.conversation_repo.get.return_value = conv
        msg = _msg(0)
        msg.metadata = None  # type: ignore[assignment]
        memory.conversation_repo.get_messages.return_value = [msg]
        resp = client.get(f"/conversations/{conv.id}/messages")
        assert resp.json()[0]["metadata"] == {}


# ===================================================================== #
# POST /conversations/{id}/close
# ===================================================================== #


@pytest.mark.unit
class TestCloseConversation:
    def test_close_active_conversation_marks_inactive_and_enqueues_summary(
        self, client: TestClient, memory: MagicMock
    ) -> None:
        conv = _conv(user_id=uuid4(), is_active=True)
        memory.conversation_repo.get.return_value = conv

        # Patch the lazy-imported summarize_session.apply_async to avoid Celery.
        with patch("agile_assistant.tasks.memory_tasks.summarize_session") as fake_task:
            fake_task.apply_async.return_value = MagicMock(id="celery-sum-1")
            resp = client.post(f"/conversations/{conv.id}/close")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(conv.id)
        assert body["is_active"] is False
        assert body["summarize_task_id"] == "celery-sum-1"
        # Conversation MUST be closed in the repo (it's the only mutating
        # call the endpoint makes besides the optional Celery enqueue).
        memory.conversation_repo.close.assert_called_once_with(conv.id)
        fake_task.apply_async.assert_called_once_with(args=[str(conv.id), str(conv.user_id)])

    def test_close_anonymous_conversation_does_not_enqueue_summary(
        self, client: TestClient, memory: MagicMock
    ) -> None:
        # Anonymous session (user_id=None) has no profile to anchor a
        # summary to — the close still succeeds, summarize_task_id is None.
        anon = _conv(user_id=None)
        anon.user_id = None  # _conv() forces a uuid; override.
        memory.conversation_repo.get.return_value = anon

        with patch("agile_assistant.tasks.memory_tasks.summarize_session") as fake_task:
            resp = client.post(f"/conversations/{anon.id}/close")

        assert resp.status_code == 200
        assert resp.json()["summarize_task_id"] is None
        fake_task.apply_async.assert_not_called()
        memory.conversation_repo.close.assert_called_once()

    def test_close_missing_conversation_returns_404(
        self, client: TestClient, memory: MagicMock
    ) -> None:
        memory.conversation_repo.get.return_value = None
        resp = client.post(f"/conversations/{uuid4()}/close")
        assert resp.status_code == 404
        # 404 path must NOT call close() — otherwise a fat-finger UUID
        # would silently archive a random conversation.
        memory.conversation_repo.close.assert_not_called()

    def test_close_invalid_uuid_returns_422(self, client: TestClient) -> None:
        resp = client.post("/conversations/not-a-uuid/close")
        assert resp.status_code == 422
