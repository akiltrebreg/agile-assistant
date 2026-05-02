"""Unit tests for ``api.routers.tasks`` (POST /tasks, GET /tasks/{id}).

Strategy: drive the router through FastAPI's ``TestClient``, with
``get_db`` and ``get_memory_manager`` overridden so no real DB is
touched. ``TaskRepository`` is patched at the router's import site so
its constructor returns our mock instead of opening connections.
``execute_workflow.apply_async`` (Celery enqueue) is also patched —
the API contract is "task is queued and a task_id comes back", not
"the worker runs to completion".

Bugs to catch:
  * 404 vs 200 contract on GET (missing task must not 500)
  * 422 contract on POST (Pydantic validation, not server-side handling)
  * `TaskCreateResponse` shape stability (clients poll `conversation_id`)
  * existing `conversation_id` honoured (no silent fresh conversation)
  * exception in the create-path surfaces as 500, not a Celery crash later
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from hse_prom_prog.api.app import app
from hse_prom_prog.api.dependencies import get_db, get_memory_manager
from hse_prom_prog.models.memory import Conversation
from hse_prom_prog.models.task import Task, TaskStatus

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


def _make_conversation(*, conversation_id: UUID | None = None) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=conversation_id or uuid4(),
        user_id=uuid4(),
        title=None,
        summary=None,
        summary_turn_index=0,
        created_at=now,
        updated_at=now,
        is_active=True,
    )


def _make_task(
    *,
    task_id: UUID | None = None,
    query: str = "тестовый запрос",
    status: TaskStatus = TaskStatus.PENDING,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> Task:
    now = datetime.now(UTC)
    return Task(
        task_id=task_id or uuid4(),
        query=query,
        status=status,
        result=result,
        error=error,
        created_at=now,
        started_at=None,
        completed_at=None,
        conversation_id=uuid4(),
    )


@pytest.fixture
def memory() -> MagicMock:
    """Mock MemoryManager + injected sub-repos used by the router."""
    mm = MagicMock()
    profile = MagicMock(id=uuid4())
    mm.profile_repo.get_or_create.return_value = profile
    mm.get_or_create_conversation.return_value = _make_conversation()
    return mm


@pytest.fixture
def task_repo() -> MagicMock:
    """Mock TaskRepository used to bypass DB on create_task / get_task."""
    return MagicMock()


@pytest.fixture
def client(memory: MagicMock, task_repo: MagicMock) -> TestClient:
    """TestClient with DB / memory dependencies overridden + TaskRepository
    patched at import site so the router never touches a real DB."""
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[get_memory_manager] = lambda: memory
    with patch("hse_prom_prog.api.routers.tasks.TaskRepository", return_value=task_repo):
        yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_celery_apply(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``execute_workflow.apply_async`` so POST /tasks returns
    immediately without enqueuing a real Celery task."""
    import hse_prom_prog.api.routers.tasks as tasks_module

    fake = MagicMock(return_value=MagicMock(id="celery-job-1"))
    monkeypatch.setattr(tasks_module.execute_workflow, "apply_async", fake)
    return fake


# ===================================================================== #
# POST /tasks
# ===================================================================== #


@pytest.mark.unit
class TestPostTasks:
    def test_valid_request_returns_202_with_task_payload(
        self,
        client: TestClient,
        memory: MagicMock,
        task_repo: MagicMock,
        mock_celery_apply: MagicMock,
    ) -> None:
        task_id = uuid4()
        conv = _make_conversation()
        memory.get_or_create_conversation.return_value = conv
        task_repo.create_task.return_value = _make_task(task_id=task_id)

        resp = client.post("/tasks", json={"query": "Расскажи о AL-1", "user_id": "user-abc"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == str(task_id)
        assert body["conversation_id"] == str(conv.id)
        assert body["status"] == "PENDING"
        assert "queued" in body["message"].lower()

    def test_apply_async_called_with_correct_args(
        self,
        client: TestClient,
        memory: MagicMock,
        task_repo: MagicMock,
        mock_celery_apply: MagicMock,
    ) -> None:
        task = _make_task()
        conv = _make_conversation()
        memory.get_or_create_conversation.return_value = conv
        task_repo.create_task.return_value = task

        client.post(
            "/tasks",
            json={"query": "vel", "user_id": "user-1", "conversation_id": str(conv.id)},
        )

        # Pin the Celery contract: positional args are [task_id, query]
        # and kwargs carry conversation_id + user_external_id + created_at_ts.
        kwargs = mock_celery_apply.call_args.kwargs
        assert kwargs["args"] == [str(task.task_id), "vel"]
        assert kwargs["kwargs"]["conversation_id"] == str(conv.id)
        assert kwargs["kwargs"]["user_external_id"] == "user-1"
        # Wallclock at enqueue is a float timestamp (queue-wait math depends on it).
        assert isinstance(kwargs["kwargs"]["created_at_ts"], float)

    def test_existing_conversation_id_passed_through(
        self,
        client: TestClient,
        memory: MagicMock,
        task_repo: MagicMock,
        mock_celery_apply: MagicMock,
    ) -> None:
        task_repo.create_task.return_value = _make_task()
        existing = uuid4()
        client.post(
            "/tasks",
            json={"query": "q", "user_id": "u1", "conversation_id": str(existing)},
        )
        # The router calls get_or_create with the exact UUID it received,
        # not a freshly-minted one.
        call_args = memory.get_or_create_conversation.call_args.args
        assert call_args[0] == existing

    def test_anonymous_request_skips_profile_lookup(
        self,
        client: TestClient,
        memory: MagicMock,
        task_repo: MagicMock,
        mock_celery_apply: MagicMock,
    ) -> None:
        # No user_id → router must not try to resolve a profile (anonymous
        # path keeps internal_user_id = None).
        task_repo.create_task.return_value = _make_task()
        client.post("/tasks", json={"query": "q"})
        memory.profile_repo.get_or_create.assert_not_called()
        # Conversation IS still created — anonymous users get a session too.
        memory.get_or_create_conversation.assert_called_once()
        assert memory.get_or_create_conversation.call_args.args[1] is None

    def test_empty_query_returns_422(self, client: TestClient) -> None:
        resp = client.post("/tasks", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client: TestClient) -> None:
        resp = client.post("/tasks", json={"user_id": "u1"})
        assert resp.status_code == 422

    def test_oversized_query_returns_422(self, client: TestClient) -> None:
        # max_length=10000 in the schema — 10001 chars must fail at
        # Pydantic validation, not get truncated server-side.
        resp = client.post("/tasks", json={"query": "x" * 10001})
        assert resp.status_code == 422

    def test_invalid_uuid_in_conversation_id_returns_422(self, client: TestClient) -> None:
        resp = client.post("/tasks", json={"query": "q", "conversation_id": "not-a-uuid"})
        assert resp.status_code == 422

    def test_db_error_during_create_returns_500(
        self,
        client: TestClient,
        memory: MagicMock,
        task_repo: MagicMock,
        mock_celery_apply: MagicMock,
    ) -> None:
        # If task_repo.create_task raises, the router catches and surfaces
        # a 500 with the original error text (not a Celery 202).
        task_repo.create_task.side_effect = RuntimeError("connection lost")
        resp = client.post("/tasks", json={"query": "q"})
        assert resp.status_code == 500
        assert "connection lost" in resp.json()["detail"]
        # Failure happened before enqueue → no Celery job created.
        mock_celery_apply.assert_not_called()


# ===================================================================== #
# GET /tasks/{task_id}
# ===================================================================== #


@pytest.mark.unit
class TestGetTask:
    def test_completed_task_returns_full_payload(
        self, client: TestClient, task_repo: MagicMock
    ) -> None:
        task_id = uuid4()
        task = _make_task(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result={"final_response": "Ответ", "issue_key": "AL-1"},
        )
        task_repo.get_task.return_value = task

        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == str(task_id)
        assert body["status"] == "COMPLETED"
        assert body["result"] == {"final_response": "Ответ", "issue_key": "AL-1"}
        assert body["error"] is None

    def test_pending_task_has_null_result(self, client: TestClient, task_repo: MagicMock) -> None:
        task = _make_task(status=TaskStatus.PENDING)
        task_repo.get_task.return_value = task
        resp = client.get(f"/tasks/{task.task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "PENDING"
        assert body["result"] is None
        assert body["completed_at"] is None

    def test_failed_task_carries_error_message(
        self, client: TestClient, task_repo: MagicMock
    ) -> None:
        task = _make_task(status=TaskStatus.FAILED, error="vLLM timeout")
        task_repo.get_task.return_value = task
        resp = client.get(f"/tasks/{task.task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "FAILED"
        assert body["error"] == "vLLM timeout"

    def test_missing_task_returns_404(self, client: TestClient, task_repo: MagicMock) -> None:
        # The contract: poll-friendly endpoint must 404, not 500, on
        # an unknown task_id — clients already retry on 5xx.
        task_repo.get_task.return_value = None
        resp = client.get(f"/tasks/{uuid4()}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_invalid_uuid_path_returns_422(self, client: TestClient) -> None:
        # FastAPI path validation — non-UUID never reaches the handler.
        resp = client.get("/tasks/not-a-uuid")
        assert resp.status_code == 422
