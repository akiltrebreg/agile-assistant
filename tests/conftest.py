"""Shared pytest fixtures for the unit and contract test suites.

Design principle: every fixture here returns a configurable ``MagicMock``
that tests can either inject directly (DI-style, when the SUT accepts
the dependency in its constructor) or wire in via ``monkeypatch.setattr``
on the import site (when the SUT instantiates its dependency internally).

No fixture in this file performs real I/O. Tests that need a live
database / Qdrant / Redis must be marked ``@pytest.mark.integration``
and are excluded from the default CI run.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

# Force tracing off for the entire test session before any project module
# (which transitively imports ``tracing``) is loaded. Setting this in
# pyproject env or a session-scoped fixture is too late: ``tracing._initialize``
# runs at import time, and a partially-loaded Langfuse client can attempt
# network I/O during test collection.
os.environ.setdefault("LANGFUSE_ENABLED", "false")
os.environ.setdefault("JUDGE_ENABLED", "false")

# Disable cross-encoder reranker by default in the test suite. Constructing
# Reranker eagerly triggers ensure_reranker_model_downloaded(), which would
# try to write into ``embedding_model_cache_dir`` (defaults to ``/app/models``,
# unwritable on a host pytest run). Tests that exercise the reranker path
# patch ``rag_agent.get_reranker`` directly so this default doesn't hide
# regressions in that branch.
os.environ.setdefault("RERANKER_ENABLED", "false")


# --------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------- #


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """A drop-in replacement for ``LLMClient``.

    Default behaviour: ``.invoke(...)`` returns an empty string. Tests
    override via ``return_value`` (single response) or ``side_effect``
    (sequenced or exception-raising responses):

        mock_llm_client.invoke.return_value = '{"intent": "task"}'
        mock_llm_client.invoke.side_effect = ["first", "second"]
        mock_llm_client.invoke.side_effect = TimeoutError()

    Most agents instantiate ``LLMClient()`` themselves, so wire the mock
    in with ``monkeypatch.setattr``:

        monkeypatch.setattr(
            "hse_prom_prog.agents.supervisor.LLMClient",
            lambda **_: mock_llm_client,
        )
    """
    from hse_prom_prog.llm.client import LLMClient

    client = MagicMock(spec=LLMClient)
    client.invoke = MagicMock(return_value="")
    client.model = "test-model"
    client.temperature = 0.0
    client.max_tokens = 600
    return client


# --------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------- #


@pytest.fixture
def mock_db() -> MagicMock:
    """A drop-in replacement for ``DatabaseConnection``.

    Default behaviour: ``.execute_query(...)`` returns ``[]`` (empty
    result set). Tests override per-call:

        mock_db.execute_query.return_value = [{"key": "AL-1", "summary": "x"}]
        mock_db.execute_query.side_effect = [rows_call1, rows_call2]
        mock_db.execute_query.side_effect = SQLAlchemyError("boom")

    For the SQL Agent, ``execute_query`` is the only method the SUT
    actually touches. ``test_connection`` and ``get_session`` are also
    spec'd in case a test needs them.
    """
    from hse_prom_prog.database.connection import DatabaseConnection

    db = MagicMock(spec=DatabaseConnection)
    db.execute_query = MagicMock(return_value=[])
    db.test_connection = MagicMock(return_value=True)
    db.database_url = "postgresql://test:test@localhost:5432/test"
    return db


# --------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------- #


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """A drop-in replacement for ``qdrant_client.QdrantClient``.

    Default behaviour: ``.collection_exists()`` returns True; ``.query_points``
    returns an object with ``.points = []``. Tests build a list of
    ``ScoredPoint``-shaped MagicMocks for richer scenarios:

        point = MagicMock(score=0.87, payload={"text": "...", "source": "doc.pdf"})
        mock_qdrant.query_points.return_value = MagicMock(points=[point])
    """
    client = MagicMock()
    client.collection_exists = MagicMock(return_value=True)
    client.query_points = MagicMock(return_value=MagicMock(points=[]))
    client.search = MagicMock(return_value=[])
    client.scroll = MagicMock(return_value=([], None))
    return client


# --------------------------------------------------------------------- #
# Celery
# --------------------------------------------------------------------- #


@pytest.fixture
def mock_celery(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch ``.delay`` / ``.apply_async`` on every Celery task in the project.

    Returns a dict keyed by task module attribute name so a test can
    assert on enqueue calls without executing the underlying logic:

        mock_celery["workflow_task"].delay.assert_called_once_with(...)
        mock_celery["summarize_session"].delay.assert_not_called()

    The mocks replace the ``.delay`` and ``.apply_async`` bound methods
    only — the task object itself remains intact, so call sites that do
    ``workflow_task.s(...)`` or read ``.name`` keep working.
    """
    from hse_prom_prog.tasks import memory_tasks, sync_tasks
    from hse_prom_prog.tasks import workflow_task as workflow_task_mod

    targets: dict[str, Any] = {
        "execute_workflow": workflow_task_mod.execute_workflow,
        "summarize_session": memory_tasks.summarize_session,
        "update_profile_async": memory_tasks.update_profile_async,
        "sync_jira_data": sync_tasks.sync_jira_data,
        "sync_knowledge_base": sync_tasks.sync_knowledge_base,
    }

    mocks: dict[str, MagicMock] = {}
    for name, task in targets.items():
        delay_mock = MagicMock(name=f"{name}.delay", return_value=MagicMock(id=str(uuid4())))
        apply_mock = MagicMock(name=f"{name}.apply_async", return_value=MagicMock(id=str(uuid4())))
        monkeypatch.setattr(task, "delay", delay_mock, raising=False)
        monkeypatch.setattr(task, "apply_async", apply_mock, raising=False)
        # Expose both via .delay (most common) and as a wrapper exposing both.
        wrapper = MagicMock(name=name)
        wrapper.delay = delay_mock
        wrapper.apply_async = apply_mock
        mocks[name] = wrapper

    return mocks


# --------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------- #


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """A drop-in replacement for ``MemoryManager``.

    Defaults give a "blank slate" memory state so a test that doesn't
    care about history/profile gets a working stub with no setup:

      * ``get_or_create_conversation`` → fresh ``Conversation`` (active=True, no title)
      * ``get_context`` → empty ``ConversationContext``
      * ``get_profile`` → ``None`` (no preferences)
      * ``save_turn`` → ``(0, 1)`` (first user/assistant pair)
      * ``update_profile`` → ``{}``

    Override per-test by reassigning ``return_value``:

        ctx = ConversationContext(summary="prev", recent_turns=[...], ...)
        mock_memory_manager.get_context.return_value = ctx
    """
    from hse_prom_prog.memory.manager import MemoryManager
    from hse_prom_prog.models.memory import Conversation, ConversationContext

    now = datetime.now(UTC)
    fresh_conversation = Conversation(
        id=uuid4(),
        user_id=uuid4(),
        title=None,
        summary=None,
        summary_turn_index=0,
        created_at=now,
        updated_at=now,
        is_active=True,
    )
    empty_context: ConversationContext = {
        "summary": "",
        "recent_turns": [],
        "history_token_count": 0,
        "needs_summarization": False,
    }

    mm = MagicMock(spec=MemoryManager)
    mm.get_or_create_conversation = MagicMock(return_value=fresh_conversation)
    mm.get_context = MagicMock(return_value=empty_context)
    mm.get_profile = MagicMock(return_value=None)
    mm.get_or_create_profile_by_external_id = MagicMock(
        return_value={
            "id": str(uuid4()),
            "external_id": "test-user",
            "display_name": None,
            "preferences": {},
            "context_summary": None,
            "total_conversations": 0,
            "total_messages": 0,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
    )
    mm.save_turn = MagicMock(return_value=(0, 1))
    mm.update_profile = MagicMock(return_value={})
    return mm


# --------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------- #


@pytest.fixture
def sample_state():
    """Factory fixture — returns a callable that builds a fresh ``WorkflowState``.

    Default state is the "neutral" shape every node sees on entry:
    empty messages, no entities, no SQL/RAG result, no error, not blocked.
    Tests override only the keys they care about:

        state = sample_state(query_type="sql", entities={"issue_key": "AL-1"})
        state = sample_state(blocked=True, guard_result={"reason": "injection"})

    Returning a dict (not the TypedDict class) so monkeypatching keys at
    runtime works without type-checker complaints — LangGraph also works
    with plain dicts since ``WorkflowState`` is ``total=False``.
    """

    def _make(**overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "messages": [],
            "original_query": "",
            "intent": "",
            "entities": {},
            "query_type": "",
            "route": "",
            "sql_query": "",
            "sql_result": [],
            "rag_response": "",
            "rag_sources": [],
            "error": "",
            "validation_result": {},
            "final_response": "",
            "blocked": False,
            "guard_result": {},
            "conversation_id": None,
            "user_id": None,
            "conversation_context": None,
            "user_profile": None,
        }
        defaults.update(overrides)
        return defaults

    return _make


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


@pytest.fixture
def fixed_uuid() -> UUID:
    """Deterministic UUID for tests that compare against an expected value."""
    return UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def frozen_now() -> datetime:
    """Fixed UTC timestamp for tests that compare timestamps."""
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
