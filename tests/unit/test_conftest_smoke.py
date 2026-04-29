"""Smoke test: every fixture in conftest can be instantiated without errors.

This file exists only to validate the test infrastructure (Phase 0). It
is safe to delete once the first real unit tests land — they will exercise
the fixtures more thoroughly. Until then, this is the only thing that
fails fast if a fixture grows a typo or an import drift in the source tree.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_mock_llm_client_is_a_mock_with_invoke(mock_llm_client):
    assert isinstance(mock_llm_client, MagicMock)
    assert mock_llm_client.invoke("any prompt") == ""
    mock_llm_client.invoke.return_value = "hello"
    assert mock_llm_client.invoke("p") == "hello"


@pytest.mark.unit
def test_mock_db_returns_empty_rows_by_default(mock_db):
    assert mock_db.execute_query("SELECT 1") == []
    mock_db.execute_query.return_value = [{"k": "AL-1"}]
    assert mock_db.execute_query("SELECT k FROM t") == [{"k": "AL-1"}]


@pytest.mark.unit
def test_mock_qdrant_query_points_default_empty(mock_qdrant):
    result = mock_qdrant.query_points(collection_name="x", query=[0.1])
    assert result.points == []
    assert mock_qdrant.collection_exists("x") is True


@pytest.mark.unit
def test_mock_celery_intercepts_delay(mock_celery):
    # Each registered task module exposes .delay as an isolated MagicMock.
    expected = {
        "execute_workflow",
        "summarize_session",
        "update_profile_async",
        "sync_jira_data",
        "sync_knowledge_base",
    }
    assert set(mock_celery) == expected
    mock_celery["execute_workflow"].delay("task-123")
    mock_celery["execute_workflow"].delay.assert_called_once_with("task-123")


@pytest.mark.unit
def test_mock_memory_manager_returns_blank_state(mock_memory_manager):
    conv = mock_memory_manager.get_or_create_conversation(None, None)
    assert conv.is_active is True
    assert conv.title is None

    ctx = mock_memory_manager.get_context(conv.id, token_budget=800)
    assert ctx == {
        "summary": "",
        "recent_turns": [],
        "history_token_count": 0,
        "needs_summarization": False,
    }


@pytest.mark.unit
def test_sample_state_factory_applies_overrides(sample_state):
    state = sample_state(query_type="sql", entities={"issue_key": "AL-1"})
    assert state["query_type"] == "sql"
    assert state["entities"] == {"issue_key": "AL-1"}
    # Untouched defaults stay neutral.
    assert state["blocked"] is False
    assert state["sql_result"] == []
    assert state["conversation_id"] is None


@pytest.mark.unit
def test_fixed_uuid_and_frozen_now(fixed_uuid, frozen_now):
    assert str(fixed_uuid) == "00000000-0000-0000-0000-000000000001"
    assert frozen_now.year == 2026
    assert frozen_now.tzinfo is not None
