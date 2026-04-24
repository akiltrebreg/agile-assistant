"""Conversation management API endpoints.

Provides the sidebar + transcript surface for the Streamlit UI:
- GET  /conversations             — list conversations for a user
- GET  /conversations/{id}/messages — full transcript
- POST /conversations/{id}/close  — mark closed, kick off summary
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from hse_prom_prog.api.dependencies import get_memory_manager
from hse_prom_prog.api.schemas.conversation import (
    ConversationCloseResponse,
    ConversationSummaryResponse,
    MessageResponse,
)
from hse_prom_prog.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get(
    "",
    response_model=list[ConversationSummaryResponse],
    summary="List conversations for a user",
)
def list_conversations(
    user_id: str = Query(..., description="External user id (cookie UUID / SSO id)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    memory: MemoryManager = Depends(get_memory_manager),
) -> list[ConversationSummaryResponse]:
    """Return the user's conversations, newest first.

    Resolves ``user_id`` (external) to the internal profile UUID, then
    hits ``ConversationRepository.list_by_user``. Each entry carries a
    fresh ``message_count`` so the sidebar can show turn counts without
    a second round-trip.
    """
    try:
        profile = memory.profile_repo.get_or_create(user_id)
    except Exception as e:
        logger.error("[API/conversations] Failed to resolve user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile lookup failed: {e}",
        ) from e

    conversations = memory.conversation_repo.list_by_user(profile.id, limit=limit, offset=offset)
    return [
        ConversationSummaryResponse(
            id=conv.id,
            title=conv.title,
            updated_at=conv.updated_at,
            is_active=conv.is_active,
            message_count=memory.conversation_repo.count_messages(conv.id),
        )
        for conv in conversations
    ]


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="Full transcript of a conversation",
)
def get_messages(
    conversation_id: UUID,
    limit: int = Query(200, ge=1, le=500),
    memory: MemoryManager = Depends(get_memory_manager),
) -> list[MessageResponse]:
    """Return all messages for a conversation in ascending turn order."""
    conv = memory.conversation_repo.get(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    messages = memory.conversation_repo.get_messages(conversation_id, limit=limit)
    return [
        MessageResponse(
            role=msg.role,
            content=msg.content,
            metadata=msg.metadata or {},
            created_at=msg.created_at,
            turn_index=msg.turn_index,
        )
        for msg in messages
    ]


@router.post(
    "/{conversation_id}/close",
    response_model=ConversationCloseResponse,
    summary="Close a conversation and kick off async summarisation",
)
def close_conversation(
    conversation_id: UUID,
    memory: MemoryManager = Depends(get_memory_manager),
) -> ConversationCloseResponse:
    """Mark a conversation closed and schedule summarisation.

    Summarisation runs asynchronously via Celery — the endpoint returns
    immediately with the task id, which is advisory (clients don't need
    to poll it; the next sidebar refresh will show the updated summary).
    """
    conv = memory.conversation_repo.get(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    memory.conversation_repo.close(conversation_id)

    summarize_task_id: str | None = None
    if conv.user_id is not None:
        try:
            # Lazy import to avoid tasks <-> api circular at module load.
            from hse_prom_prog.tasks.memory_tasks import summarize_session  # noqa: PLC0415

            celery_task = summarize_session.apply_async(
                args=[str(conversation_id), str(conv.user_id)]
            )
            summarize_task_id = str(celery_task.id)
        except Exception as e:  # pragma: no cover — Celery unavailable in unit tests
            logger.warning(
                "[API/conversations] Could not enqueue summarisation for %s: %s",
                conversation_id,
                e,
            )

    return ConversationCloseResponse(
        id=conversation_id,
        is_active=False,
        summarize_task_id=summarize_task_id,
    )
