"""Task management API endpoints.

This module provides REST API endpoints for async task management:
- POST /tasks - Create and queue a new task
- GET /tasks/{task_id} - Get task status and result
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from hse_prom_prog.api.dependencies import get_db, get_memory_manager
from hse_prom_prog.api.schemas.task import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskResponse,
)
from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.database.task_repository import TaskRepository
from hse_prom_prog.memory.manager import MemoryManager
from hse_prom_prog.tasks.workflow_task import execute_workflow

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post(
    "",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a new workflow task",
    description="Submits a query for async processing. Returns immediately with task_id.",
)
def create_task(
    request: TaskCreateRequest,
    db: DatabaseConnection = Depends(get_db),
    memory: MemoryManager = Depends(get_memory_manager),
) -> TaskCreateResponse:
    """Create and queue a new workflow task.

    When ``user_id`` or ``conversation_id`` are provided, the memory layer
    is engaged: the user profile is resolved (or created), and either the
    existing conversation is looked up or a fresh one is created and
    returned in the response. The Celery task then replays this id.

    Args:
        request: Task creation request with user query and optional
            memory-layer identifiers.
        db: Database connection (injected dependency).
        memory: Memory manager (injected dependency).

    Returns:
        TaskCreateResponse with task_id, conversation_id, and initial status.

    Raises:
        HTTPException: 500 if task creation fails.
    """
    try:
        # Resolve user id: external → internal UUID. Anonymous sessions pass
        # None all the way through — conversation is still created but has
        # no user_id FK, so it never shows up in any sidebar.
        internal_user_id: UUID | None = None
        if request.user_id:
            profile = memory.profile_repo.get_or_create(request.user_id)
            internal_user_id = profile.id

        conv = memory.get_or_create_conversation(request.conversation_id, internal_user_id)

        task_repo = TaskRepository(db)
        task = task_repo.create_task(
            query=request.query,
            conversation_id=conv.id,
        )

        celery_task = execute_workflow.apply_async(
            args=[str(task.task_id), request.query],
            kwargs={
                "conversation_id": str(conv.id),
                "user_external_id": request.user_id,
            },
            task_id=None,
        )

        logger.info(
            "[API] Created task %s (celery=%s, conversation=%s, user=%s)",
            task.task_id,
            celery_task.id,
            conv.id,
            request.user_id,
        )

        return TaskCreateResponse(
            task_id=task.task_id,
            conversation_id=conv.id,
            status=task.status.value,
            message="Task created and queued for processing",
        )

    except Exception as e:
        logger.error(f"[API] Failed to create task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {e!s}",
        ) from e


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get task status and result",
    description="Retrieves current status and result of a task by ID.",
)
def get_task(
    task_id: UUID,
    db: DatabaseConnection = Depends(get_db),
) -> TaskResponse:
    """Retrieve task status and result.

    This endpoint is idempotent and safe to poll repeatedly.
    Returns current task status:
    - PENDING: Task queued, waiting for worker
    - PROCESSING: Task actively being executed
    - COMPLETED: Task finished successfully (result available)
    - FAILED: Task encountered an error (error message available)

    Args:
        task_id: Task UUID to retrieve.
        db: Database connection (injected dependency).

    Returns:
        TaskResponse with current status, result (if completed), or error (if failed).

    Raises:
        HTTPException: 404 if task not found.
    """
    repo = TaskRepository(db)
    task = repo.get_task(task_id)

    if not task:
        logger.warning(f"[API] Task {task_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    logger.debug(f"[API] Retrieved task {task_id}, status: {task.status.value}")

    return TaskResponse(
        task_id=task.task_id,
        query=task.query,
        status=task.status.value,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        conversation_id=task.conversation_id,
    )
