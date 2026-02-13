"""Task management API endpoints.

This module provides REST API endpoints for async task management:
- POST /tasks - Create and queue a new task
- GET /tasks/{task_id} - Get task status and result
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from hse_prom_prog.api.dependencies import get_db
from hse_prom_prog.api.schemas.task import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskResponse,
)
from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.database.task_repository import TaskRepository
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
) -> TaskCreateResponse:
    """Create and queue a new workflow task.

    This endpoint:
    1. Creates a task record in PostgreSQL with PENDING status
    2. Sends the task to Celery for async processing
    3. Returns immediately with HTTP 202 Accepted

    The client can then poll GET /tasks/{task_id} to check status.

    Args:
        request: Task creation request with user query.
        db: Database connection (injected dependency).

    Returns:
        TaskCreateResponse with task_id and initial status.

    Raises:
        HTTPException: 500 if task creation fails.
    """
    try:
        # Create task repository
        repo = TaskRepository(db)

        # Create task in database with PENDING status
        task = repo.create_task(query=request.query)

        # Send task to Celery for async execution
        celery_task = execute_workflow.apply_async(
            args=[str(task.task_id), request.query],
            task_id=None,  # Let Celery generate its own ID
        )

        logger.info(
            f"[API] Created task {task.task_id}, "
            f"Celery task {celery_task.id}, "
            f"query: {request.query[:50]}..."
        )

        return TaskCreateResponse(
            task_id=task.task_id,
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
    )
