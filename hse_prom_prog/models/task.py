"""Define task domain models and enums.

This module exposes the business entities for task execution tracking.
No SQLAlchemy ORM models — we use raw SQL consistent with existing
codebase patterns.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class TaskStatus(StrEnum):
    """Enumerate task execution lifecycle states.

    Represents the lifecycle stages of an async task:
    - ``PENDING``: Task created and queued, waiting for processing.
    - ``PROCESSING``: Task actively being executed by a worker.
    - ``COMPLETED``: Task finished successfully with results.
    - ``FAILED``: Task encountered an error during execution.
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Task:
    """Domain model for task execution tracking.

    This is NOT a SQLAlchemy ORM model. It represents the business entity
    for task execution, consistent with the repository pattern using raw
    SQL.

    Attributes:
        task_id: Unique task identifier (UUID).
        query: User query to process.
        status: Current task status (PENDING/PROCESSING/COMPLETED/FAILED).
        result: Task execution result (dict with final_response and issue_key).
        error: Error message if task failed.
        celery_task_id: Internal Celery task identifier.
        created_at: Timestamp when task was created.
        started_at: Timestamp when task execution started.
        completed_at: Timestamp when task finished (success or failure).
        workflow_state: Full LangGraph state for debugging.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_id: UUID,
        query: str,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        celery_task_id: str | None = None,
        created_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        workflow_state: dict[str, Any] | None = None,
        conversation_id: UUID | None = None,
    ) -> None:
        """Initialize Task instance.

        Args:
            task_id: Unique task identifier.
            query: User query to process.
            status: Task status.
            result: Optional result data (for COMPLETED status).
            error: Optional error message (for FAILED status).
            celery_task_id: Optional Celery internal task ID.
            created_at: Task creation timestamp.
            started_at: Task start timestamp.
            completed_at: Task completion timestamp.
            workflow_state: Optional full workflow state for debugging.
            conversation_id: Optional short-term memory conversation id.
        """
        self.task_id = task_id
        self.query = query
        self.status = status
        self.result = result
        self.error = error
        self.celery_task_id = celery_task_id
        self.created_at = created_at
        self.started_at = started_at
        self.completed_at = completed_at
        self.workflow_state = workflow_state
        self.conversation_id = conversation_id

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the task.

        Returns:
            Dictionary representation with ISO-formatted timestamps.
        """
        return {
            "task_id": str(self.task_id),
            "query": self.query,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "conversation_id": (str(self.conversation_id) if self.conversation_id else None),
        }

    def __repr__(self) -> str:
        """Return a developer-friendly summary string.

        Returns:
            String showing task ID, status and a query preview.
        """
        return (
            f"Task(task_id={self.task_id}, status={self.status.value}, query={self.query[:50]}...)"
        )
