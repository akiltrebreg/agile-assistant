"""Repository for task CRUD operations using raw SQL.

This module provides data access layer for task entities, following the
repository pattern with raw SQL queries (no ORM) consistent with existing codebase.
"""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.database.connection import DatabaseConnection
from hse_prom_prog.models.task import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskRepository:
    """Repository for task database operations.

    Provides CRUD operations for tasks table using raw SQL queries,
    consistent with existing DatabaseConnection.execute_query() pattern.

    Attributes:
        db: Database connection instance.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        """Initialize repository with database connection.

        Args:
            db: DatabaseConnection instance for executing queries.
        """
        self.db = db

    def create_task(
        self,
        query: str,
        celery_task_id: str | None = None,
        conversation_id: UUID | None = None,
    ) -> Task:
        """Create a new task with PENDING status.

        Args:
            query: User query to process.
            celery_task_id: Optional Celery task ID for tracking.
            conversation_id: Optional memory-layer conversation id.

        Returns:
            Created Task instance with generated task_id.

        Raises:
            SQLAlchemyError: If database operation fails.
        """
        sql = """
            INSERT INTO tasks (query, status, celery_task_id, conversation_id)
            VALUES (:query, :status, :celery_task_id, :conversation_id)
            RETURNING task_id, query, status, result, error, celery_task_id,
                      created_at, started_at, completed_at, workflow_state,
                      conversation_id
        """

        try:
            with self.db.get_session() as session:
                result = session.execute(
                    text(sql),
                    {
                        "query": query,
                        "status": TaskStatus.PENDING.value,
                        "celery_task_id": celery_task_id,
                        "conversation_id": (str(conversation_id) if conversation_id else None),
                    },
                )
                session.commit()
                row = result.fetchone()
                task = self._row_to_task(row)
                logger.info(f"[TaskRepository] Created task {task.task_id}")
                return task
        except SQLAlchemyError as e:
            logger.error(f"[TaskRepository] Failed to create task: {e}")
            raise

    def get_task(self, task_id: UUID) -> Task | None:
        """Retrieve task by ID.

        Args:
            task_id: Task UUID to retrieve.

        Returns:
            Task instance if found, None otherwise.

        Raises:
            SQLAlchemyError: If database operation fails.
        """
        sql = """
            SELECT task_id, query, status, result, error, celery_task_id,
                   created_at, started_at, completed_at, workflow_state,
                   conversation_id
            FROM tasks
            WHERE task_id = :task_id
        """

        try:
            with self.db.get_session() as session:
                result = session.execute(text(sql), {"task_id": str(task_id)})
                row = result.fetchone()
                if row:
                    task = self._row_to_task(row)
                    logger.debug(f"[TaskRepository] Retrieved task {task_id}")
                    return task
                logger.debug(f"[TaskRepository] Task {task_id} not found")
                return None
        except SQLAlchemyError as e:
            logger.error(f"[TaskRepository] Failed to get task {task_id}: {e}")
            raise

    def update_task_status(
        self,
        task_id: UUID,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        workflow_state: dict[str, Any] | None = None,
    ) -> None:
        """Update task status and optional result/error data.

        Automatically sets appropriate timestamps based on status:
        - PROCESSING: Sets started_at to current time
        - COMPLETED/FAILED: Sets completed_at to current time

        Args:
            task_id: Task UUID to update.
            status: New task status.
            result: Optional result data (for COMPLETED status).
            error: Optional error message (for FAILED status).
            workflow_state: Optional full workflow state for debugging.

        Raises:
            SQLAlchemyError: If database operation fails.
        """
        # Build dynamic SQL based on status and provided data
        updates = ["status = :status"]
        params: dict[str, Any] = {
            "task_id": str(task_id),
            "status": status.value,
        }

        # Set started_at when transitioning to PROCESSING
        if status == TaskStatus.PROCESSING:
            updates.append("started_at = :started_at")
            params["started_at"] = datetime.now()

        # Set completed_at when transitioning to terminal state
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            updates.append("completed_at = :completed_at")
            params["completed_at"] = datetime.now()

        # Update result if provided (usually for COMPLETED)
        if result is not None:
            updates.append("result = :result")
            params["result"] = json.dumps(result, default=str)

        # Update error if provided (usually for FAILED)
        if error is not None:
            updates.append("error = :error")
            params["error"] = error

        # Update workflow_state if provided
        if workflow_state is not None:
            updates.append("workflow_state = :workflow_state")
            params["workflow_state"] = json.dumps(workflow_state, default=str)

        sql = f"""
            UPDATE tasks
            SET {", ".join(updates)}
            WHERE task_id = :task_id
        """

        try:
            with self.db.get_session() as session:
                session.execute(text(sql), params)
                session.commit()
                logger.info(f"[TaskRepository] Updated task {task_id} to status {status.value}")
        except SQLAlchemyError as e:
            logger.error(f"[TaskRepository] Failed to update task {task_id}: {e}")
            raise

    def update_conversation_id(self, task_id: UUID, conversation_id: UUID) -> None:
        """Repoint a task at a different conversation.

        Used when the worker rotates a stale conversation mid-flight: the
        request was originally accepted against the old conversation, but
        actually executed in the context of a freshly created one. Without
        this the audit row in ``tasks`` keeps pointing at the closed
        conversation, which makes debugging "which session ran this query"
        misleading.
        """
        sql = "UPDATE tasks SET conversation_id = :conversation_id WHERE task_id = :task_id"
        try:
            with self.db.get_session() as session:
                session.execute(
                    text(sql),
                    {
                        "task_id": str(task_id),
                        "conversation_id": str(conversation_id),
                    },
                )
                session.commit()
                logger.info(
                    "[TaskRepository] Repointed task %s to conversation %s",
                    task_id,
                    conversation_id,
                )
        except SQLAlchemyError as e:
            logger.error(
                "[TaskRepository] Failed to repoint task %s to conversation %s: %s",
                task_id,
                conversation_id,
                e,
            )
            raise

    def _row_to_task(self, row: Any) -> Task:
        """Convert database row to Task instance.

        Args:
            row: Database row from query result.

        Returns:
            Task instance populated from row data.
        """
        return Task(
            task_id=row.task_id,
            query=row.query,
            status=TaskStatus(row.status),
            result=row.result,  # Already parsed from JSONB
            error=row.error,
            celery_task_id=row.celery_task_id,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            workflow_state=row.workflow_state,  # Already parsed from JSONB
            conversation_id=getattr(row, "conversation_id", None),
        )
