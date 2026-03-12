"""Celery task for executing AgileWorkflow.

This module wraps the existing LangGraph workflow in a Celery task,
managing status updates in PostgreSQL throughout execution.
Includes retry logic with exponential backoff for transient failures.
"""

import logging
from uuid import UUID

from celery import Task as CeleryTask
from celery.exceptions import SoftTimeLimitExceeded

from hse_prom_prog.database.connection import get_database
from hse_prom_prog.database.task_repository import TaskRepository
from hse_prom_prog.graph.workflow import AgileWorkflow
from hse_prom_prog.llm.client import get_llm_client
from hse_prom_prog.models.task import TaskStatus
from hse_prom_prog.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


class WorkflowTask(CeleryTask):
    """Custom Celery task class with lifecycle hooks."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails after all retries exhausted."""
        logger.error(f"[Celery] Task {task_id} failed permanently: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is being retried."""
        logger.warning(f"[Celery] Task {task_id} retrying due to: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"[Celery] Task {task_id} completed successfully")


@celery_app.task(
    bind=True,
    base=WorkflowTask,
    name="hse_prom_prog.tasks.execute_workflow",
    # Retry on transient errors (network, DB, vLLM timeouts)
    autoretry_for=(ConnectionError, OSError, TimeoutError),
    retry_backoff=True,  # Exponential backoff: 1s, 2s, 4s, ...
    retry_backoff_max=60,  # Max 60s between retries
    retry_jitter=True,  # Add randomness to prevent thundering herd
    max_retries=3,  # Up to 3 retries for transient failures
)
def execute_workflow(self, task_id: str, query: str) -> None:
    """Execute AgileWorkflow for a given task.

    Status flow: PENDING -> PROCESSING -> COMPLETED / FAILED.

    Retries automatically on transient network/DB/timeout errors
    with exponential backoff. Non-retryable errors (e.g. bad query)
    go directly to FAILED.

    Args:
        self: Celery task instance (injected by bind=True).
        task_id: Application task UUID (from tasks table).
        query: User query to process.
    """
    task_uuid = UUID(task_id)
    db = None

    try:
        db = get_database()
        repo = TaskRepository(db)

        # Set PROCESSING only on first attempt, not on retries
        if self.request.retries == 0:
            logger.info(f"[WorkflowTask {task_id}] Starting workflow execution")
            repo.update_task_status(task_uuid, TaskStatus.PROCESSING)
        else:
            logger.info(f"[WorkflowTask {task_id}] Retry {self.request.retries}/{self.max_retries}")

        # Execute existing workflow (unchanged business logic)
        llm_client = get_llm_client()
        workflow = AgileWorkflow(llm_client, db_connection=db)

        logger.info(f"[WorkflowTask {task_id}] Running AgileWorkflow for: {query[:50]}...")
        result = workflow.run(query)

        # Update to COMPLETED
        final_response = result.get("final_response", "No response generated")
        issue_key = result.get("issue_key", "UNKNOWN")

        logger.info(f"[WorkflowTask {task_id}] Workflow completed successfully")
        repo.update_task_status(
            task_uuid,
            TaskStatus.COMPLETED,
            result={
                "final_response": final_response,
                "issue_key": issue_key,
            },
            workflow_state=result,
        )

    except SoftTimeLimitExceeded:
        # Hard timeout -- no retries, fail immediately
        error_msg = "Task exceeded soft time limit"
        logger.error(f"[WorkflowTask {task_id}] {error_msg}")
        _update_task_failed(db, task_uuid, task_id, error_msg)
        raise

    except (ConnectionError, OSError, TimeoutError):
        # Transient errors -- autoretry handles these.
        # On the last retry mark as FAILED in DB before re-raise.
        if self.request.retries >= self.max_retries:
            error_msg = f"Failed after {self.max_retries} retries (transient error)"
            _update_task_failed(db, task_uuid, task_id, error_msg)
        raise

    except Exception as e:
        # Non-retryable errors -- mark FAILED immediately
        error_msg = f"{type(e).__name__}: {e!s}"
        logger.error(f"[WorkflowTask {task_id}] Workflow failed: {error_msg}", exc_info=True)
        _update_task_failed(db, task_uuid, task_id, error_msg)
        raise

    finally:
        if db is not None:
            db.close()


def _update_task_failed(db, task_uuid: UUID, task_id: str, error_msg: str) -> None:
    """Update task status to FAILED in database.

    Args:
        db: Database connection (may be None).
        task_uuid: Task UUID.
        task_id: Task ID string (for logging).
        error_msg: Error message to store.
    """
    try:
        if db is None:
            db = get_database()
        repo = TaskRepository(db)
        repo.update_task_status(task_uuid, TaskStatus.FAILED, error=error_msg)
    except Exception as update_error:
        logger.error(f"[WorkflowTask {task_id}] Failed to update status to FAILED: {update_error}")
