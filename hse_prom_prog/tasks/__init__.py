"""Expose Celery tasks for async workflow execution.

Re-exports the Celery app instance plus the workflow and memory tasks
that the API enqueues.
"""

from hse_prom_prog.tasks.celery_app import celery_app
from hse_prom_prog.tasks.memory_tasks import summarize_session, update_profile_async
from hse_prom_prog.tasks.workflow_task import execute_workflow

__all__ = [
    "celery_app",
    "execute_workflow",
    "summarize_session",
    "update_profile_async",
]
