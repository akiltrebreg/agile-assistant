"""Expose Celery tasks for async workflow execution.

Re-exports the Celery app instance plus the workflow and memory tasks
that the API enqueues.
"""

from agile_assistant.tasks.celery_app import celery_app
from agile_assistant.tasks.memory_tasks import summarize_session, update_profile_async
from agile_assistant.tasks.workflow_task import execute_workflow

__all__ = [
    "celery_app",
    "execute_workflow",
    "summarize_session",
    "update_profile_async",
]
