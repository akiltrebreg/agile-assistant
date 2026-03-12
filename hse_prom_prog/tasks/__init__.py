"""Celery tasks for async workflow execution."""

from hse_prom_prog.tasks.celery_app import celery_app
from hse_prom_prog.tasks.workflow_task import execute_workflow

__all__ = ["celery_app", "execute_workflow"]
