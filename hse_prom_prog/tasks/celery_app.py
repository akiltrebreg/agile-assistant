"""Celery application factory.

This module creates and configures the Celery application for async task processing.
"""

import logging

from celery import Celery

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """Create and configure Celery application.

    Configures Celery with:
    - Redis as message broker
    - No result backend (PostgreSQL is single source of truth)
    - Task time limits from settings
    - JSON serialization for messages
    - Task acknowledgment settings for reliability

    Returns:
        Configured Celery instance.
    """
    celery_app = Celery(
        "hse_prom_prog",
        broker=settings.celery_broker,
        backend=None,  # No result backend - PostgreSQL stores all state
        include=[
            "hse_prom_prog.tasks.workflow_task",
            "hse_prom_prog.tasks.memory_tasks",
        ],
    )

    # Configure Celery settings
    celery_app.conf.update(
        # Task tracking
        task_track_started=settings.celery_task_track_started,
        # Time limits
        task_time_limit=settings.celery_task_time_limit,
        task_soft_time_limit=settings.celery_task_soft_time_limit,
        # Reliability settings
        task_acks_late=True,  # Only acknowledge after task completes
        task_reject_on_worker_lost=True,  # Requeue if worker crashes
        worker_prefetch_multiplier=1,  # Don't prefetch tasks (fair distribution)
        worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
        # Serialization
        task_serializer="json",
        accept_content=["json"],
        # Timezone
        timezone="UTC",
        enable_utc=True,
        # Result backend disabled
        result_backend=None,
        task_ignore_result=True,
    )

    logger.info(
        f"[Celery] App created with broker={settings.celery_broker}, "
        f"time_limit={settings.celery_task_time_limit}s"
    )
    return celery_app


# Global Celery instance
celery_app = create_celery_app()
