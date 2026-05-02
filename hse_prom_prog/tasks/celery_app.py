"""Create and configure the Celery application.

Centralises broker, beat and signal wiring so worker processes receive a
fully observable task runtime — Prometheus metrics, JSON-only payloads,
beat schedules for periodic syncs and a dedicated ``judge`` queue.
"""

import logging
import os
import time

from celery import Celery
from celery.schedules import crontab
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    worker_process_init,
    worker_ready,
)
from prometheus_client import start_http_server

from hse_prom_prog.config import settings
from hse_prom_prog.metrics import (
    CELERY_ACTIVE_TASKS,
    CELERY_TASK_DURATION,
    CELERY_TASKS_TOTAL,
    initialize_label_combinations,
)

logger = logging.getLogger(__name__)

# Port the worker exposes for Prometheus scraping. The Celery worker
# has no HTTP server of its own, so we spin up a tiny one alongside
# task processing — same registry as celery signals + workflow_task,
# so pipeline metrics land here. Configurable via PROMETHEUS_PORT so
# multiple workers (default queue + judge) can co-exist on the same
# docker network: celery-worker uses 9100, celery-judge uses 9101.
_METRICS_PORT = int(os.environ.get("PROMETHEUS_PORT", "9100"))

# Per-task start time keyed by Celery task UUID. Threadsafe-enough for
# the threads pool (CPython dict assignment is atomic) and small —
# entries are cleared in task_postrun.
_task_starts: dict[str, float] = {}


def _short_name(task_name: str | None) -> str:
    """Strip the dotted module prefix off a task name.

    ``hse_prom_prog.tasks.execute_workflow`` -> ``execute_workflow``.
    Keeps Prometheus label cardinality readable.

    Args:
        task_name: Fully qualified Celery task name, or ``None``.

    Returns:
        Trailing component of ``task_name``, or ``"unknown"`` when missing.
    """
    if not task_name:
        return "unknown"
    return task_name.rsplit(".", 1)[-1]


def create_celery_app() -> Celery:
    """Build and configure the Celery application.

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
            # Phase 4: LLM-as-a-Judge. Listed explicitly (rather than
            # relying on autodiscover) because evaluate_response_async
            # ships in its own queue and we want the import to fail
            # loudly at worker boot if the file is missing.
            "hse_prom_prog.tasks.judge_task",
            # Phase 5: periodic data refresh (Celery Beat). Beat itself
            # runs in a separate ``celery-beat`` process and only
            # schedules — the actual sync_jira_data / sync_knowledge_base
            # invocations run on this worker.
            "hse_prom_prog.tasks.sync_tasks",
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
        # Phase 4 routing: judge tasks must land on the dedicated
        # ``judge`` queue, served by the celery-judge worker. Without
        # this entry an enqueue with queue="judge" would still work
        # (we pass the queue at apply_async-time), but the route makes
        # the topology explicit and resilient to a missing kwarg.
        task_routes={
            "evaluate_response": {"queue": "judge"},
        },
        # ── Beat schedule (Phase 5) ────────────────────────────
        # Beat schedules the *task* into Redis; the regular default-queue
        # worker picks it up. We do NOT route sync tasks to a dedicated
        # queue: at most one of each runs per slot, and the lock in
        # sync_tasks.py prevents overlap, so co-locating them with
        # workflow_task is fine. The CSV sync's TRUNCATE+COPY blocks
        # SQL-Agent reads for ~seconds (Postgres MVCC handles it
        # transparently) — acceptable at a 6h cadence.
        beat_schedule={
            "sync-jira-data-every-6h": {
                "task": "sync_jira_data",
                "schedule": crontab(hour="*/6", minute=0),
            },
            "sync-knowledge-base-daily": {
                "task": "sync_knowledge_base",
                # 03:00 UTC — off-peak relative to typical workday usage
                # of the assistant, so the brief Qdrant collection
                # rebuild lands when nobody is asking RAG questions.
                "schedule": crontab(hour=3, minute=0),
            },
        },
        # Beat persists its "last run" timestamp here. Default is
        # ``celerybeat-schedule`` in CWD, which lands inside the
        # container and is lost on restart. Putting it under /tmp
        # keeps it out of the source tree; for production with a
        # restart-resilient cadence, mount a volume here.
        beat_schedule_filename="/tmp/celerybeat-schedule",
    )

    logger.info(
        f"[Celery] App created with broker={settings.celery_broker}, "
        f"time_limit={settings.celery_task_time_limit}s"
    )
    return celery_app


# ── Prometheus metrics endpoint ───────────────────────────────
# The threads pool runs everything in the worker's main process, so
# ``worker_process_init`` fires once per worker. Using ``worker_ready``
# as a fallback because some Celery versions don't emit
# ``worker_process_init`` for the threads pool.
def _start_metrics_server(**_kwargs) -> None:
    """Start the Prometheus exporter HTTP server inside the worker.

    Idempotent — repeated calls log and bail out instead of crashing
    the worker (``start_http_server`` raises ``OSError`` on port reuse).

    Pre-touches every known counter label combination via
    ``initialize_label_combinations`` so Grafana renders explicit
    ``0`` instead of ``No data`` on guardrail/sanitizer panels — a
    healthy system may go a full shift without a single block, and
    ``No data`` is indistinguishable from a broken scrape.
    """
    try:
        start_http_server(_METRICS_PORT)
        logger.info("[Celery] Prometheus metrics server listening on :%d", _METRICS_PORT)
    except OSError as exc:
        # Already bound (e.g. worker_ready firing after worker_process_init).
        logger.debug("[Celery] Metrics server already running: %s", exc)

    initialize_label_combinations()


worker_process_init.connect(_start_metrics_server)
worker_ready.connect(_start_metrics_server)


# ── Celery signals → Prometheus ───────────────────────────────
@task_prerun.connect
def _on_task_prerun(task_id=None, task=None, **_kwargs) -> None:
    """Record task start time and bump the in-flight gauge."""
    if task_id is not None:
        _task_starts[task_id] = time.time()
    CELERY_ACTIVE_TASKS.inc()


@task_postrun.connect
def _on_task_postrun(task_id=None, task=None, state=None, **_kwargs) -> None:
    """Observe task duration and emit success counters when applicable."""
    CELERY_ACTIVE_TASKS.dec()
    started = _task_starts.pop(task_id, None) if task_id else None
    short = _short_name(getattr(task, "name", None))
    if started is not None:
        CELERY_TASK_DURATION.labels(task_name=short).observe(time.time() - started)
    # Success path only — failures and retries are counted by their
    # own signal handlers below to avoid double-counting.
    if state == "SUCCESS":
        CELERY_TASKS_TOTAL.labels(task_name=short, status="success").inc()


@task_failure.connect
def _on_task_failure(sender=None, **_kwargs) -> None:
    """Increment the per-task failure counter on terminal failures."""
    short = _short_name(getattr(sender, "name", None))
    CELERY_TASKS_TOTAL.labels(task_name=short, status="failure").inc()


@task_retry.connect
def _on_task_retry(sender=None, **_kwargs) -> None:
    """Increment the per-task retry counter when Celery re-queues a task."""
    short = _short_name(getattr(sender, "name", None))
    CELERY_TASKS_TOTAL.labels(task_name=short, status="retry").inc()


# Global Celery instance
celery_app = create_celery_app()
