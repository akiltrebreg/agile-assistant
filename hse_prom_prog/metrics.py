"""Prometheus metrics registry for the Agile AI Assistant (Phase 1).

This module is the single source of truth for all custom Prometheus
metrics used by the application. Defining the Counter / Histogram /
Gauge objects in one place avoids duplicate-registration errors in
``prometheus_client.REGISTRY`` and gives importers a stable handle.

Where each metric is populated:
  * ``PIPELINE_*`` and ``TASKS_*`` — Celery worker (workflow_task.py).
  * ``CELERY_*`` — Celery worker (celery_app.py signals).
  * ``CELERY_QUEUE_LENGTH`` — populated lazily on each /metrics scrape
    by ``QueueLengthCollector`` (api/app.py), which queries Redis LLEN.

Process model assumptions (Phase 1):
  * Celery worker runs with ``--pool=threads --concurrency=4`` — a
    single Python process, so the default in-memory registry is safe.
    If we ever switch to ``--pool=prefork``, configure
    ``PROMETHEUS_MULTIPROC_DIR`` and migrate to the multiprocess
    collector (see prometheus_client.multiprocess docs).
  * FastAPI runs under gunicorn with 2 UvicornWorker processes. Each
    worker has its own registry, so HTTP counters from
    prometheus-fastapi-instrumentator are split across workers and
    Prometheus may scrape either. Trends remain meaningful; absolute
    rates are undercounted by a factor of N. Acceptable for Phase 1;
    multiprocess mode would fix it cleanly.
"""

from prometheus_client import Counter, Gauge, Histogram

# Shared namespace and a short Fibonacci-like bucket scale chosen to
# match the SLOs documented in monitoring_plan.md (§3.1):
#   simple < 1s, sql 3-8s, hybrid up to 30s, timeout at 60s.
_NAMESPACE = "agile_assistant"
_PIPELINE_BUCKETS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0, 60.0)

# ── E2E Pipeline ──────────────────────────────────────────────
PIPELINE_DURATION = Histogram(
    "pipeline_duration_seconds",
    "End-to-end pipeline duration from request receipt to response generation",
    labelnames=("query_type",),
    namespace=_NAMESPACE,
    buckets=_PIPELINE_BUCKETS,
)

# Queue wait covers the gap between API enqueue (tasks.created_at) and
# the worker actually starting the task. Mostly Redis-broker overhead
# (10-100ms) until backlog grows; sub-second buckets dominate.
PIPELINE_QUEUE_WAIT = Histogram(
    "pipeline_queue_wait_seconds",
    "Time spent waiting in Celery queue before processing starts",
    namespace=_NAMESPACE,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

TASKS_TOTAL = Counter(
    "tasks_total",
    "Total number of completed tasks",
    labelnames=("status",),
    namespace=_NAMESPACE,
)

TASKS_IN_PROGRESS = Gauge(
    "tasks_in_progress",
    "Number of tasks currently being processed",
    namespace=_NAMESPACE,
)

# ── Celery ────────────────────────────────────────────────────
# Buckets cover background memory tasks (summarisation < 10s) up to
# the workflow hard limit (celery_task_time_limit = 600s).
CELERY_TASK_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Celery task execution duration",
    labelnames=("task_name",),
    namespace=_NAMESPACE,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

CELERY_TASKS_TOTAL = Counter(
    "celery_tasks_total",
    "Total Celery tasks by name and final status",
    labelnames=("task_name", "status"),
    namespace=_NAMESPACE,
)

CELERY_ACTIVE_TASKS = Gauge(
    "celery_active_tasks",
    "Currently executing Celery tasks",
    namespace=_NAMESPACE,
)

CELERY_QUEUE_LENGTH = Gauge(
    "celery_queue_length",
    "Number of tasks waiting in Redis queue",
    namespace=_NAMESPACE,
)
