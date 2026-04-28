"""Periodic data sync tasks scheduled by Celery Beat.

Two pipelines, both reusing existing one-shot CLIs so there is no
divergence between manual ``python -m`` runs and the scheduled job:

- ``sync_jira_data``     — reloads Jira CSVs (S3 → PostgreSQL) every 6h.
- ``sync_knowledge_base`` — re-ingests RAG documents (S3 → Qdrant) daily.

Why Celery Beat (and not k8s CronJob): the workers already export
Prometheus metrics, ship Langfuse traces, and have retry/timeout policies
configured — re-running this pipeline as a Celery task means the
existing observability stack instruments it for free. A k8s CronJob
would be an ephemeral pod that needs a separate scrape strategy
(push gateway) and its own retry plumbing.

Concurrency: each task acquires a Redis SETNX lock keyed on the
``source`` label, with a TTL slightly longer than the worst-case
runtime. If a previous run is still going (or a worker died holding
the lock), the task no-ops and logs — Beat will fire again on the
next slot. The Postgres TRUNCATE+COPY inside ``load_csv`` is itself
atomic per table (AccessExclusiveLock on TRUNCATE, COPY in same
transaction), so concurrent SQL Agent reads see the OLD snapshot
until commit, then the NEW one — no torn reads, no app-level
coordination needed.
"""

from __future__ import annotations

import logging
import time

import psycopg2
import redis
from celery import shared_task

from hse_prom_prog.config import settings
from hse_prom_prog.database import load_csv
from hse_prom_prog.metrics import (
    DATA_SYNC_DURATION,
    DATA_SYNC_ROWS,
    DATA_SYNC_TIMESTAMP,
    DATA_SYNC_TOTAL,
)
from hse_prom_prog.rag.ingest import run_ingestion

logger = logging.getLogger(__name__)

# Lock TTLs: chosen as ~2x the realistic runtime so a crashed worker's
# lock expires before the next Beat tick fires the same task. Don't
# raise these without also raising the Beat interval — otherwise
# back-to-back ticks could overlap.
_JIRA_LOCK_TTL_SECONDS = 1200  # CSV: ~5 min worst case
_KB_LOCK_TTL_SECONDS = 2 * 60 * 60  # KB:  ~30-60 min worst case

_LOCK_KEY_PREFIX = "agile:sync:lock:"


def _redis_client() -> redis.Redis:
    """Return a Redis client pointed at the same broker the workers use."""
    return redis.Redis.from_url(settings.celery_broker, decode_responses=True)


def _try_acquire(client: redis.Redis, source: str, ttl: int) -> bool:
    """SET NX + EX — returns True if we got the lock, False if someone else holds it."""
    return bool(client.set(name=_LOCK_KEY_PREFIX + source, value="1", nx=True, ex=ttl))


def _release(client: redis.Redis, source: str) -> None:
    try:
        client.delete(_LOCK_KEY_PREFIX + source)
    except redis.RedisError:
        # Lock TTL will reap it; no action needed.
        logger.warning("[sync] Failed to release lock for %s (will TTL out)", source, exc_info=True)


def _record_outcome(source: str, status: str, started: float, rows: int | None = None) -> None:
    """Emit Prometheus metrics for one sync attempt."""
    DATA_SYNC_DURATION.labels(source=source).observe(time.time() - started)
    DATA_SYNC_TOTAL.labels(source=source, status=status).inc()
    if status == "success":
        DATA_SYNC_TIMESTAMP.labels(source=source).set(time.time())
        if rows is not None:
            DATA_SYNC_ROWS.labels(source=source).set(rows)


@shared_task(
    name="sync_jira_data",
    bind=True,
    # Single retry only — Beat will fire again in 6h. Retrying many
    # times in a row would just spam vLLM-adjacent S3 hits without
    # fixing the underlying outage.
    autoretry_for=(Exception,),
    max_retries=1,
    default_retry_delay=300,
    retry_backoff=False,
)
def sync_jira_data(self) -> dict[str, int | str]:
    """Reload Jira CSVs from S3 into PostgreSQL.

    Returns a small dict so Flower / Langfuse trace shows actual numbers
    instead of just ``None``. Result backend is disabled in celery_app,
    so the dict is observability-only — workflow_task does not consume it.
    """
    source = "jira_csv"
    client = _redis_client()
    started = time.time()

    if not _try_acquire(client, source, _JIRA_LOCK_TTL_SECONDS):
        logger.warning("[sync] %s: another run is in progress, skipping", source)
        return {"source": source, "status": "skipped"}

    try:
        logger.info("[sync] %s: starting", source)
        # load_csv.main() handles S3 download, TRUNCATE, COPY, COUNT.
        # It exits the process on misconfig — so wrap in try and treat
        # SystemExit as failure rather than swallowing it as success.
        try:
            load_csv.main()
        except SystemExit as exc:
            raise RuntimeError(f"load_csv aborted: code={exc.code}") from exc

        # Re-query row counts for the metric (load_csv prints them but
        # doesn't return; cheaper than threading a return value through).
        rows_total = 0
        with psycopg2.connect(settings.database_url) as conn, conn.cursor() as cur:
            for table in ("report_agile_dashboard", "report_agile_dashboard_metrics"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                rows_total += cur.fetchone()[0]

        _record_outcome(source, "success", started, rows=rows_total)
        logger.info("[sync] %s: ok (%d rows total)", source, rows_total)
        return {"source": source, "status": "success", "rows": rows_total}

    except Exception:
        _record_outcome(source, "failure", started)
        logger.exception("[sync] %s: failed", source)
        raise
    finally:
        _release(client, source)


@shared_task(
    name="sync_knowledge_base",
    bind=True,
    autoretry_for=(Exception,),
    max_retries=1,
    default_retry_delay=600,
    retry_backoff=False,
)
def sync_knowledge_base(self) -> dict[str, int | str]:
    """Re-ingest documents from S3 into Qdrant.

    ``run_ingestion`` is destructive — it drops and recreates the
    Qdrant collection. There is a brief window (seconds) where the
    collection has a partial document set; RAG fallbacks (dense → no
    results → vector store warning) handle this gracefully and the
    next /metrics scrape will reflect the new chunk count.
    """
    source = "knowledge_base"
    client = _redis_client()
    started = time.time()

    if not _try_acquire(client, source, _KB_LOCK_TTL_SECONDS):
        logger.warning("[sync] %s: another run is in progress, skipping", source)
        return {"source": source, "status": "skipped"}

    try:
        logger.info("[sync] %s: starting", source)
        chunks = run_ingestion()
        _record_outcome(source, "success", started, rows=chunks)
        logger.info("[sync] %s: ok (%d chunks)", source, chunks)
        return {"source": source, "status": "success", "rows": chunks}

    except Exception:
        _record_outcome(source, "failure", started)
        logger.exception("[sync] %s: failed", source)
        raise
    finally:
        _release(client, source)
