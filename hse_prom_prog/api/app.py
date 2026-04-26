"""FastAPI application factory.

This module creates and configures the FastAPI application for async task processing.
"""

import logging

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_fastapi_instrumentator import Instrumentator

from hse_prom_prog.api.routers import conversations, tasks
from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)


class QueueLengthCollector:
    """Custom Prometheus collector that emits the Celery queue length.

    Querying Redis at scrape time (rather than on a timer) keeps the
    gauge fresh without a background loop and avoids stale data when
    the worker is idle. ``LLEN celery`` is O(1).
    """

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
        return self._client

    def collect(self):
        family = GaugeMetricFamily(
            "agile_assistant_celery_queue_length",
            "Number of tasks waiting in Redis queue (LLEN celery)",
        )
        try:
            length = self._get_client().llen("celery")
            family.add_metric([], float(length))
        except Exception as exc:
            logger.debug("[QueueLengthCollector] LLEN failed: %s", exc)
            # Reset the cached client so a transient Redis blip recovers
            # next scrape instead of pinning a dead connection.
            self._client = None
        yield family


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Configures:
    - API metadata (title, description, version)
    - CORS middleware for cross-origin requests
    - Task management router
    - Health check endpoint
    - Prometheus instrumentation (HTTP metrics + custom queue collector)

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="HSE Prom Prog API",
        description="Async task processing API for Jira workflow queries",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware: origins controlled via CORS_ORIGINS env var
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include task management router
    app.include_router(tasks.router)
    # Memory-layer conversations + transcripts
    app.include_router(conversations.router)

    # Health check endpoint
    @app.get("/health", tags=["health"])
    def health_check():
        """Health check endpoint for monitoring.

        Returns:
            Dictionary with health status.
        """
        return {"status": "healthy", "service": "hse-prom-prog-api"}

    # Prometheus: HTTP middleware + /metrics endpoint. Excluding the
    # endpoint itself avoids self-referential noise on every scrape.
    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
    )
    try:
        REGISTRY.register(QueueLengthCollector())
    except ValueError:
        # Already registered (gunicorn workers reuse the module on
        # reload; second registration would otherwise crash startup).
        logger.debug("QueueLengthCollector already registered")

    logger.info("FastAPI app created successfully")
    return app


# Global app instance
app = create_app()
