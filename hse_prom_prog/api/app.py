"""FastAPI application factory.

This module creates and configures the FastAPI application for async task processing.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hse_prom_prog.api.routers import tasks
from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Configures:
    - API metadata (title, description, version)
    - CORS middleware for cross-origin requests
    - Task management router
    - Health check endpoint

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

    # Health check endpoint
    @app.get("/health", tags=["health"])
    def health_check():
        """Health check endpoint for monitoring.

        Returns:
            Dictionary with health status.
        """
        return {"status": "healthy", "service": "hse-prom-prog-api"}

    logger.info("FastAPI app created successfully")
    return app


# Global app instance
app = create_app()
