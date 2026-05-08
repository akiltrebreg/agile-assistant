"""Centralised logging configuration.

Used by every entrypoint (CLI, FastAPI app factory, Celery worker init)
to install the same root-logger config: stdout + a rotating file under
``/app/logs/{service}.log`` so Promtail can ship the lines to Loki.

Idempotent: a second call with the same service name is a no-op, so it's
safe to call from multiple places (e.g. Celery's ``worker_process_init``
fires per pool process and we don't want stacked handlers).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 50 MiB per file, keep 5 backups => ~250 MiB max per service. Conservative
# default that survives a typical day of INFO-level traffic without
# trimming, while keeping disk usage bounded across all services.
_FILE_MAX_BYTES = 50 * 1024 * 1024
_FILE_BACKUP_COUNT = 5

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Sentinel attribute on the root logger: lets us detect a previous
# setup_logging() call without creating a module-level mutable state
# (which breaks under multiprocessing fork in Celery).
_INSTALLED_ATTR = "_hse_logging_installed"


def setup_logging(
    service: str,
    *,
    level: str = "INFO",
    log_dir: str | Path = "/app/logs",
) -> None:
    """Install stdout + rotating-file handlers on the root logger.

    Args:
        service: Short service identifier used as the log file basename
            (e.g. ``"api"`` -> ``/app/logs/api.log``).
        level: Logging level name (``DEBUG``/``INFO``/``WARNING``/...).
        log_dir: Directory for the rotating file. Defaults to the
            container path that's bind-mounted from the host's ``./logs``.
            Created if it does not exist; if creation fails (read-only
            FS, permission denied) only the stdout handler is attached
            so the process still logs.

    Notes:
        Calling setup_logging twice with the same service is a no-op —
        we set a sentinel attribute on the root logger after the first
        successful install. Calling with a *different* service replaces
        the file handler (CLI scenario where main.py runs once per
        invocation; otherwise irrelevant).
    """
    root = logging.getLogger()

    # Skip if already configured for this service. This guards Celery's
    # worker_process_init signal which fires once per forked pool worker.
    if getattr(root, _INSTALLED_ATTR, None) == service:
        return

    # Replace any pre-existing handlers (basicConfig from a stray import,
    # uvicorn's default config, etc.) so we have a single source of truth.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(_LOG_FORMAT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # File handler is best-effort: if the mount is missing or the
    # directory isn't writable we fall back to stdout-only rather than
    # crash the service at startup.
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / f"{service}.log",
            maxBytes=_FILE_MAX_BYTES,
            backupCount=_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        # Use the stdout handler we just installed.
        root.warning(
            "Could not attach file handler at %s: %s. Logging to stdout only.",
            log_dir,
            exc,
        )

    setattr(root, _INSTALLED_ATTR, service)
