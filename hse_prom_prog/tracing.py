"""Langfuse tracing — single source of truth for the SDK.

All modules import ``observe`` / ``langfuse_context`` / ``langfuse_client``
from here, never from ``langfuse`` directly. This keeps the swap radius
small if we ever change tracing backends and gives us one place to
implement graceful degradation when the SDK is missing or disabled.

Behaviour:

* ``settings.langfuse_enabled`` is the master kill-switch. When False
  (or when the SDK fails to import) the module exposes a no-op
  ``observe`` decorator and a no-op ``langfuse_context`` so existing
  ``@observe`` annotations and ``langfuse_context.update_*`` calls
  stay valid but produce no side effects.
* The decorator-style SDK reads its config from environment variables.
  We mirror Pydantic settings into ``os.environ`` via ``setdefault`` so
  that explicit shell/docker env values still win.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)


class _NoopContext:
    """Drop-in stand-in for ``langfuse.decorators.langfuse_context``.

    Every method is a no-op so call sites don't need ``if`` guards.
    Returning ``None`` matches what the real SDK returns from these
    methods, so callers can chain safely.
    """

    def update_current_observation(self, **_kwargs: Any) -> None:
        return None

    def update_current_trace(self, **_kwargs: Any) -> None:
        return None

    def get_current_trace_id(self) -> str | None:
        return None

    def get_current_observation_id(self) -> str | None:
        return None

    def flush(self) -> None:
        return None


def _noop_observe(*args: Any, **kwargs: Any) -> Any:
    """No-op replacement for ``@observe``.

    Supports both ``@observe`` and ``@observe(name="x", as_type="generation")``
    invocation forms.
    """
    if args and callable(args[0]) and not kwargs:
        # Used as @observe without parentheses.
        return args[0]

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _decorator


def _initialize() -> tuple[Any, Callable[..., Any], Any]:
    """Build the (client, observe, langfuse_context) triple.

    Returns no-op stand-ins if the SDK is unavailable or disabled so the
    rest of the codebase can import these names unconditionally.
    """
    if not settings.langfuse_enabled:
        logger.info("[Tracing] Langfuse disabled via settings — using no-op decorators")
        return None, _noop_observe, _NoopContext()

    try:
        from langfuse import Langfuse  # noqa: PLC0415
        from langfuse.decorators import langfuse_context as _ctx  # noqa: PLC0415
        from langfuse.decorators import observe as _observe  # noqa: PLC0415
    except Exception as exc:
        logger.warning("[Tracing] Langfuse SDK import failed (%s) — disabling tracing", exc)
        return None, _noop_observe, _NoopContext()

    # Mirror Pydantic settings into the env where the decorator SDK looks
    # them up. ``setdefault`` keeps explicit shell / docker values winning.
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if settings.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key or None,
            secret_key=settings.langfuse_secret_key or None,
            host=settings.langfuse_host,
            enabled=settings.langfuse_enabled,
        )
    except Exception as exc:
        logger.warning("[Tracing] Langfuse client init failed (%s) — disabling tracing", exc)
        return None, _noop_observe, _NoopContext()

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info(
            "[Tracing] Langfuse credentials not configured — SDK loaded but spans "
            "will not be sent until LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set"
        )
    else:
        logger.info("[Tracing] Langfuse initialised (host=%s)", settings.langfuse_host)
    return client, _observe, _ctx


langfuse_client, observe, langfuse_context = _initialize()


__all__ = ["langfuse_client", "langfuse_context", "observe"]
