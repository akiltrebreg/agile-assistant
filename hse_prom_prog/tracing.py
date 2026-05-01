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

    Every method is a no-op so call sites don't need ``if`` guards. The
    real SDK returns ``None`` from these methods, so callers can chain
    safely against either implementation.
    """

    def update_current_observation(self, **_kwargs: Any) -> None:
        """No-op replacement for ``langfuse_context.update_current_observation``."""

    def update_current_trace(self, **_kwargs: Any) -> None:
        """No-op replacement for ``langfuse_context.update_current_trace``."""

    def get_current_trace_id(self) -> str | None:
        """Return ``None`` — there is no active trace when tracing is disabled."""
        return None

    def get_current_observation_id(self) -> str | None:
        """Return ``None`` — there is no active observation when tracing is disabled."""
        return None

    def flush(self) -> None:
        """No-op replacement for ``langfuse_context.flush``."""


def _noop_observe(*args: Any, **kwargs: Any) -> Any:
    """No-op replacement for the ``@observe`` decorator.

    Supports both ``@observe`` (bare) and ``@observe(name="x",
    as_type="generation")`` invocation forms by sniffing ``args``: if
    the first positional argument is a callable and no kwargs were
    given, the decorator was applied directly; otherwise the helper
    returns an inner decorator that simply returns the wrapped function.

    Args:
        *args: Either ``(fn,)`` for bare ``@observe`` or empty for the
            parenthesised form.
        **kwargs: Decorator options accepted by the real SDK; ignored.

    Returns:
        The original function (bare form) or an identity decorator
        (parenthesised form).
    """
    if args and callable(args[0]) and not kwargs:
        # Used as @observe without parentheses.
        return args[0]

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _decorator


def _initialize() -> tuple[Any, Callable[..., Any], Any]:
    """Build the ``(client, observe, langfuse_context)`` triple.

    Returns no-op stand-ins when ``settings.langfuse_enabled`` is False
    or the SDK is unavailable so the rest of the codebase can import
    these names unconditionally. When the SDK loads successfully, the
    Pydantic settings are mirrored into ``os.environ`` via
    ``setdefault`` so that explicit shell / docker env values still win.

    Returns:
        Tuple of ``(client, observe, langfuse_context)``:

        * ``client`` — ``Langfuse`` instance or ``None`` if disabled.
        * ``observe`` — real or no-op ``@observe`` decorator.
        * ``langfuse_context`` — real SDK module or :class:`_NoopContext`.
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


def make_langgraph_callback(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any | None:
    """Build a Langfuse ``CallbackHandler`` for a single LangGraph run.

    LangGraph's Pregel runtime executes each node in its own RunnableConfig
    scope and does not propagate Python contextvars across nodes. That is
    why ``@observe`` decorators on node methods (Supervisor, ResponseAgent,
    ResponseGuard, ...) created spans that never reached the active trace
    in Langfuse v2: every node opened a fresh root trace and the spans
    were silently dropped on flush.

    The Langchain/LangGraph callback handler is the supported integration:
    it hooks into the runtime via ``RunnableConfig.callbacks`` and emits
    one trace per ``graph.invoke``, with one span per node and per LLM
    call. It supersedes the imperative ``langfuse_client.trace(...)``
    pattern we used before.

    Returns ``None`` when tracing is disabled or the SDK / langchain
    integration is unavailable, so callers can pass the result to
    ``RunnableConfig.callbacks`` unconditionally (use ``[h] if h else []``).
    """
    if langfuse_client is None:
        return None

    try:
        # langfuse.callback ships inside the main langfuse v2 package;
        # no extra dependency required (the integration sits next to the
        # decorator API we already use).
        from langfuse.callback import CallbackHandler  # noqa: PLC0415
    except Exception as exc:
        logger.warning("[Tracing] Langfuse CallbackHandler unavailable: %s", exc)
        return None

    try:
        return CallbackHandler(
            public_key=settings.langfuse_public_key or None,
            secret_key=settings.langfuse_secret_key or None,
            host=settings.langfuse_host,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or [],
        )
    except Exception as exc:
        logger.warning("[Tracing] CallbackHandler init failed: %s", exc)
        return None


__all__ = [
    "langfuse_client",
    "langfuse_context",
    "make_langgraph_callback",
    "observe",
]
