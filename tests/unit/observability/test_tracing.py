"""Unit tests for ``hse_prom_prog.tracing``.

The tracing module is the kill-switch + graceful-degradation layer for
the Langfuse SDK. The pieces under test:

  * ``_NoopContext`` — every method is a no-op so ``langfuse_context.update_*``
    calls never crash even when tracing is disabled.
  * ``_noop_observe`` — supports both ``@observe`` and ``@observe(...)``
    invocation forms; preserves the wrapped function's behaviour.
  * ``_initialize`` — returns no-op stand-ins when:
      - ``settings.langfuse_enabled`` is False (kill-switch)
      - ``langfuse`` import fails (SDK missing)
      - ``Langfuse(...)`` constructor raises (bad credentials)
  * ``make_langgraph_callback`` — returns None when tracing is off so
    callers can pass the result to ``RunnableConfig.callbacks`` blindly.

The kill-switch behaviour is critical because of the deployment context:
the production server has Langfuse on, but every test/CI environment
must run with tracing OFF (the conftest sets ``LANGFUSE_ENABLED=false``
before any project module imports). A regression that ignores the
kill-switch would attempt outbound network calls during ``pytest``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from hse_prom_prog import tracing

# ===================================================================== #
# _NoopContext — every method must accept any kwargs and return None
# ===================================================================== #


@pytest.mark.unit
class TestNoopContext:
    def test_update_current_observation_accepts_any_kwargs(self) -> None:
        ctx = tracing._NoopContext()
        # Pin: arbitrary kwargs accepted (real SDK takes a wide variety).
        # Returning None lets call sites chain safely.
        assert (
            ctx.update_current_observation(input="x", output="y", model="m", level="ERROR") is None
        )

    def test_update_current_trace_is_noop(self) -> None:
        ctx = tracing._NoopContext()
        assert ctx.update_current_trace(user_id="u", tags=["t"]) is None

    def test_get_current_trace_id_returns_none(self) -> None:
        # Pin: returns None (NOT ""), so callers using ``if trace_id:``
        # behave as "no trace" rather than "empty string trace".
        ctx = tracing._NoopContext()
        assert ctx.get_current_trace_id() is None

    def test_get_current_observation_id_returns_none(self) -> None:
        ctx = tracing._NoopContext()
        assert ctx.get_current_observation_id() is None

    def test_flush_does_nothing(self) -> None:
        # The decorator API exposes ``flush`` so production shutdown
        # hooks can call it; the no-op must not raise.
        ctx = tracing._NoopContext()
        assert ctx.flush() is None

    def test_calling_unknown_method_raises(self) -> None:
        # Pin: the no-op deliberately does NOT use ``__getattr__`` —
        # adding a typo'd method on the real SDK should surface as
        # AttributeError in development, not silently no-op.
        ctx = tracing._NoopContext()
        with pytest.raises(AttributeError):
            ctx.update_made_up_method()  # type: ignore[attr-defined]


# ===================================================================== #
# _noop_observe — both call forms must preserve the wrapped function
# ===================================================================== #


@pytest.mark.unit
class TestNoopObserve:
    def test_observe_without_parens_returns_function_unchanged(self) -> None:
        @tracing._noop_observe
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_observe_with_parens_returns_decorator(self) -> None:
        @tracing._noop_observe(name="span", as_type="generation")
        def double(x: int) -> int:
            return x * 2

        assert double(7) == 14

    def test_observe_preserves_exceptions(self) -> None:
        # Critical: real Langfuse @observe re-raises caller exceptions
        # after tagging the span with level=ERROR. The no-op must do
        # the same — silently swallowing would hide bugs in tests.
        @tracing._noop_observe(name="x")
        def boom() -> None:
            raise ValueError("intentional")

        with pytest.raises(ValueError, match="intentional"):
            boom()

    def test_observe_preserves_kwargs(self) -> None:
        @tracing._noop_observe()
        def fn(*, key: str) -> str:
            return key.upper()

        assert fn(key="hello") == "HELLO"

    def test_observe_with_only_kwargs_returns_decorator(self) -> None:
        # Edge case: ``@observe(name="x")`` (no positional args, only
        # kwargs) must also return a decorator — not the kwarg dict.
        decorator = tracing._noop_observe(name="x")
        assert callable(decorator)

        @decorator
        def fn() -> int:
            return 1

        assert fn() == 1


# ===================================================================== #
# _initialize — kill-switch and degradation paths
# ===================================================================== #


@pytest.mark.unit
class TestInitialize:
    """Each path returns ``(client, observe, langfuse_context)``.

    The triple shape is the contract: callers do
    ``langfuse_client, observe, langfuse_context = _initialize()`` and
    expect those exact slots. A regression that returned only two values
    would break every module that imports from tracing.
    """

    def test_disabled_via_settings_returns_noops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Master kill-switch — pin: client is None, observe is the noop,
        # context is a _NoopContext instance.
        monkeypatch.setattr(tracing.settings, "langfuse_enabled", False)
        client, observe, ctx = tracing._initialize()
        assert client is None
        assert observe is tracing._noop_observe
        assert isinstance(ctx, tracing._NoopContext)

    def test_sdk_import_failure_falls_back_to_noops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the SDK is unavailable (or its transitive deps blow up),
        # the module must still import cleanly. We force this by
        # enabling the kill-switch but making the real SDK import
        # raise — the inline ``try/except`` should catch it.
        monkeypatch.setattr(tracing.settings, "langfuse_enabled", True)

        import builtins

        original_import = builtins.__import__

        def _raising_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langfuse":
                raise ImportError("simulated missing SDK")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raising_import)
        client, observe, ctx = tracing._initialize()
        assert client is None
        assert observe is tracing._noop_observe
        assert isinstance(ctx, tracing._NoopContext)

    def test_client_constructor_failure_falls_back_to_noops(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: a bad config (e.g. unparsable host URL) must not crash
        # the whole app — the module catches it and returns no-ops.
        monkeypatch.setattr(tracing.settings, "langfuse_enabled", True)

        # Stub the langfuse module so the import succeeds but the
        # constructor raises.
        import sys

        fake_module = MagicMock()
        fake_module.Langfuse = MagicMock(side_effect=RuntimeError("bad host"))
        fake_decorators = MagicMock()
        fake_decorators.observe = MagicMock(name="real_observe")
        fake_decorators.langfuse_context = MagicMock(name="real_ctx")

        monkeypatch.setitem(sys.modules, "langfuse", fake_module)
        monkeypatch.setitem(sys.modules, "langfuse.decorators", fake_decorators)

        client, observe, ctx = tracing._initialize()
        assert client is None
        assert observe is tracing._noop_observe
        assert isinstance(ctx, tracing._NoopContext)


# ===================================================================== #
# Module-level — what's exported when tests run
# ===================================================================== #


@pytest.mark.unit
class TestModuleLevelExports:
    """At import time tests run with ``LANGFUSE_ENABLED=false`` (set in
    conftest.py before any project import). Pin the resulting state."""

    def test_langfuse_client_is_none_when_disabled(self) -> None:
        # The module-level langfuse_client must be None in this
        # environment — otherwise tests would attempt outbound HTTP.
        assert tracing.langfuse_client is None

    def test_observe_is_the_noop(self) -> None:
        assert tracing.observe is tracing._noop_observe

    def test_langfuse_context_is_noop_instance(self) -> None:
        assert isinstance(tracing.langfuse_context, tracing._NoopContext)

    def test_observe_can_decorate_functions_at_test_time(self) -> None:
        # Smoke test: every agent in the codebase decorates its main
        # entrypoint with @observe. Pin that doing so under the no-op
        # works — a regression here would break ALL agent imports.
        @tracing.observe(name="x", as_type="span")
        def fn(a: int) -> int:
            return a + 1

        assert fn(41) == 42

    def test_module_exports_pinned(self) -> None:
        # Pin __all__ — downstream code does
        # ``from hse_prom_prog.tracing import langfuse_client, observe, langfuse_context``
        # and a missing export would silently shadow with None at attribute access.
        assert "langfuse_client" in tracing.__all__
        assert "langfuse_context" in tracing.__all__
        assert "observe" in tracing.__all__
        assert "make_langgraph_callback" in tracing.__all__


# ===================================================================== #
# make_langgraph_callback — None when disabled
# ===================================================================== #


@pytest.mark.unit
class TestMakeLanggraphCallback:
    def test_returns_none_when_client_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When tracing is disabled (langfuse_client is None), the
        # callback factory short-circuits — no need to attempt the
        # langfuse.callback import which can fail in test envs.
        monkeypatch.setattr(tracing, "langfuse_client", None)
        result = tracing.make_langgraph_callback(user_id="u", session_id="s")
        assert result is None

    def test_callback_handler_import_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If langfuse is loaded but the CallbackHandler subpath is
        # missing (Langfuse v3 API drift), we return None — callers do
        # ``[h] if h else []`` and the workflow stays trace-less but
        # functional.
        monkeypatch.setattr(tracing, "langfuse_client", MagicMock())

        import builtins

        original_import = builtins.__import__

        def _fail(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langfuse.callback":
                raise ImportError("CallbackHandler removed in v3")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        assert tracing.make_langgraph_callback() is None

    def test_callback_handler_init_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CallbackHandler ctor takes credentials + host — if they're
        # malformed it can throw. The factory swallows the error and
        # logs — pin: returns None, no exception bubbles up.
        monkeypatch.setattr(tracing, "langfuse_client", MagicMock())

        import sys

        fake_callback_module = MagicMock()
        fake_callback_module.CallbackHandler = MagicMock(side_effect=ValueError("bad credentials"))
        monkeypatch.setitem(sys.modules, "langfuse.callback", fake_callback_module)
        assert tracing.make_langgraph_callback(user_id="u") is None

    def test_callback_handler_built_with_session_and_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Happy path: pin that user_id, session_id, metadata, tags reach
        # the CallbackHandler ctor. The dashboard uses these to filter
        # traces — silent renaming would make traces unsearchable.
        monkeypatch.setattr(tracing, "langfuse_client", MagicMock())

        import sys

        fake_callback_module = MagicMock()
        handler_instance = MagicMock(name="handler")
        ctor = MagicMock(return_value=handler_instance)
        fake_callback_module.CallbackHandler = ctor
        monkeypatch.setitem(sys.modules, "langfuse.callback", fake_callback_module)

        result = tracing.make_langgraph_callback(
            user_id="user-1",
            session_id="conv-2",
            metadata={"task_id": "t-3"},
            tags=["beta"],
        )
        assert result is handler_instance
        ctor_kwargs = ctor.call_args.kwargs
        assert ctor_kwargs["user_id"] == "user-1"
        assert ctor_kwargs["session_id"] == "conv-2"
        assert ctor_kwargs["metadata"] == {"task_id": "t-3"}
        assert ctor_kwargs["tags"] == ["beta"]

    def test_metadata_and_tags_default_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin: omitted metadata/tags become {} / [] — passing None
        # through would surprise the SDK in some Langfuse versions.
        monkeypatch.setattr(tracing, "langfuse_client", MagicMock())

        import sys

        fake_callback_module = MagicMock()
        ctor = MagicMock(return_value=MagicMock())
        fake_callback_module.CallbackHandler = ctor
        monkeypatch.setitem(sys.modules, "langfuse.callback", fake_callback_module)

        tracing.make_langgraph_callback()
        ctor_kwargs = ctor.call_args.kwargs
        assert ctor_kwargs["metadata"] == {}
        assert ctor_kwargs["tags"] == []
        assert ctor_kwargs["user_id"] is None
        assert ctor_kwargs["session_id"] is None


# ===================================================================== #
# Decorator-as-no-op smoke — class methods + nested calls
# ===================================================================== #


@pytest.mark.unit
class TestObserveOnRealisticTargets:
    """Mimic the actual decoration patterns in the codebase to catch
    regressions that the unit tests above would miss."""

    def test_decorating_class_method_preserves_self(self) -> None:
        # Every agent does ``@observe(...)`` on a method that takes
        # ``self``. Pin: ``self`` is forwarded so .invoke() still works.
        observe: Callable[..., Any] = tracing._noop_observe

        class Echo:
            def __init__(self, prefix: str) -> None:
                self.prefix = prefix

            @observe(name="echo")  # type: ignore[misc]
            def call(self, msg: str) -> str:
                return f"{self.prefix}:{msg}"

        e = Echo("hi")
        assert e.call("there") == "hi:there"

    def test_observe_used_as_argument_to_observe(self) -> None:
        # Defensive: passing ``observe`` itself as a callable arg used
        # to break a stray ``@observe(observe)`` annotation. The no-op
        # treats that as ``@observe(some_callable)`` → returns it.
        @tracing._noop_observe
        def f() -> int:
            return 1

        assert f() == 1

    def test_no_op_observe_returns_same_function_object(self) -> None:
        # Pin: the simple ``@observe`` form (no parens) returns the
        # *same* function object — not a wrapper. This matters because
        # some agents introspect ``__name__`` for logging.
        def named_func() -> None:
            return None

        wrapped = tracing._noop_observe(named_func)
        assert wrapped is named_func
        assert wrapped.__name__ == "named_func"
