"""Unit tests for ``hse_prom_prog.llm.client``.

The LLMClient is a thin wrapper around LangChain's ``ChatOpenAI``. We test:

  * Construction reads from ``settings`` defaults; explicit kwargs win.
  * ``invoke`` returns ``response.content`` (the AIMessage shape) without
    coercing it through ``str()`` first — that would corrupt JSON outputs.
  * Errors from the underlying client propagate (no silent fallback).
  * ``response_format`` and per-call ``max_tokens`` reach the bind layer
    only when set — bind() with empty kwargs would create a useless
    wrapper on every hot-path call.
  * ``_extract_usage`` reads both new (``usage_metadata``) and legacy
    (``response_metadata.token_usage``) shapes.

We do NOT spin up httpx or a fake vLLM. ``ChatOpenAI`` is patched at
the module's import site so the constructor sees a MagicMock factory.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hse_prom_prog.llm import client as llm_client

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_chat_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``ChatOpenAI`` at the import site so LLMClient gets a mock.

    Returns the *factory* mock so tests can inspect constructor kwargs
    OR reach into the per-instance ``invoke`` mock via ``return_value``.
    """
    factory = MagicMock()
    instance = MagicMock()
    response = MagicMock()
    response.content = "default response"
    response.usage_metadata = None
    response.response_metadata = {}
    instance.invoke.return_value = response
    instance.bind.return_value = instance  # bind() returns a chain-able client
    factory.return_value = instance
    monkeypatch.setattr(llm_client, "ChatOpenAI", factory)
    return factory


# ===================================================================== #
# Construction
# ===================================================================== #


@pytest.mark.unit
class TestLLMClientConstruction:
    def test_defaults_read_from_settings(self, fake_chat_openai: MagicMock) -> None:
        client = llm_client.LLMClient()
        # Settings come from config.py; pin only that they were USED.
        from hse_prom_prog.config import settings

        assert client.base_url == settings.vllm_base_url
        assert client.model == settings.vllm_model
        assert client.temperature == settings.vllm_temperature
        assert client.max_tokens == settings.vllm_max_tokens

    def test_explicit_kwargs_override_settings(self, fake_chat_openai: MagicMock) -> None:
        client = llm_client.LLMClient(
            base_url="http://localhost:9999/v1",
            model="overridden-model",
            api_key="custom-key",
            temperature=0.5,
            max_tokens=128,
        )
        assert client.base_url == "http://localhost:9999/v1"
        assert client.model == "overridden-model"
        assert client.api_key == "custom-key"
        assert client.temperature == 0.5
        assert client.max_tokens == 128
        # Pin: ChatOpenAI received the same kwargs (they're not silently dropped).
        ctor_kwargs = fake_chat_openai.call_args.kwargs
        assert ctor_kwargs["base_url"] == "http://localhost:9999/v1"
        assert ctor_kwargs["model"] == "overridden-model"
        assert ctor_kwargs["temperature"] == 0.5

    def test_repetition_penalty_lands_in_extra_body(self, fake_chat_openai: MagicMock) -> None:
        # vLLM accepts non-OpenAI knobs only via ``extra_body``. If this
        # ever moves to a top-level kwarg the SDK silently ignores it,
        # so the field-name + dict-shape combo is worth pinning.
        llm_client.LLMClient()
        ctor_kwargs = fake_chat_openai.call_args.kwargs
        assert "extra_body" in ctor_kwargs
        assert "repetition_penalty" in ctor_kwargs["extra_body"]

    def test_get_llm_client_factory_returns_instance(self, fake_chat_openai: MagicMock) -> None:
        # The factory exists for places that want a stub-friendly handle
        # without instantiating LLMClient directly. Pin: it forwards kwargs.
        client = llm_client.get_llm_client(model="x")
        assert isinstance(client, llm_client.LLMClient)
        assert client.model == "x"


# ===================================================================== #
# invoke() — happy path
# ===================================================================== #


@pytest.mark.unit
class TestInvokeHappyPath:
    def test_returns_response_content(self, fake_chat_openai: MagicMock) -> None:
        client = llm_client.LLMClient()
        # Configure the AIMessage content explicitly.
        msg = MagicMock()
        msg.content = "hello world"
        msg.usage_metadata = None
        msg.response_metadata = {}
        fake_chat_openai.return_value.invoke.return_value = msg

        out = client.invoke("привет")
        assert out == "hello world"

    def test_passes_prompt_through_unchanged(self, fake_chat_openai: MagicMock) -> None:
        client = llm_client.LLMClient()
        client.invoke("какая velocity у cthulhu?")
        # Pin: prompt is forwarded verbatim. A regression that prepended
        # a system message here would break SQL Agent's external prompt
        # construction (it builds its own system messages).
        fake_chat_openai.return_value.invoke.assert_called_with("какая velocity у cthulhu?")

    def test_falls_back_to_str_when_response_lacks_content_attr(
        self, fake_chat_openai: MagicMock
    ) -> None:
        # Defensive: some gateways return a bare string. Pin the fallback
        # so a future refactor doesn't produce ``"<MagicMock ...>"`` output.
        plain = "raw string response"
        fake_chat_openai.return_value.invoke.return_value = plain
        client = llm_client.LLMClient()
        assert client.invoke("q") == "raw string response"


# ===================================================================== #
# invoke() — bind() optimisation
# ===================================================================== #


@pytest.mark.unit
class TestInvokeBindOptimisation:
    """Bind is called only when there are extra knobs. ``bind()`` allocates
    a new RunnableBinding object — calling it with no kwargs every turn
    is wasteful and was a real perf bug fixed in this codebase."""

    def test_no_bind_call_when_no_overrides(self, fake_chat_openai: MagicMock) -> None:
        instance = fake_chat_openai.return_value
        client = llm_client.LLMClient()
        client.invoke("plain prompt")
        instance.bind.assert_not_called()

    def test_bind_called_with_response_format_only(self, fake_chat_openai: MagicMock) -> None:
        instance = fake_chat_openai.return_value
        client = llm_client.LLMClient()
        schema = {"type": "json_schema", "json_schema": {"name": "x"}}
        client.invoke("p", response_format=schema)
        instance.bind.assert_called_once_with(response_format=schema)

    def test_bind_called_with_both_overrides(self, fake_chat_openai: MagicMock) -> None:
        instance = fake_chat_openai.return_value
        client = llm_client.LLMClient()
        schema = {"type": "json_schema"}
        client.invoke("p", response_format=schema, max_tokens=42)
        bind_kwargs = instance.bind.call_args.kwargs
        assert bind_kwargs["response_format"] == schema
        assert bind_kwargs["max_tokens"] == 42

    def test_max_tokens_alone_triggers_bind(self, fake_chat_openai: MagicMock) -> None:
        # Per-call max_tokens override is used by short structured-output
        # classifiers — pin that it actually reaches the bind layer (and
        # not just the metadata).
        instance = fake_chat_openai.return_value
        client = llm_client.LLMClient()
        client.invoke("p", max_tokens=64)
        instance.bind.assert_called_once_with(max_tokens=64)


# ===================================================================== #
# invoke() — error path
# ===================================================================== #


@pytest.mark.unit
class TestInvokeErrors:
    def test_underlying_exception_propagates(self, fake_chat_openai: MagicMock) -> None:
        # The contract: LLMClient does NOT swallow errors. Callers (agents)
        # rely on this to feed Validator/Response a precise note instead
        # of an empty string that would falsely look like "no answer".
        instance = fake_chat_openai.return_value
        instance.invoke.side_effect = TimeoutError("vllm down")
        client = llm_client.LLMClient()
        with pytest.raises(TimeoutError, match="vllm down"):
            client.invoke("p")

    def test_arbitrary_exception_class_preserved(self, fake_chat_openai: MagicMock) -> None:
        # Some agents (Response Agent) inspect ``type(e).__name__`` for
        # the ``timeout`` heuristic. A regression that wrapped errors in
        # a generic ``RuntimeError`` would defeat that classification.
        class CustomLLMError(Exception):
            pass

        instance = fake_chat_openai.return_value
        instance.invoke.side_effect = CustomLLMError("oh no")
        client = llm_client.LLMClient()
        with pytest.raises(CustomLLMError):
            client.invoke("p")

    def test_invalid_json_passes_through_unparsed(self, fake_chat_openai: MagicMock) -> None:
        # LLMClient is JSON-agnostic — parsing is the caller's job.
        # Pin: even garbled-JSON output reaches the caller verbatim so
        # Supervisor's repair-and-retry path can run.
        msg = MagicMock()
        msg.content = '{"intent": "task", "entities": {bad json'
        msg.usage_metadata = None
        msg.response_metadata = {}
        fake_chat_openai.return_value.invoke.return_value = msg
        client = llm_client.LLMClient()
        out = client.invoke("p")
        assert out == '{"intent": "task", "entities": {bad json'


# ===================================================================== #
# _extract_usage — both formats
# ===================================================================== #


@pytest.mark.unit
class TestExtractUsage:
    def test_modern_usage_metadata_wins(self) -> None:
        msg = MagicMock()
        msg.usage_metadata = {
            "input_tokens": 12,
            "output_tokens": 34,
            "total_tokens": 46,
        }
        # Even if legacy is also present, modern field wins.
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 999}
        }
        usage = llm_client._extract_usage(msg)
        assert usage == {"input": 12, "output": 34, "total": 46}

    def test_legacy_token_usage_falls_back(self) -> None:
        msg = MagicMock()
        msg.usage_metadata = None
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
        }
        usage = llm_client._extract_usage(msg)
        # Legacy keys are translated to the canonical Langfuse names.
        assert usage == {"input": 7, "output": 11, "total": 18}

    def test_neither_format_returns_none(self) -> None:
        msg = MagicMock()
        msg.usage_metadata = None
        msg.response_metadata = {}
        assert llm_client._extract_usage(msg) is None

    def test_partial_modern_format_uses_zero_defaults(self) -> None:
        # If the upstream only fills input but not output, we still produce
        # a dict — Langfuse's billing math tolerates zero, but None for the
        # whole dict would lose visibility entirely.
        msg = MagicMock()
        msg.usage_metadata = {"input_tokens": 10}  # no output / total
        usage = llm_client._extract_usage(msg)
        assert usage == {"input": 10, "output": 0, "total": 0}

    def test_response_metadata_missing_attribute(self) -> None:
        # Response objects from some gateways don't expose
        # ``response_metadata`` at all. Pin: graceful return of None,
        # not an AttributeError that crashes the request path.
        class _Bare:
            usage_metadata = None
            # no response_metadata attribute

        assert llm_client._extract_usage(_Bare()) is None


# ===================================================================== #
# Module-level — observe decorator integration
# ===================================================================== #


@pytest.mark.unit
class TestObserveDecorator:
    """``LLMClient.invoke`` is decorated with ``@observe(as_type='generation')``.

    Tests run with ``LANGFUSE_ENABLED=false`` (set in tests/conftest.py),
    so the decorator is the no-op stand-in. Pin: decoration does not
    distort the return value or argument forwarding."""

    def test_invoke_signature_unchanged_under_noop_observe(
        self, fake_chat_openai: MagicMock
    ) -> None:
        # The no-op observe must preserve positional + keyword args. This
        # is a defence against re-importing the SDK in the wrong order.
        msg = MagicMock()
        msg.content = "echo: hello"
        msg.usage_metadata = None
        msg.response_metadata = {}
        fake_chat_openai.return_value.invoke.return_value = msg

        client = llm_client.LLMClient()
        out = client.invoke("hello", max_tokens=10)
        assert out == "echo: hello"

    def test_module_uses_observe_from_tracing(self) -> None:
        # Pin the import path: client.py imports observe from tracing
        # (the kill-switch + no-op layer) — NOT directly from langfuse.
        # A regression here would bypass the kill-switch.
        from hse_prom_prog import tracing

        assert llm_client.observe is tracing.observe


# ===================================================================== #
# Tracing metadata — best-effort recording
# ===================================================================== #


@pytest.mark.unit
class TestTracingMetadata:
    def test_observation_updated_with_model_and_input(
        self,
        fake_chat_openai: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Verify the LLMClient feeds the langfuse_context with the prompt
        # and model so production traces can replay calls — even if
        # tracing itself is no-op'd in tests, the call-pattern matters.
        captured: list[dict[str, Any]] = []

        def _record(**kw: Any) -> None:
            captured.append(kw)

        ctx_mock = MagicMock()
        ctx_mock.update_current_observation.side_effect = _record
        monkeypatch.setattr(llm_client, "langfuse_context", ctx_mock)

        msg = MagicMock()
        msg.content = "ok"
        msg.usage_metadata = None
        msg.response_metadata = {}
        fake_chat_openai.return_value.invoke.return_value = msg

        client = llm_client.LLMClient()
        client.invoke("ping", max_tokens=10)
        # First update is the input/model snapshot (before invoke).
        first = captured[0]
        assert first["model"] == client.model
        assert first["input"] == "ping"
        # response_format flag pinned: None when no structured output.
        assert first["model_parameters"]["response_format"] is None
        assert first["model_parameters"]["max_tokens"] == 10

    def test_error_records_level_error(
        self,
        fake_chat_openai: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: on error path, langfuse_context is updated with level=ERROR
        # so failed generations are visible in the dashboard. Without this
        # the trace would still show "running" forever after a crash.
        captured: list[dict[str, Any]] = []
        ctx_mock = MagicMock()
        ctx_mock.update_current_observation.side_effect = lambda **kw: captured.append(kw)
        monkeypatch.setattr(llm_client, "langfuse_context", ctx_mock)

        instance = fake_chat_openai.return_value
        instance.invoke.side_effect = TimeoutError("boom")
        client = llm_client.LLMClient()

        with pytest.raises(TimeoutError):
            client.invoke("p")

        # The last update must carry level=ERROR.
        assert any(c.get("level") == "ERROR" for c in captured)
        assert any("TimeoutError" in (c.get("status_message") or "") for c in captured)


# ===================================================================== #
# httpx-style transport errors
# ===================================================================== #


@pytest.mark.unit
class TestTransportLayerErrors:
    """The underlying httpx transport surfaces ConnectError / ReadTimeout.
    Pin: the LLMClient does NOT translate these — agents need the
    original class to make timeout-vs-other-error decisions."""

    def test_connect_error_propagates(self, fake_chat_openai: MagicMock) -> None:
        # ConnectionRefusedError is the closest stdlib analogue to
        # httpx.ConnectError without dragging the dep into tests.
        instance = fake_chat_openai.return_value
        instance.invoke.side_effect = ConnectionRefusedError("vllm not listening")
        client = llm_client.LLMClient()
        with pytest.raises(ConnectionRefusedError):
            client.invoke("p")

    def test_read_timeout_propagates(self, fake_chat_openai: MagicMock) -> None:
        # Stdlib TimeoutError stands in for httpx.ReadTimeout — both
        # match the "timeout" heuristic in Response Agent's _is_timeout.
        instance = fake_chat_openai.return_value
        instance.invoke.side_effect = TimeoutError("read timeout after 30s")
        client = llm_client.LLMClient()
        with pytest.raises(TimeoutError, match="read timeout"):
            client.invoke("p")


# ===================================================================== #
# Construction error paths
# ===================================================================== #


@pytest.mark.unit
class TestConstructionErrors:
    def test_chat_openai_failure_propagates(self) -> None:
        # If ChatOpenAI ctor explodes (e.g. URL parse error), LLMClient
        # must not swallow it — startup-time crashes are easier to
        # diagnose than silent get_llm_client() returning a half-built
        # object.
        with (
            patch.object(llm_client, "ChatOpenAI", side_effect=ValueError("invalid url")),
            pytest.raises(ValueError, match="invalid url"),
        ):
            llm_client.LLMClient()
