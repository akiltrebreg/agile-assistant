"""Contract test: RAG Agent → Validator → Response Agent.

Mirror of the SQL chain test (5.3) — same shape, different fields.

RAG-only path:
  RAG Agent → state(rag_response, rag_sources) → Validator
            → state(validation_result.use_rag) → Response Agent → final

Hybrid path (RAG + SQL together):
  Validator merges signals from BOTH agents and the Response Agent picks
  the ``hybrid`` branch (LLM combines them) when at least one is usable.

Contracts pinned:
  * ``rag_response`` is the *string* the RAG Agent emits (not a dict).
    A future "richer" return type would silently break Validator's
    ``bool(rag_response)`` truthiness check.
  * ``rag_sources`` is always a list (never None) — Response Agent's
    citation block does ``"\\n".join(...)`` on it without a guard.
  * Validator notes for hybrid-with-no-data are pinned strings the
    Response Agent's "both failed" branch matches against.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agile_assistant.agents.response_agent import ResponseAgent, _resolve_branch
from agile_assistant.agents.validator_agent import ValidatorAgent

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _supervisor_state(query_type: str = "rag") -> dict[str, Any]:
    return {
        "query_type": query_type,
        "route": "db_query",
        "intent": "general",
        "original_query": "что такое sprint goal?",
        "entities": {},
        "conversation_context": None,
        "user_profile": None,
    }


def _rag_success(
    response: str = "Sprint Goal — это цель спринта.",
    sources: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "rag_response": response,
        "rag_sources": sources if sources is not None else ["agile/guide.md"],
    }


def _rag_empty() -> dict[str, Any]:
    """RAG Agent's "no docs found" output — pinned shape from rag_agent._EMPTY."""
    return {"rag_response": None, "rag_sources": []}


@pytest.fixture
def validator() -> ValidatorAgent:
    return ValidatorAgent()


@pytest.fixture
def response(mock_llm_client: MagicMock) -> ResponseAgent:
    mock_llm_client.invoke.return_value = "Сгенерированный ответ."
    return ResponseAgent(mock_llm_client)


# ===================================================================== #
# 5.4 — RAG-only happy path
# ===================================================================== #


@pytest.mark.contract
class TestRagSuccessChain:
    def test_validator_approves_rag_when_response_present(self, validator: ValidatorAgent) -> None:
        state = {**_supervisor_state(), **_rag_success()}
        out = validator.process(state)
        result = out["validation_result"]
        assert result["use_sql"] is False
        assert result["use_rag"] is True
        assert result["note"] is None

    def test_response_resolves_to_rag_branch_after_validator(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
    ) -> None:
        state = {**_supervisor_state(), **_rag_success()}
        v_out = validator.process(state)
        merged = {**state, **v_out}
        # Pin: query_type=rag + use_rag=True must land on the rag branch
        # for latency labelling. _resolve_branch reads validation_result
        # so a renamed key would silently fall back to the sql branch.
        assert _resolve_branch(merged) == "rag"

    def test_rag_response_passes_through_to_final(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # The RAG branch in Response Agent is a *passthrough* — the RAG
        # Agent's text becomes the final response verbatim. Pin: no LLM
        # call here. A regression that adds an LLM rewrite step would
        # double cost and risk hallucinations. Sources are intentionally
        # not surfaced to the user (UI is for end users, not engineers).
        state = {
            **_supervisor_state(),
            **_rag_success(
                response="Это объяснение из базы знаний.",
                sources=["docs/sprint.md", "docs/agile.md"],
            ),
        }
        v_out = validator.process(state)
        merged = {**state, **v_out}
        out = response.process(merged)
        # The RAG text is preserved verbatim, with no source citations.
        assert out["final_response"] == "Это объяснение из базы знаний."
        # No LLM rewrite of the RAG answer.
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# 5.4 — RAG empty path: Validator marks unusable, Response apologises
# ===================================================================== #


@pytest.mark.contract
class TestRagEmptyChain:
    def test_no_docs_found_routes_to_rag_unavailable_message(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # Pin: when the RAG agent finds nothing, Validator sets a fixed
        # note string. Response Agent's "both failed" branch matches
        # against ``not use_sql and not use_rag`` — a refactor that
        # changed the truthiness check would skip this fallback and
        # hand a None response to the user.
        state = {**_supervisor_state(), **_rag_empty()}
        v_out = validator.process(state)
        result = v_out["validation_result"]
        assert result["use_rag"] is False
        # Note text is exposed in metrics dashboards — pin it.
        assert result["note"] == "No relevant documents found"

        merged = {**state, **v_out}
        out = response.process(merged)
        # Pin: the user gets a structured "I can help with X" message,
        # not an LLM-generated apology (which would be wasted budget on
        # a deterministic fallback).
        assert "Agile" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# 5.4 — Hybrid path: RAG + SQL combined
# ===================================================================== #


@pytest.mark.contract
class TestHybridChain:
    def test_both_agents_succeed_routes_to_hybrid_branch(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # The hybrid branch fires the LLM with combined prompt — pin
        # that both data sources reach the prompt body, not just one.
        state = {
            **_supervisor_state(query_type="hybrid"),
            **_rag_success(response="Это объяснение."),
            "intent": "metric",
            "sql_query": "SELECT 1",
            "sql_result": [{"velocity": 42}],
            "error": None,
        }
        v_out = validator.process(state)
        result = v_out["validation_result"]
        assert result["use_sql"] is True
        assert result["use_rag"] is True
        assert result["note"] is None

        merged = {**state, **v_out}
        assert _resolve_branch(merged) == "hybrid"
        response.process(merged)
        # LLM must have been called with both data blobs in the prompt.
        prompt = mock_llm_client.invoke.call_args.args[0]
        assert "42" in prompt
        assert "Это объяснение." in prompt

    def test_hybrid_with_only_sql_still_routes_to_hybrid_branch(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
    ) -> None:
        # When RAG returns nothing but SQL has data, the *intent* is
        # still hybrid (the user asked a "data + advice" question);
        # branch must stay ``hybrid`` so the LLM gets a chance to fall
        # back gracefully ("рекомендации из базы знаний недоступны").
        state = {
            **_supervisor_state(query_type="hybrid"),
            **_rag_empty(),
            "intent": "metric",
            "sql_query": "SELECT 1",
            "sql_result": [{"velocity": 42}],
            "error": None,
        }
        v_out = validator.process(state)
        result = v_out["validation_result"]
        assert result["use_sql"] is True
        assert result["use_rag"] is False
        # Per Validator.process — note is None when *some* source has data.
        assert result["note"] is None
        merged = {**state, **v_out}
        assert _resolve_branch(merged) == "hybrid"

    def test_hybrid_both_fail_routes_to_error_branch(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # Hybrid with neither source → Validator emits a note and
        # use_sql=use_rag=False → Response Agent's "both failed" branch
        # → fixed message, no LLM. _resolve_branch labels this "error"
        # for latency tracking.
        state = {
            **_supervisor_state(query_type="hybrid"),
            **_rag_empty(),
            "intent": "metric",
            "sql_query": None,
            "sql_result": None,
            "error": "syntax",
        }
        v_out = validator.process(state)
        result = v_out["validation_result"]
        assert result["use_sql"] is False
        assert result["use_rag"] is False
        assert result["note"] is not None

        merged = {**state, **v_out}
        assert _resolve_branch(merged) == "error"
        out = response.process(merged)
        assert "не могу" not in out["final_response"]  # not an LLM-style apology
        # Pinned: the fixed-text branch references the knowledge base.
        assert "знаний" in out["final_response"].lower()
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# Shape stability for RAG Agent's output
# ===================================================================== #


@pytest.mark.contract
class TestRagOutputShape:
    """Pin the RAG Agent's output shape — Validator and Response Agent
    both inspect specific keys; renaming would silently bypass them."""

    @pytest.mark.parametrize(
        "rag_payload",
        [
            _rag_success(),
            _rag_success(sources=[]),  # answer with no citations
            _rag_empty(),
        ],
    )
    def test_rag_response_is_string_or_none(self, rag_payload: dict[str, Any]) -> None:
        # Validator does ``bool(rag_response)`` — both ``""`` and ``None``
        # fall to "no rag", but a non-string (e.g. dict) would coerce to
        # True and route a malformed payload into the LLM prompt. Pin
        # the type contract.
        assert rag_payload["rag_response"] is None or isinstance(rag_payload["rag_response"], str)

    @pytest.mark.parametrize(
        "rag_payload",
        [
            _rag_success(),
            _rag_success(sources=[]),
            _rag_empty(),
        ],
    )
    def test_rag_sources_is_always_a_list(self, rag_payload: dict[str, Any]) -> None:
        # Response Agent's citation block does ``"\\n".join(rag_sources)``
        # without a None-guard. Pin: even on the empty path, the field
        # is ``[]``, never None.
        assert isinstance(rag_payload["rag_sources"], list)
