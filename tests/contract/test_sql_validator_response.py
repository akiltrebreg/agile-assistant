"""Contract test: SQL Agent → Validator → Response Agent.

Three-step chain that exercises the *only* path most user queries take
("какая velocity команды cthulhu" → SQL → Validator → Response).

Contracts pinned here:

  * SQL Agent's output keys (``sql_query``, ``sql_result``, ``error``,
    ``original_query``) match exactly what Validator reads.
  * Validator's output (``validation_result`` dict with ``use_sql`` /
    ``use_rag`` / ``note``) is consumed by Response Agent's branch
    routing in ``_resolve_branch`` and ``_process_impl``.
  * The chain is *idempotent on shape*: feeding the same SQL output
    through twice produces the same Validator result and routes to
    the same Response branch.

We do NOT exercise the SQL LangGraph (covered in unit tests) — we
inject the SQL Agent's output dict directly. The Validator is the
real implementation; the Response Agent's LLM call is mocked.
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


def _supervisor_state(query_type: str = "sql", intent: str = "metric") -> dict[str, Any]:
    """Pre-Validator state as it would look after Supervisor + SQL Agent."""
    return {
        "query_type": query_type,
        "route": "db_query",
        "intent": intent,
        "original_query": "какая velocity у cthulhu?",
        "entities": {"team_name": "cthulhu", "metric_name": "velocity"},
        "conversation_context": None,
        "user_profile": None,
    }


def _sql_success(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """SQL Agent's success-path output."""
    return {
        "original_query": "какая velocity у cthulhu?",
        "sql_query": "SELECT velocity FROM metrics WHERE team='cthulhu'",
        "sql_result": rows if rows is not None else [{"velocity": 42}],
        "error": None,
    }


def _sql_failure(error: str = "syntax error") -> dict[str, Any]:
    """SQL Agent's error-path output."""
    return {
        "original_query": "какая velocity у cthulhu?",
        "sql_query": None,
        "sql_result": None,
        "error": error,
    }


@pytest.fixture
def validator() -> ValidatorAgent:
    return ValidatorAgent()


@pytest.fixture
def response(mock_llm_client: MagicMock) -> ResponseAgent:
    mock_llm_client.invoke.return_value = "Velocity у cthulhu — 42 SP."
    return ResponseAgent(mock_llm_client)


# ===================================================================== #
# 5.3 — Happy path: SQL succeeds → Validator approves → Response uses sql_result
# ===================================================================== #


@pytest.mark.contract
class TestSqlSuccessChain:
    def test_validator_reads_sql_output_keys(self, validator: ValidatorAgent) -> None:
        # Pin the field names: a rename in SQL Agent (e.g. sql_result →
        # results) without updating Validator would silently set
        # use_sql=False even on success.
        state = {**_supervisor_state(), **_sql_success()}
        out = validator.process(state)
        result = out["validation_result"]
        assert result["use_sql"] is True
        assert result["use_rag"] is False
        assert result["note"] is None

    def test_response_resolves_to_metric_branch_after_validator(
        self, validator: ValidatorAgent, response: ResponseAgent
    ) -> None:
        # The chain SQL→Validator→Response must land in a SQL-flavoured
        # branch (``sql_metric`` for intent=metric). Pin the branch label
        # so a routing-table refactor in _resolve_branch is caught.
        state = {**_supervisor_state(), **_sql_success()}
        v_out = validator.process(state)
        merged = {**state, **v_out}
        assert _resolve_branch(merged) == "sql_metric"
        # Final response was generated, not the empty-result placeholder.
        out = response.process(merged)
        assert out["final_response"] == "Velocity у cthulhu — 42 SP."

    def test_full_chain_passes_sql_result_to_response_llm(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # The data ultimately reaches the LLM prompt only if Validator's
        # use_sql=True flag survives the merge. Pin: Response Agent's
        # LLM was actually called (i.e. no "no data" short-circuit).
        state = {**_supervisor_state(), **_sql_success([{"velocity": 99}])}
        v_out = validator.process(state)
        merged = {**state, **v_out}
        response.process(merged)
        assert mock_llm_client.invoke.called
        prompt = mock_llm_client.invoke.call_args.args[0]
        # Pin: the row dict made it into the prompt — no transform layer
        # silently dropped it between Validator approval and prompt build.
        assert "99" in prompt


# ===================================================================== #
# 5.3 — Empty result path: SQL returns [] but no error → "not found" message
# ===================================================================== #


@pytest.mark.contract
class TestSqlEmptyResultChain:
    def test_empty_sql_result_routes_to_not_found(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # Validator marks use_sql=False when sql_result is empty (note
        # is "No SQL data"); Response Agent's _process_sql_response then
        # short-circuits to a fixed "not found" message.
        state = {**_supervisor_state(intent="task"), **_sql_success(rows=[])}
        state["entities"] = {"issue_key": "AL-9999"}
        v_out = validator.process(state)
        assert v_out["validation_result"]["use_sql"] is False
        assert v_out["validation_result"]["note"] is not None

        merged = {**state, **v_out}
        out = response.process(merged)
        # Pin: the issue_key from upstream entities surfaces in the
        # "not found" message — without it the user can't tell what
        # we tried to look up.
        assert "AL-9999" in out["final_response"]
        assert "не найдена" in out["final_response"].lower()
        # No LLM call: empty-result short-circuit is a fixed string.
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# 5.3 — Error path: SQL Agent raises → Validator records → Response apologises
# ===================================================================== #


@pytest.mark.contract
class TestSqlErrorChain:
    def test_sql_error_propagates_to_response_message(
        self,
        validator: ValidatorAgent,
        response: ResponseAgent,
        mock_llm_client: MagicMock,
    ) -> None:
        # The chain must preserve the original error string all the way
        # through — losing it would force users to ask twice with the
        # same broken query and no diagnostic info.
        state = {**_supervisor_state(), **_sql_failure("relation does not exist")}
        v_out = validator.process(state)
        result = v_out["validation_result"]
        assert result["use_sql"] is False
        # Validator's note is the SQL error verbatim (or "No SQL data" fallback).
        assert "relation does not exist" in result["note"]

        merged = {**state, **v_out}
        out = response.process(merged)
        # Pin: error is reflected in the user-facing message; LLM not called.
        assert "relation does not exist" in out["final_response"]
        mock_llm_client.invoke.assert_not_called()


# ===================================================================== #
# Shape stability
# ===================================================================== #


@pytest.mark.contract
class TestValidatorOutputShape:
    """The Validator's output dict shape is read by Response Agent's
    ``_resolve_branch``. Adding/renaming keys here without updating
    Response would silently change branch routing."""

    _EXPECTED_VALIDATION_KEYS = frozenset({"use_sql", "use_rag", "note"})

    @pytest.mark.parametrize(
        ("query_type", "sql_payload"),
        [
            ("sql", _sql_success()),
            ("sql", _sql_success(rows=[])),
            ("sql", _sql_failure()),
        ],
    )
    def test_validation_result_shape_stable_across_paths(
        self,
        validator: ValidatorAgent,
        query_type: str,
        sql_payload: dict[str, Any],
    ) -> None:
        state = {**_supervisor_state(query_type=query_type), **sql_payload}
        out = validator.process(state)
        assert "validation_result" in out
        assert frozenset(out["validation_result"].keys()) == self._EXPECTED_VALIDATION_KEYS

    def test_resolve_branch_total_over_validator_outputs(self, validator: ValidatorAgent) -> None:
        # _resolve_branch must produce a usable label for *every* valid
        # Validator output. A None or unknown branch would fall through
        # to the default and silently mis-route latency metrics.
        for sql_payload in (_sql_success(), _sql_success([]), _sql_failure()):
            state = {**_supervisor_state(), **sql_payload}
            v_out = validator.process(state)
            merged = {**state, **v_out}
            branch = _resolve_branch(merged)
            assert branch in {"sql_task", "sql_filter", "sql_metric", "sql"}
