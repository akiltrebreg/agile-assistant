"""Unit tests for ``ValidatorAgent.process``.

Pure routing logic — no LLM, no I/O, no external state. Decides which of
SQL / RAG outputs flow through to the Response Agent based on:

  * ``query_type``         — sql / rag / hybrid (default: sql)
  * ``sql_ok``             — sql_result is truthy AND error is empty
  * ``rag_ok``             — rag_response is truthy

Bugs to catch: a hybrid response with one source missing must NOT be
flagged as a hard failure (use_sql/use_rag carry the survivor); a "sql"
query with an error must propagate that error as the note (so the
Response Agent can surface it instead of silently producing empty output).
"""

from __future__ import annotations

from typing import Any

import pytest

from agile_assistant.agents.validator_agent import ValidatorAgent


@pytest.fixture
def validator() -> ValidatorAgent:
    return ValidatorAgent()


def _state(**overrides: Any) -> dict[str, Any]:
    """Minimal state with workflow-shape defaults; tests override as needed."""
    base: dict[str, Any] = {
        "query_type": "sql",
        "sql_result": [],
        "rag_response": "",
        "rag_sources": [],
        "error": "",
    }
    base.update(overrides)
    return base


def _payload(validator: ValidatorAgent, **overrides: Any) -> dict[str, Any]:
    """Run validator and return the inner ``validation_result`` payload directly."""
    out = validator.process(_state(**overrides))
    assert "validation_result" in out
    return out["validation_result"]


# ===================================================================== #
# query_type == "sql"
# ===================================================================== #


@pytest.mark.unit
class TestSqlBranch:
    def test_sql_with_data_uses_only_sql(self, validator: ValidatorAgent) -> None:
        payload = _payload(
            validator,
            query_type="sql",
            sql_result=[{"key": "AL-1"}],
        )
        assert payload == {"use_sql": True, "use_rag": False, "note": None}

    def test_sql_with_empty_result_falls_back_with_default_note(
        self, validator: ValidatorAgent
    ) -> None:
        # No DB error, just empty result set — Response Agent gets a
        # generic "no data" note so it can format a polite reply.
        payload = _payload(validator, query_type="sql", sql_result=[])
        assert payload["use_sql"] is False
        assert payload["use_rag"] is False
        assert payload["note"] == "No SQL data"

    def test_sql_error_propagates_through_note(self, validator: ValidatorAgent) -> None:
        # The error string must surface to the Response Agent verbatim —
        # bury it and the user sees a misleading "no data found" reply.
        payload = _payload(
            validator,
            query_type="sql",
            sql_result=[],
            error="psycopg2.errors.UndefinedTable: no such table",
        )
        assert payload["use_sql"] is False
        assert "UndefinedTable" in payload["note"]

    def test_sql_error_with_data_still_fails(self, validator: ValidatorAgent) -> None:
        # If both populated — error must win (it indicates the data is
        # partial / unreliable).
        payload = _payload(
            validator,
            query_type="sql",
            sql_result=[{"key": "AL-1"}],
            error="connection lost mid-query",
        )
        assert payload["use_sql"] is False


# ===================================================================== #
# query_type == "rag"
# ===================================================================== #


@pytest.mark.unit
class TestRagBranch:
    def test_rag_with_response_uses_only_rag(self, validator: ValidatorAgent) -> None:
        payload = _payload(
            validator,
            query_type="rag",
            rag_response="Velocity — это метрика командной скорости.",
            rag_sources=["agile-guide.pdf"],
        )
        assert payload == {"use_sql": False, "use_rag": True, "note": None}

    def test_rag_with_empty_response_falls_back(self, validator: ValidatorAgent) -> None:
        payload = _payload(validator, query_type="rag", rag_response="")
        assert payload["use_sql"] is False
        assert payload["use_rag"] is False
        assert payload["note"] == "No relevant documents found"

    def test_rag_branch_ignores_sql_error(self, validator: ValidatorAgent) -> None:
        # An SQL error has no bearing on a RAG-only query — the validator
        # must not let it leak into the note or block use_rag.
        payload = _payload(
            validator,
            query_type="rag",
            rag_response="some doc text",
            error="some sql error",
        )
        assert payload["use_rag"] is True
        assert payload["note"] is None


# ===================================================================== #
# query_type == "hybrid"
# ===================================================================== #


@pytest.mark.unit
class TestHybridBranch:
    def test_both_sources_available(self, validator: ValidatorAgent) -> None:
        payload = _payload(
            validator,
            query_type="hybrid",
            sql_result=[{"key": "AL-1"}],
            rag_response="Practice description",
            rag_sources=["doc.pdf"],
        )
        assert payload == {"use_sql": True, "use_rag": True, "note": None}

    def test_sql_missing_rag_present(self, validator: ValidatorAgent) -> None:
        # Survivor flows through — note stays None so the Response Agent
        # doesn't surface a "missing data" warning when one source is enough.
        payload = _payload(
            validator,
            query_type="hybrid",
            sql_result=[],
            rag_response="doc text",
        )
        assert payload == {"use_sql": False, "use_rag": True, "note": None}

    def test_sql_present_rag_missing(self, validator: ValidatorAgent) -> None:
        payload = _payload(
            validator,
            query_type="hybrid",
            sql_result=[{"key": "AL-1"}],
            rag_response="",
        )
        assert payload == {"use_sql": True, "use_rag": False, "note": None}

    def test_both_missing_sets_note_to_default(self, validator: ValidatorAgent) -> None:
        payload = _payload(validator, query_type="hybrid", sql_result=[], rag_response="")
        assert payload["use_sql"] is False
        assert payload["use_rag"] is False
        assert payload["note"] == "No data from SQL or RAG"

    def test_both_missing_with_sql_error_uses_error_text(self, validator: ValidatorAgent) -> None:
        # A more informative note when we actually know what failed —
        # propagate the SQL error verbatim instead of the generic message.
        payload = _payload(
            validator,
            query_type="hybrid",
            sql_result=[],
            rag_response="",
            error="UndefinedTable",
        )
        assert payload["note"] == "UndefinedTable"


# ===================================================================== #
# Defaults & shape
# ===================================================================== #


@pytest.mark.unit
class TestDefaultsAndShape:
    def test_missing_query_type_defaults_to_sql_branch(self, validator: ValidatorAgent) -> None:
        # state.get("query_type", "sql") — when supervisor hasn't set it
        # the validator must still produce a coherent payload.
        out = validator.process({"sql_result": [{"k": 1}], "error": ""})
        assert out["validation_result"]["use_sql"] is True
        assert out["validation_result"]["use_rag"] is False

    def test_returned_state_only_contains_validation_result(
        self, validator: ValidatorAgent
    ) -> None:
        # LangGraph state updates must be partial — the validator must
        # not echo back unrelated keys (no `query_type` etc.) to avoid
        # silently overriding fields elsewhere.
        out = validator.process(_state(query_type="sql", sql_result=[{"k": 1}]))
        assert set(out.keys()) == {"validation_result"}

    def test_payload_keys_are_stable(self, validator: ValidatorAgent) -> None:
        # Pin the contract: every payload has these three keys, no others —
        # downstream Response Agent reads them by name.
        for query_type in ("sql", "rag", "hybrid"):
            payload = _payload(validator, query_type=query_type)
            assert set(payload.keys()) == {"use_sql", "use_rag", "note"}
