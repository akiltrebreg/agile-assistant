"""Validator agent: checks outputs of SQL Agent and RAG Agent before Response Agent.

Ensures that at least one agent returned valid data and picks
the best combination for the final response.
"""

import logging
from typing import Any

from hse_prom_prog.metrics import VALIDATOR_DATA_MISSING, VALIDATOR_RESULTS
from hse_prom_prog.tracing import langfuse_context

logger = logging.getLogger(__name__)


def _record(use_sql: bool, use_rag: bool) -> None:
    """Record the ``(use_sql, use_rag)`` pair on ``VALIDATOR_RESULTS``.

    Bool-to-str cast is required: Prometheus labels must be strings, and
    relying on ``str(True)`` produces ``"True"`` / ``"False"`` — explicit
    ``lower()`` matches the convention used elsewhere in the codebase.
    """
    VALIDATOR_RESULTS.labels(
        use_sql=str(use_sql).lower(),
        use_rag=str(use_rag).lower(),
    ).inc()


class ValidatorAgent:
    """Validates and merges results from SQL Agent and RAG Agent.

    For 'hybrid' queries both agents run; the Validator decides which
    results are usable and passes them to the Response Agent.
    """

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate agent outputs and produce a ``validation_result``.

        Args:
            state: Workflow state with ``sql_result``, ``rag_response``,
                ``error``, etc.

        Returns:
            State update with the ``validation_result`` dict (keys
            ``use_sql``, ``use_rag``, ``note``).
        """
        query_type = state.get("query_type", "sql")
        sql_result = state.get("sql_result")
        sql_error = state.get("error")
        rag_response = state.get("rag_response")
        rag_sources = state.get("rag_sources", [])

        sql_ok = bool(sql_result) and not sql_error
        rag_ok = bool(rag_response)

        logger.info(
            "[Validator] query_type=%s, sql_ok=%s, rag_ok=%s",
            query_type,
            sql_ok,
            rag_ok,
        )

        langfuse_context.update_current_observation(
            input={
                "query_type": query_type,
                "has_sql_result": bool(sql_result),
                "has_rag_response": rag_ok,
                "sql_error": bool(sql_error),
            },
        )

        if query_type == "sql":
            _record(sql_ok, False)
            if not sql_ok:
                VALIDATOR_DATA_MISSING.labels(source="sql").inc()
            payload = {
                "use_sql": sql_ok,
                "use_rag": False,
                "note": None if sql_ok else (sql_error or "No SQL data"),
            }
            langfuse_context.update_current_observation(output=payload)
            return {"validation_result": payload}

        if query_type == "rag":
            _record(False, rag_ok)
            if not rag_ok:
                VALIDATOR_DATA_MISSING.labels(source="rag").inc()
            payload = {
                "use_sql": False,
                "use_rag": rag_ok,
                "note": None if rag_ok else "No relevant documents found",
            }
            langfuse_context.update_current_observation(output=payload)
            return {"validation_result": payload}

        # hybrid — use whatever is available
        if not sql_ok and not rag_ok:
            note = sql_error or "No data from SQL or RAG"
            logger.warning("[Validator] Both agents returned nothing: %s", note)
            _record(False, False)
            VALIDATOR_DATA_MISSING.labels(source="both").inc()
            payload = {"use_sql": False, "use_rag": False, "note": note}
            langfuse_context.update_current_observation(output=payload)
            return {"validation_result": payload}

        logger.info(
            "[Validator] Hybrid — sql=%s, rag=%s, sources=%d",
            sql_ok,
            rag_ok,
            len(rag_sources),
        )
        _record(sql_ok, rag_ok)
        # Hybrid with one source missing — record which one didn't supply data.
        if not sql_ok:
            VALIDATOR_DATA_MISSING.labels(source="sql").inc()
        if not rag_ok:
            VALIDATOR_DATA_MISSING.labels(source="rag").inc()
        payload = {"use_sql": sql_ok, "use_rag": rag_ok, "note": None}
        langfuse_context.update_current_observation(output=payload)
        return {"validation_result": payload}
