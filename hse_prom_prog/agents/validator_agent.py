"""Validator agent: checks outputs of SQL Agent and RAG Agent before Response Agent.

Ensures that at least one agent returned valid data and picks
the best combination for the final response.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """Validates and merges results from SQL Agent and RAG Agent.

    For 'hybrid' queries both agents run; the Validator decides which
    results are usable and passes them to the Response Agent.
    """

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate agent outputs and produce a validation_result.

        Args:
            state: Workflow state with sql_result, rag_response, error, etc.

        Returns:
            State update with validation_result dict.
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

        if query_type == "sql":
            return {
                "validation_result": {
                    "use_sql": sql_ok,
                    "use_rag": False,
                    "note": None if sql_ok else (sql_error or "No SQL data"),
                },
            }

        if query_type == "rag":
            return {
                "validation_result": {
                    "use_sql": False,
                    "use_rag": rag_ok,
                    "note": None if rag_ok else "No relevant documents found",
                },
            }

        # hybrid — use whatever is available
        if not sql_ok and not rag_ok:
            note = sql_error or "No data from SQL or RAG"
            logger.warning("[Validator] Both agents returned nothing: %s", note)
            return {
                "validation_result": {
                    "use_sql": False,
                    "use_rag": False,
                    "note": note,
                },
            }

        logger.info(
            "[Validator] Hybrid — sql=%s, rag=%s, sources=%d",
            sql_ok,
            rag_ok,
            len(rag_sources),
        )
        return {
            "validation_result": {
                "use_sql": sql_ok,
                "use_rag": rag_ok,
                "note": None,
            },
        }
