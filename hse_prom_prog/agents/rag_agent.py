"""RAG agent: retrieves relevant documents from Qdrant and generates an answer.

The agent retrieves top-k chunks via cosine similarity and sends them
as context to the LLM for answer generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hse_prom_prog.llm.client import LLMClient

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class RAGAgent:
    """Agent that answers questions using retrieved knowledge-base documents.

    Attributes:
        llm_client: LLM client for generating answers.
        retriever: Qdrant vector store retriever (top-4, cosine similarity).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: VectorStoreRetriever,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        logger.info("[RAG Agent] Initialized")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve context and generate RAG-based answer.

        Args:
            state: Workflow state with original_query, intent, entities.

        Returns:
            State update with rag_response and rag_sources.
        """
        original_query = state.get("original_query", "")
        logger.info("[RAG Agent] Processing query: %s", original_query[:80])

        # Step 1: Retrieve top-k documents
        try:
            docs = self.retriever.invoke(original_query)
        except Exception as e:
            logger.error("[RAG Agent] Retrieval failed: %s", e)
            return {
                "rag_response": None,
                "rag_sources": [],
                "error": f"RAG retrieval error: {e}",
            }

        if not docs:
            logger.info("[RAG Agent] No relevant documents found")
            return {
                "rag_response": None,
                "rag_sources": [],
            }

        # Step 2: Build context and collect sources
        context = "\n\n".join(doc.page_content for doc in docs)
        sources: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "")
            label = f"{category}/{source}" if category else source
            if label not in sources:
                sources.append(label)

        # Step 3: Generate answer
        try:
            answer = self._generate_answer(original_query, context)
        except Exception as e:
            logger.error("[RAG Agent] LLM generation failed: %s", e)
            return {
                "rag_response": None,
                "rag_sources": sources,
                "error": f"RAG generation error: {e}",
            }

        logger.info("[RAG Agent] Generated answer with %d sources", len(sources))
        return {
            "rag_response": answer,
            "rag_sources": sources,
        }

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(self, question: str, context: str) -> str:
        """Send context + question to LLM and return the answer."""
        prompt = (
            f"Ответь на вопрос на основе контекста.\n\nКонтекст:\n{context}\n\nВопрос: {question}"
        )
        return self.llm_client.invoke(prompt)
