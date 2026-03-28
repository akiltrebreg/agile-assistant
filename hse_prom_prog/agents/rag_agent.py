"""RAG agent: retrieves relevant documents from Qdrant and generates an answer.

The agent retrieves top-k chunks via cosine similarity, optionally reranks them
with a cross-encoder, and sends the best ones as context to the LLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hse_prom_prog.config import settings
from hse_prom_prog.llm.client import LLMClient
from hse_prom_prog.rag.reranker import get_reranker

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_EMPTY = {"rag_response": None, "rag_sources": []}
_MAX_CONTEXT_CHARS = 4000


class RAGAgent:
    """Agent that answers questions using retrieved knowledge-base documents.

    Attributes:
        llm_client: LLM client for generating answers.
        retriever: Qdrant retriever for initial candidate retrieval.
        reranker: Cross-encoder reranker (None if disabled).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Any,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.reranker = get_reranker() if settings.reranker_enabled else None
        logger.info("[RAG Agent] Initialized (reranker=%s)", settings.reranker_enabled)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve context and generate RAG-based answer."""
        original_query = state.get("original_query", "")
        logger.info("[RAG Agent] Processing query: %s", original_query[:80])

        docs = self._retrieve_and_rerank(original_query)
        if not docs:
            return {**_EMPTY}

        context, sources = self._build_context(docs)

        try:
            answer = self._generate_answer(original_query, context)
        except Exception as e:
            logger.error("[RAG Agent] LLM generation failed: %s", e)
            return {**_EMPTY, "rag_sources": sources, "error": str(e)}

        logger.info("[RAG Agent] Generated answer with %d sources", len(sources))
        return {"rag_response": answer, "rag_sources": sources}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_and_rerank(self, query: str) -> list[Document]:
        """Retrieve candidates and optionally rerank."""
        try:
            docs = self.retriever.invoke(query)
        except Exception as e:
            logger.error("[RAG Agent] Retrieval failed: %s", e)
            return []

        if not docs:
            logger.info("[RAG Agent] No relevant documents found")
            return []

        if self.reranker:
            docs = self.reranker.rerank(query, docs)
            if not docs:
                logger.info("[RAG Agent] No docs survived reranking")

        return docs

    @staticmethod
    def _build_context(docs: list[Document]) -> tuple[str, list[str]]:
        """Build context string and source labels from documents."""
        parts: list[str] = []
        total = 0
        for doc in docs:
            chunk = doc.page_content
            if total + len(chunk) > _MAX_CONTEXT_CHARS:
                remaining = _MAX_CONTEXT_CHARS - total
                if remaining > 0:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            total += len(chunk) + 2

        sources: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "")
            label = f"{category}/{source}" if category else source
            if label not in sources:
                sources.append(label)

        return "\n\n".join(parts), sources

    def _generate_answer(self, question: str, context: str) -> str:
        """Send context + question to LLM and return the answer."""
        prompt = (
            f"Ответь на вопрос на основе контекста.\n\nКонтекст:\n{context}\n\nВопрос: {question}"
        )
        return self.llm_client.invoke(prompt)
