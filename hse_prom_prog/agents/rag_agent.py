"""RAG agent: retrieves relevant documents from Qdrant and generates an answer.

The agent uses the retriever to find context in the knowledge base,
then sends the context + user query to the LLM for answer generation.
"""

import logging
from typing import Any

from langchain_core.vectorstores import VectorStoreRetriever

from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Maximum number of characters from retrieved docs to include in the prompt
_MAX_CONTEXT_CHARS = 4000


class RAGAgent:
    """Agent that answers questions using retrieved knowledge-base documents.

    Attributes:
        llm_client: LLM client for generating answers.
        retriever: Qdrant vector store retriever.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: VectorStoreRetriever,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        logger.info("[RAG Agent] Initialized")

    def _retrieve(self, query: str) -> tuple[str, list[str]]:
        """Retrieve relevant chunks and return (context_text, source_names)."""
        docs = self.retriever.invoke(query)

        if not docs:
            return "", []

        context_parts: list[str] = []
        sources: list[str] = []
        total_chars = 0

        for doc in docs:
            text = doc.page_content
            if total_chars + len(text) > _MAX_CONTEXT_CHARS:
                break
            context_parts.append(text)
            total_chars += len(text)

            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "")
            label = f"{category}/{source}" if category else source
            if label not in sources:
                sources.append(label)

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "[RAG Agent] Retrieved %d chunks (%d chars) from %d sources",
            len(context_parts),
            total_chars,
            len(sources),
        )
        return context, sources

    def _generate_answer(self, original_query: str, context: str, sources: list[str]) -> str:
        """Send context + query to LLM and return the answer."""
        prompt = (
            "Ты — ассистент для анализа Agile-практик и метрик.\n"
            "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
            "Если в контексте нет ответа, честно скажи об этом.\n"
            "Отвечай на русском языке.\n\n"
            f"Контекст из базы знаний:\n{context}\n\n"
            f"Вопрос пользователя: {original_query}\n\n"
            "Ответ:"
        )
        return self.llm_client.invoke(prompt)

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve context and generate RAG-based answer.

        Args:
            state: Workflow state with original_query.

        Returns:
            State update with rag_response and rag_sources.
        """
        original_query = state.get("original_query", "")
        logger.info("[RAG Agent] Processing query: %s", original_query[:80])

        try:
            context, sources = self._retrieve(original_query)
        except Exception as e:
            logger.error("[RAG Agent] Retrieval failed: %s", e)
            return {
                "rag_response": None,
                "rag_sources": [],
                "error": f"RAG retrieval error: {e}",
            }

        if not context:
            logger.info("[RAG Agent] No relevant documents found")
            return {
                "rag_response": None,
                "rag_sources": [],
            }

        try:
            answer = self._generate_answer(original_query, context, sources)
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
