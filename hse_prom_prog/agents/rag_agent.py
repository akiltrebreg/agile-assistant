"""RAG agent: retrieves relevant documents from Qdrant and generates an answer.

The agent uses advanced retrieval strategies (Query Rewriting, Multi-Query,
HyDE) combined with hybrid search (BM25 + vector) and Reciprocal Rank Fusion
to find context in the knowledge base, then sends the context + user query to
the LLM for answer generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

from hse_prom_prog.llm.client import LLMClient

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever
    from langchain_qdrant import QdrantVectorStore

    from hse_prom_prog.rag.bm25_index import BM25Index
    from hse_prom_prog.rag.reranker import Reranker

logger = logging.getLogger(__name__)

# Maximum number of characters from retrieved docs to include in the prompt
_MAX_CONTEXT_CHARS = 4000

# Number of results per individual search (increased for reranking)
_TOP_K = 10

# RRF smoothing constant (standard value from the original paper)
_RRF_K = 60

# Warning appended when faithfulness check fails after retry
_FAITHFULNESS_WARNING = (
    "\n\n⚠️ Внимание: ответ мог быть сгенерирован с использованием информации, "
    "выходящей за рамки предоставленного контекста. Проверьте ключевые утверждения "
    "по указанным источникам."
)


class RAGAgent:
    """Agent that answers questions using retrieved knowledge-base documents.

    Uses three retrieval strategies for better recall:
    1. Query Rewriting — LLM rewrites vague queries into precise search terms
    2. Multi-Query — LLM generates 3 alternative formulations
    3. HyDE — LLM generates a hypothetical answer for embedding-based search

    Combined with hybrid search (BM25 + vector) and RRF for ranking,
    cross-encoder reranking with threshold filtering and "Lost in the
    Middle" mitigation, plus metadata filtering by category when intent
    is known.

    Attributes:
        llm_client: LLM client for generating answers.
        retriever: Qdrant vector store retriever (fallback for vector search).
        vector_store: QdrantVectorStore for filtered similarity search.
        bm25_index: BM25 keyword search index.
        reranker: Cross-encoder reranker for second-stage scoring.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: VectorStoreRetriever,
        vector_store: QdrantVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        logger.info(
            "[RAG Agent] Initialized (vector_store=%s, bm25=%s, reranker=%s)",
            "yes" if vector_store else "no",
            "yes" if bm25_index else "no",
            "yes" if reranker else "no",
        )

    # ------------------------------------------------------------------
    # Query enhancement strategies
    # ------------------------------------------------------------------

    def _rewrite_query(self, query: str) -> str:
        """Rewrite a conversational query into a precise search query."""
        prompt = (
            "Ты — помощник для переформулировки вопросов.\n"
            "Перепиши вопрос пользователя в чёткий поисковый запрос "
            "для поиска по базе знаний об Agile-метриках и практиках.\n"
            "Сохрани все упомянутые сущности (команды, метрики, практики).\n"
            "Выведи ТОЛЬКО переформулированный запрос, без пояснений.\n\n"
            f"Вопрос: {query}\n\n"
            "Поисковый запрос:"
        )
        try:
            rewritten = self.llm_client.invoke(prompt).strip()
            if not rewritten:
                return query
            logger.info(
                "[RAG Agent] Rewrote query: '%s' -> '%s'",
                query[:60],
                rewritten[:60],
            )
            return rewritten
        except Exception as e:
            logger.warning("[RAG Agent] Query rewrite failed: %s, using original", e)
            return query

    def _generate_multi_queries(self, query: str) -> list[str]:
        """Generate 3 alternative formulations of the query for broader recall."""
        prompt = (
            "Сгенерируй ровно 3 различные формулировки данного вопроса "
            "для поиска по базе знаний об Agile-метриках и практиках.\n"
            "Каждая формулировка должна искать информацию с разной стороны.\n"
            "Выведи ровно 3 строки, по одному запросу на строку, без нумерации.\n\n"
            f"Исходный вопрос: {query}\n\n"
            "Альтернативные запросы:"
        )
        try:
            raw = self.llm_client.invoke(prompt)
            variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
            if not variants:
                return [query]
            logger.info("[RAG Agent] Generated %d multi-queries", len(variants))
            return variants
        except Exception as e:
            logger.warning("[RAG Agent] Multi-query generation failed: %s", e)
            return [query]

    def _generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical answer document for embedding-based search."""
        prompt = (
            "Ты — эксперт по Agile-метрикам и практикам.\n"
            "Напиши короткий информативный параграф (3-5 предложений), "
            "который мог бы быть фрагментом из базы знаний и который "
            "отвечает на данный вопрос.\n"
            "Пиши как в документации, без обращений к пользователю.\n\n"
            f"Вопрос: {query}\n\n"
            "Фрагмент документации:"
        )
        try:
            hyde_doc = self.llm_client.invoke(prompt).strip()
            logger.info("[RAG Agent] Generated HyDE document (%d chars)", len(hyde_doc))
            return hyde_doc
        except Exception as e:
            logger.warning("[RAG Agent] HyDE generation failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Hybrid search components
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_category_filter(state: dict[str, Any]) -> str | None:
        """Determine metadata category filter from workflow state."""
        intent = state.get("intent", "")
        if intent == "metric":
            return "metrics"
        return None

    def _vector_search(
        self,
        query: str,
        k: int = _TOP_K,
        category: str | None = None,
    ) -> list[Document]:
        """Run vector similarity search, optionally filtered by category."""
        if self.vector_store is not None and category:
            qdrant_filter = Filter(
                must=[FieldCondition(key="metadata.category", match=MatchValue(value=category))]
            )
            return self.vector_store.similarity_search(query, k=k, filter=qdrant_filter)
        if self.vector_store is not None:
            return self.vector_store.similarity_search(query, k=k)
        return self.retriever.invoke(query)

    def _bm25_search(
        self,
        query: str,
        k: int = _TOP_K,
        category: str | None = None,
    ) -> list[Document]:
        """Run BM25 keyword search. Returns empty list if index unavailable."""
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, k=k, category=category)

    @staticmethod
    def _rrf_merge(
        ranked_lists: list[list[Document]],
        k: int = _RRF_K,
    ) -> list[Document]:
        """Merge ranked lists using Reciprocal Rank Fusion.

        RRF score for document d = Σ 1 / (k + rank_i(d)) across all lists.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                content = doc.page_content
                if content not in doc_map:
                    doc_map[content] = doc
                    scores[content] = 0.0
                scores[content] += 1.0 / (k + rank + 1)

        sorted_keys = sorted(scores, key=lambda c: scores[c], reverse=True)
        return [doc_map[c] for c in sorted_keys]

    def _rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Rerank documents using cross-encoder, if available.

        Falls back to returning documents unchanged if reranker is None
        or if reranking fails.
        """
        if self.reranker is None:
            return documents

        try:
            reranked = self.reranker.rerank(query, documents)
            logger.info(
                "[RAG Agent] Reranked %d -> %d documents",
                len(documents),
                len(reranked),
            )
            return reranked
        except Exception as e:
            logger.warning("[RAG Agent] Reranking failed: %s, using original order", e)
            return documents

    @staticmethod
    def _resolve_parents(documents: list[Document]) -> list[Document]:
        """Resolve child chunks to parent content and dedup by parent_id.

        For each child: if parent_content is in metadata, create a Document
        with page_content = parent_content.  Dedup by parent_id (or by
        page_content for backward compatibility with old-format chunks).
        """
        seen_ids: set[str] = set()
        parents: list[Document] = []

        for doc in documents:
            parent_id = doc.metadata.get("parent_id")
            parent_content = doc.metadata.get("parent_content")

            dedup_key = parent_id if parent_id else doc.page_content
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)

            resolved_content = parent_content if parent_content else doc.page_content
            parents.append(Document(page_content=resolved_content, metadata={**doc.metadata}))

        return parents

    @staticmethod
    def _format_context_block(doc: Document) -> str:
        """Format a document into a context block with structured metadata header.

        Format:
            [Источник: <filename> | Раздел: <section> | Стр. <page>]
            <content>

        Page part is omitted for documents without page metadata.
        """
        source = doc.metadata.get("source", "unknown")
        source_name = Path(source).name if source != "unknown" else "unknown"

        section = doc.metadata.get("section", "\u2014")
        page = doc.metadata.get("page")

        header_parts = [f"Источник: {source_name}", f"Раздел: {section}"]
        if page is not None:
            header_parts.append(f"Стр. {page}")

        header = " | ".join(header_parts)
        return f"[{header}]\n{doc.page_content}"

    def _hybrid_search_sub_query(
        self,
        sub_query: str,
        category_filter: str | None,
    ) -> list[Document]:
        """Run hybrid search (vector + BM25) for a single sub-query and RRF-merge."""
        try:
            vector_results = self._vector_search(sub_query, k=_TOP_K, category=category_filter)
        except Exception as e:
            logger.warning("[RAG Agent] Vector search failed for sub-query: %s", e)
            vector_results = []

        bm25_results = self._bm25_search(sub_query, k=_TOP_K, category=category_filter)
        return self._rrf_merge([vector_results, bm25_results])

    def _assemble_context(
        self,
        docs: list[Document],
    ) -> tuple[str, list[str]]:
        """Truncate docs to _MAX_CONTEXT_CHARS and collect sources."""
        context_parts: list[str] = []
        sources: list[str] = []
        total_chars = 0

        for doc in docs:
            block = self._format_context_block(doc)
            if total_chars + len(block) > _MAX_CONTEXT_CHARS:
                break
            context_parts.append(block)
            total_chars += len(block)

            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "")
            label = f"{category}/{source}" if category else source
            if label not in sources:
                sources.append(label)

        return "\n\n---\n\n".join(context_parts), sources

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        category_filter: str | None = None,
    ) -> tuple[str, list[str]]:
        """Retrieve relevant chunks using advanced strategies.

        Pipeline: Query Rewrite → Multi-Query + HyDE → hybrid search (vector + BM25)
        → RRF merge → dedup → cross-encoder rerank → resolve parents → truncate.
        """
        # Step 1: Rewrite the query
        rewritten = self._rewrite_query(query)

        # Step 2: Generate multi-queries and HyDE document
        multi_queries = self._generate_multi_queries(rewritten)
        hyde_doc = self._generate_hyde_document(rewritten)

        # Step 3: Collect all search queries
        search_queries = list(multi_queries)
        if hyde_doc:
            search_queries.append(hyde_doc)

        # Step 4: Hybrid search for each sub-query + RRF merge + dedup
        all_docs: list[Document] = []
        seen_contents: set[str] = set()

        for sq in search_queries:
            for doc in self._hybrid_search_sub_query(sq, category_filter):
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)

        if not all_docs:
            return "", []

        # Step 5: Cross-encoder reranking (threshold + "Lost in the Middle")
        all_docs = self._rerank(query, all_docs)

        if not all_docs:
            return "", []

        # Step 6: Resolve children to parents, dedup by parent_id
        all_docs = self._resolve_parents(all_docs)

        if not all_docs:
            return "", []

        # Step 7: Assemble context with structured metadata headers
        context, sources = self._assemble_context(all_docs)
        logger.info(
            "[RAG Agent] Retrieved %d parents (%d chars) from %d sources "
            "(via %d sub-queries, hybrid=%s)",
            len(all_docs),
            len(context),
            len(sources),
            len(search_queries),
            "yes" if self.bm25_index else "vector-only",
        )
        return context, sources

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(self, original_query: str, context: str, sources: list[str]) -> str:
        """Send context + query to LLM and return the answer."""
        prompt = (
            "Ты — ассистент для анализа Agile-практик и метрик.\n"
            "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
            "Если в контексте нет ответа, честно скажи об этом.\n"
            "Отвечай на русском языке.\n\n"
            "Правила оформления ответа:\n"
            "1. Для каждого утверждения ссылайся на источник в формате "
            "[имя_файла, стр. N] или [имя_файла, раздел], "
            "используя метаданные из заголовков контекстных блоков.\n"
            "2. Если два источника противоречат друг другу, явно укажи:\n"
            '   "⚠️ Источники содержат противоречие: [источник1] утверждает X, '
            'а [источник2] утверждает Y"\n\n'
            f"Контекст из базы знаний:\n{context}\n\n"
            f"Вопрос пользователя: {original_query}\n\n"
            "Ответ:"
        )
        return self.llm_client.invoke(prompt)

    def _check_faithfulness(self, answer: str, context: str) -> bool:
        """Check if answer is grounded in the provided context.

        Returns True if all claims in the answer are supported by context.
        """
        prompt = (
            "Ты — строгий верификатор фактов.\n"
            "Проверь, подтверждается ли КАЖДОЕ утверждение в ответе контекстом.\n\n"
            "Если все утверждения подтверждаются — выведи ТОЛЬКО слово: FAITHFUL\n"
            "Если хотя бы одно не подтверждается — выведи ТОЛЬКО слово: UNFAITHFUL\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ОТВЕТ:\n{answer}\n\n"
            "Вердикт:"
        )
        result = self.llm_client.invoke(prompt).strip().upper()
        return "UNFAITHFUL" not in result

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
        category_filter = self._resolve_category_filter(state)
        logger.info(
            "[RAG Agent] Processing query: %s (category_filter=%s)",
            original_query[:80],
            category_filter,
        )

        try:
            context, sources = self._retrieve(original_query, category_filter)
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

        # Faithfulness check with one retry
        try:
            is_faithful = self._check_faithfulness(answer, context)
            if not is_faithful:
                logger.warning("[RAG Agent] Answer failed faithfulness check, retrying")
                try:
                    answer = self._generate_answer(original_query, context, sources)
                    is_faithful = self._check_faithfulness(answer, context)
                except Exception as e:
                    logger.warning("[RAG Agent] Faithfulness retry failed: %s", e)
                    is_faithful = False
                if not is_faithful:
                    logger.warning("[RAG Agent] Answer still unfaithful after retry")
                    answer += _FAITHFULNESS_WARNING
        except Exception as e:
            logger.warning("[RAG Agent] Faithfulness check failed: %s, skipping", e)

        logger.info("[RAG Agent] Generated answer with %d sources", len(sources))
        return {
            "rag_response": answer,
            "rag_sources": sources,
        }
