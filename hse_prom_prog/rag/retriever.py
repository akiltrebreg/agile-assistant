"""Qdrant retriever with three search modes: dense, sparse, hybrid.

Modes (controlled by ``SEARCH_TYPE`` env / ``settings.search_type``):

- **dense**  — cosine similarity on dense vectors (default, backward-compatible)
- **sparse** — BM25 keyword search via Qdrant sparse vectors
- **hybrid** — dense + sparse with native Qdrant RRF fusion (prefetch API)

All modes expose the same ``.invoke(query)`` interface returning
``list[Document]`` so callers (rag_agent, run_eval) need no changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from hse_prom_prog.config import settings
from hse_prom_prog.metrics import (
    RAG_CHUNKS_RETRIEVED,
    RAG_FALLBACKS,
    RAG_REQUESTS,
    RAG_RETRIEVAL_DURATION,
    RAG_TOP_SCORE,
)
from hse_prom_prog.rag.embeddings import get_embeddings, get_target_dim, truncate_vector
from hse_prom_prog.rag.ingest import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from hse_prom_prog.rag.sparse import embed_sparse
from hse_prom_prog.tracing import langfuse_context, observe

logger = logging.getLogger(__name__)

# Module-level singletons
_vector_store: QdrantVectorStore | None = None
_retriever: MultiModeRetriever | None = None


# ── Dense vector store (used by dense & hybrid modes) ────────


def _build_vector_store() -> QdrantVectorStore:
    """Create a QdrantVectorStore backed by the existing collection."""
    embeddings = get_embeddings()
    client = QdrantClient(url=settings.qdrant_url)
    collection = settings.qdrant_collection_name

    if not client.collection_exists(collection):
        logger.warning(
            "[Retriever] Collection '%s' does not exist. "
            "Run ingestion first: python -m hse_prom_prog.rag.ingest",
            collection,
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        vector_name=DENSE_VECTOR_NAME,
        validate_collection_config=False,
    )
    logger.info("[Retriever] Vector store initialized (collection=%s)", collection)
    return store


def get_vector_store() -> QdrantVectorStore:
    """Return the module-level vector store singleton (lazy init)."""
    global _vector_store  # noqa: PLW0603
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store


# ── Multi-mode retriever ─────────────────────────────────────


def _point_to_document(point: models.ScoredPoint) -> Document:
    """Convert a Qdrant ScoredPoint to a LangChain Document."""
    payload = point.payload or {}
    return Document(
        page_content=payload.get("page_content", ""),
        metadata=payload.get("metadata", {}),
    )


class MultiModeRetriever:
    """Retriever that supports dense, sparse, and hybrid search.

    Exposes ``.invoke(query)`` so it's a drop-in replacement for
    LangChain's ``VectorStoreRetriever``.

    If sparse vectors are missing from the collection (old ingestion),
    sparse/hybrid modes log a warning and fall back to dense search.
    """

    def __init__(self, search_type: str, k: int) -> None:
        self.search_type = search_type
        self.k = k
        self._client = QdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection_name
        self._embeddings = get_embeddings()
        # Cache dimensions for truncation
        test_vec = self._embeddings.embed_query("test")
        self._full_dim = len(test_vec)
        self._target_dim = get_target_dim(self._full_dim)

    @observe(name="retrieval")
    def invoke(self, query: str, **_kwargs: Any) -> list[Document]:
        langfuse_context.update_current_observation(
            input={"query": query, "search_type": self.search_type, "k": self.k},
        )
        if self.search_type == "dense":
            docs = self._dense_search(query)
        elif self.search_type == "sparse":
            docs = self._sparse_search(query)
        elif self.search_type == "hybrid":
            docs = self._hybrid_search(query)
        else:
            msg = f"Unknown search_type: {self.search_type}"
            raise ValueError(msg)

        # Top-score is the most useful debugging signal alongside count;
        # full chunk text would blow up Langfuse storage so we surface
        # only the source/category metadata.
        sources = [d.metadata.get("source") for d in docs[:5] if d.metadata]
        langfuse_context.update_current_observation(
            output={"chunks_count": len(docs), "preview_sources": sources},
        )
        return docs

    # ── dense ────────────────────────────────────────────────

    def _dense_search(self, query: str) -> list[Document]:
        vector = self._embeddings.embed_query(query)
        vector = truncate_vector(vector, self._target_dim, self._full_dim)
        start = time.time()
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=self.k,
        )
        RAG_RETRIEVAL_DURATION.labels(search_type="dense").observe(time.time() - start)
        points = results.points
        docs = [_point_to_document(p) for p in points]
        RAG_CHUNKS_RETRIEVED.labels(search_type="dense").observe(len(docs))
        if points:
            RAG_TOP_SCORE.labels(search_type="dense").observe(float(points[0].score))
        RAG_REQUESTS.labels(search_type="dense").inc()
        logger.debug("[Retriever] dense search returned %d docs", len(docs))
        return docs

    # ── sparse (BM25) ───────────────────────────────────────

    def _sparse_search(self, query: str) -> list[Document]:
        sparse_vec = embed_sparse(query)
        start = time.time()
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=self.k,
            )
        except Exception:
            logger.warning(
                "[Retriever] Sparse search failed (collection may lack BM25 vectors). "
                "Falling back to dense search.",
                exc_info=True,
            )
            RAG_FALLBACKS.labels(from_mode="sparse", to_mode="dense").inc()
            return self._dense_search(query)
        RAG_RETRIEVAL_DURATION.labels(search_type="sparse").observe(time.time() - start)
        points = results.points
        docs = [_point_to_document(p) for p in points]
        RAG_CHUNKS_RETRIEVED.labels(search_type="sparse").observe(len(docs))
        if points:
            RAG_TOP_SCORE.labels(search_type="sparse").observe(float(points[0].score))
        RAG_REQUESTS.labels(search_type="sparse").inc()
        logger.debug("[Retriever] sparse search returned %d docs", len(docs))
        return docs

    # ── hybrid (native Qdrant RRF fusion via prefetch) ──────

    def _hybrid_search(self, query: str) -> list[Document]:
        """Dense + sparse search fused with Qdrant-native RRF.

        Uses the ``prefetch`` API (Qdrant >= 1.7): two prefetch queries
        (dense and sparse) are executed server-side and merged via
        Reciprocal Rank Fusion before returning top-k results.

        Falls back to dense-only if sparse vectors are unavailable.
        """
        dense_vector = self._embeddings.embed_query(query)
        dense_vector = truncate_vector(dense_vector, self._target_dim, self._full_dim)
        sparse_vector = embed_sparse(query)

        start = time.time()
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=self.k,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=self.k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=self.k,
            )
        except Exception:
            logger.warning(
                "[Retriever] Hybrid search failed (collection may lack BM25 vectors). "
                "Falling back to dense search.",
                exc_info=True,
            )
            RAG_FALLBACKS.labels(from_mode="hybrid", to_mode="dense").inc()
            return self._dense_search(query)
        RAG_RETRIEVAL_DURATION.labels(search_type="hybrid").observe(time.time() - start)
        points = results.points
        docs = [_point_to_document(p) for p in points]
        RAG_CHUNKS_RETRIEVED.labels(search_type="hybrid").observe(len(docs))
        if points:
            RAG_TOP_SCORE.labels(search_type="hybrid").observe(float(points[0].score))
        RAG_REQUESTS.labels(search_type="hybrid").inc()
        logger.debug("[Retriever] hybrid RRF search returned %d docs", len(docs))
        return docs


# ── public factory ───────────────────────────────────────────


def get_retriever() -> MultiModeRetriever:
    """Return the module-level retriever singleton (lazy init).

    Always returns ``MultiModeRetriever`` so the retrieval-stage metrics
    (``rag_retrieval_duration_seconds``, ``rag_top_score``,
    ``rag_chunks_retrieved``, ``rag_requests``) actually fire — the
    LangChain ``VectorStoreRetriever`` short-circuit that previously
    handled the dense+reranker path bypassed all of them.
    ``_dense_search`` calls ``truncate_vector`` which is a no-op when
    ``embedding_dimension is None``, so dropping the short-circuit
    costs nothing.
    """
    global _retriever  # noqa: PLW0603
    if _retriever is not None:
        return _retriever

    search_type = settings.search_type
    k = settings.retriever_initial_k if settings.reranker_enabled else settings.retriever_top_k

    _retriever = MultiModeRetriever(search_type=search_type, k=k)

    logger.info(
        "[Retriever] Initialized (mode=%s, k=%d, reranker=%s)",
        search_type,
        k,
        settings.reranker_enabled,
    )
    return _retriever
