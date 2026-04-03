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
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from hse_prom_prog.config import settings
from hse_prom_prog.rag.embeddings import get_embeddings, get_target_dim, truncate_vector
from hse_prom_prog.rag.ingest import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from hse_prom_prog.rag.sparse import embed_sparse

logger = logging.getLogger(__name__)

# Module-level singletons
_vector_store: QdrantVectorStore | None = None
_retriever: MultiModeRetriever | VectorStoreRetriever | None = None


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

    def invoke(self, query: str, **_kwargs: Any) -> list[Document]:
        if self.search_type == "dense":
            return self._dense_search(query)
        if self.search_type == "sparse":
            return self._sparse_search(query)
        if self.search_type == "hybrid":
            return self._hybrid_search(query)
        msg = f"Unknown search_type: {self.search_type}"
        raise ValueError(msg)

    # ── dense ────────────────────────────────────────────────

    def _dense_search(self, query: str) -> list[Document]:
        vector = self._embeddings.embed_query(query)
        vector = truncate_vector(vector, self._target_dim, self._full_dim)
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=self.k,
        )
        docs = [_point_to_document(p) for p in results.points]
        logger.debug("[Retriever] dense search returned %d docs", len(docs))
        return docs

    # ── sparse (BM25) ───────────────────────────────────────

    def _sparse_search(self, query: str) -> list[Document]:
        sparse_vec = embed_sparse(query)
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
            return self._dense_search(query)
        docs = [_point_to_document(p) for p in results.points]
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
            return self._dense_search(query)
        docs = [_point_to_document(p) for p in results.points]
        logger.debug("[Retriever] hybrid RRF search returned %d docs", len(docs))
        return docs


# ── public factory ───────────────────────────────────────────


def get_retriever() -> MultiModeRetriever | VectorStoreRetriever:
    """Return the module-level retriever singleton (lazy init).

    When ``search_type`` is "dense" and reranker is enabled, falls back
    to the LangChain ``VectorStoreRetriever`` with ``initial_k`` to keep
    the existing reranker pipeline unchanged.  Otherwise returns a
    ``MultiModeRetriever``.
    """
    global _retriever  # noqa: PLW0603
    if _retriever is not None:
        return _retriever

    search_type = settings.search_type
    k = settings.retriever_initial_k if settings.reranker_enabled else settings.retriever_top_k

    if search_type == "dense" and settings.reranker_enabled:
        # Reranker pipeline expects LangChain retriever with initial_k
        store = get_vector_store()
        _retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    else:
        _retriever = MultiModeRetriever(search_type=search_type, k=k)

    logger.info(
        "[Retriever] Initialized (mode=%s, k=%d, reranker=%s)",
        search_type,
        k,
        settings.reranker_enabled,
    )
    return _retriever
