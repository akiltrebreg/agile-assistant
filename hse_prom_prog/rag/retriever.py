"""Qdrant retriever: connects to an existing collection and returns relevant docs.

The vector store and retriever are created once and reused across requests.
"""

import logging

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons
_vector_store: QdrantVectorStore | None = None
_retriever: VectorStoreRetriever | None = None

# Number of relevant chunks to return
_TOP_K = 4


def _build_vector_store() -> QdrantVectorStore:
    """Create a QdrantVectorStore backed by the existing collection."""
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

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
    )
    logger.info("[Retriever] Vector store initialized (collection=%s)", collection)
    return store


def get_vector_store() -> QdrantVectorStore:
    """Return the module-level vector store singleton (lazy init)."""
    global _vector_store  # noqa: PLW0603
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store


def get_retriever() -> VectorStoreRetriever:
    """Return the module-level retriever singleton (lazy init)."""
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        store = get_vector_store()
        _retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": _TOP_K},
        )
        logger.info("[Retriever] Retriever initialized (k=%d)", _TOP_K)
    return _retriever
