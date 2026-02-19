"""BM25 keyword search index built from Qdrant collection documents.

The index is loaded once from Qdrant (via scroll) and cached as a singleton.
Provides keyword-based search complementary to dense vector search.
"""

import logging
import re

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton
_bm25_index: "BM25Index | None" = None


class BM25Index:
    """BM25 keyword search index over knowledge-base documents.

    Attributes:
        _documents: All documents loaded from Qdrant.
        _bm25: BM25Okapi index built from tokenized documents.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("[BM25] Index built with %d documents", len(documents))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text: lowercase + split on non-alphanumeric (keeps Russian)."""
        return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", text.lower())

    def search(
        self,
        query: str,
        k: int = 4,
        category: str | None = None,
    ) -> list[Document]:
        """Search for relevant documents using BM25 scoring.

        Args:
            query: Search query string.
            k: Number of top results to return.
            category: Optional metadata category filter.

        Returns:
            Top-k documents sorted by BM25 score.
        """
        scores = self._bm25.get_scores(self._tokenize(query))

        scored = [
            (scores[i], doc)
            for i, doc in enumerate(self._documents)
            if not category or doc.metadata.get("category") == category
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


def _build_bm25_index() -> BM25Index:
    """Build BM25 index by scrolling all documents from Qdrant."""
    client = QdrantClient(url=settings.qdrant_url)
    collection = settings.qdrant_collection_name

    if not client.collection_exists(collection):
        msg = f"Collection '{collection}' does not exist"
        raise RuntimeError(msg)

    documents: list[Document] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )
        if offset is None:
            break

    logger.info("[BM25] Loaded %d documents from Qdrant", len(documents))
    return BM25Index(documents)


def get_bm25_index() -> BM25Index:
    """Return the module-level BM25 index singleton (lazy init)."""
    global _bm25_index  # noqa: PLW0603
    if _bm25_index is None:
        _bm25_index = _build_bm25_index()
    return _bm25_index
