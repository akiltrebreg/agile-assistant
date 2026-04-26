"""Cross-encoder reranker for second-stage relevance scoring.

Loads a CrossEncoder model once as a module-level singleton (lazy init).
Provides reranking, threshold filtering, and "Lost in the Middle" reordering.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from hse_prom_prog.config import settings
from hse_prom_prog.metrics import RAG_CHUNKS_AFTER_RERANKER, RAG_RERANKER_DURATION

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Minimum number of docs to apply lost-in-the-middle reordering
_MIN_DOCS_FOR_REORDER = 2

# Module-level singleton
_reranker: "Reranker | None" = None  # noqa: UP037


class Reranker:
    """Cross-encoder reranker for second-stage document relevance scoring.

    Attributes:
        _model: CrossEncoder model instance.
        _threshold: Minimum score to keep a document.
        _top_n: Maximum number of documents to return after reranking.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float,
        top_n: int,
    ) -> None:
        self._model = CrossEncoder(model_name)
        self._threshold = threshold
        self._top_n = top_n
        logger.info(
            "[Reranker] Initialized (model=%s, threshold=%.2f, top_n=%d)",
            model_name,
            threshold,
            top_n,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Rerank documents by cross-encoder relevance, filter, and reorder.

        Pipeline:
        1. Score each document against the query using CrossEncoder.
        2. Filter out documents below self._threshold.
        3. Take top self._top_n by score.
        4. Apply "Lost in the Middle" reordering.

        Args:
            query: The original user query.
            documents: Candidate documents from first-stage retrieval.

        Returns:
            Reranked, filtered, and reordered list of documents.
        """
        if not documents:
            return []

        start = time.time()
        # Step 1: Score all documents against the query
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        scored_docs = list(zip(scores, documents, strict=True))

        # Step 2: Threshold filter
        scored_docs = [(score, doc) for score, doc in scored_docs if score >= self._threshold]

        if not scored_docs:
            logger.info("[Reranker] All documents below threshold %.2f", self._threshold)
            RAG_RERANKER_DURATION.observe(time.time() - start)
            RAG_CHUNKS_AFTER_RERANKER.observe(0)
            return []

        # Step 3: Sort by score descending, take top_n
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        scored_docs = scored_docs[: self._top_n]

        logger.info(
            "[Reranker] Kept %d docs (scores: %.3f .. %.3f)",
            len(scored_docs),
            scored_docs[0][0],
            scored_docs[-1][0],
        )

        RAG_RERANKER_DURATION.observe(time.time() - start)
        RAG_CHUNKS_AFTER_RERANKER.observe(len(scored_docs))

        # Step 4: "Lost in the Middle" reordering
        return self._lost_in_the_middle_reorder(scored_docs)

    @staticmethod
    def _lost_in_the_middle_reorder(
        scored_docs: list[tuple[float, Document]],
    ) -> list[Document]:
        """Reorder so most relevant docs are at start and end of the list.

        LLMs pay more attention to the beginning and end of their context
        window. This places the highest-scored documents at the edges
        and the lowest-scored in the middle.

        For input sorted desc [best, 2nd, 3rd, 4th, 5th]:
          - Even-ranked (0, 2, 4) fill from the left:  positions 0, 1, 2
          - Odd-ranked  (1, 3)    fill from the right: positions 4, 3
          Result: [best, 3rd, 5th, 4th, 2nd]

        Args:
            scored_docs: (score, doc) tuples, already sorted by score desc.

        Returns:
            Reordered list of Document objects.
        """
        if len(scored_docs) <= _MIN_DOCS_FOR_REORDER:
            return [doc for _, doc in scored_docs]

        docs = [doc for _, doc in scored_docs]
        result: list[Document] = [None] * len(docs)  # type: ignore[list-item]
        left = 0
        right = len(docs) - 1

        for i, doc in enumerate(docs):
            if i % 2 == 0:
                result[left] = doc
                left += 1
            else:
                result[right] = doc
                right -= 1

        return result


def _build_reranker() -> Reranker:
    """Build Reranker from application settings."""
    return Reranker(
        model_name=settings.reranker_model,
        threshold=settings.reranker_threshold,
        top_n=settings.reranker_top_n,
    )


def get_reranker() -> Reranker:
    """Return the module-level Reranker singleton (lazy init)."""
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        _reranker = _build_reranker()
    return _reranker
