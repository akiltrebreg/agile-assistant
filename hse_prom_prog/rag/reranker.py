"""Cross-encoder reranker for second-stage relevance scoring.

Loads a CrossEncoder model once as a module-level singleton (lazy init).
Provides reranking, threshold filtering, and "Lost in the Middle" reordering.

The model snapshot is downloaded once from Yandex Cloud Object Storage on
first use (``s3://{s3_models_bucket}/{s3_models_path}/{reranker_model}/``)
and cached under ``settings.embedding_model_cache_dir`` — the same
volume + the same idiom used for the embedding model snapshot. The
HuggingFace Hub is never contacted unless ``s3_models_bucket`` is unset
(back-compat fallback for ad-hoc local runs without Yandex creds).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from hse_prom_prog.config import settings
from hse_prom_prog.metrics import RAG_CHUNKS_AFTER_RERANKER, RAG_RERANKER_DURATION
from hse_prom_prog.rag.embeddings import _HF_SNAPSHOT_MARKER, _download_model_from_s3
from hse_prom_prog.tracing import langfuse_context, observe

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
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model id or local path of the
                CrossEncoder used for relevance scoring.
            threshold: Minimum score below which documents are dropped.
            top_n: Maximum number of documents kept after sorting.
        """
        self._model = CrossEncoder(model_name)
        self._threshold = threshold
        self._top_n = top_n
        logger.info(
            "[Reranker] Initialized (model=%s, threshold=%.2f, top_n=%d)",
            model_name,
            threshold,
            top_n,
        )

    @observe(name="reranker")
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
        langfuse_context.update_current_observation(
            input={"chunks_count": len(documents), "threshold": self._threshold},
        )
        if not documents:
            langfuse_context.update_current_observation(output={"chunks_after": 0, "scores": []})
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
            langfuse_context.update_current_observation(
                output={"chunks_after": 0, "scores": [], "reason": "below_threshold"},
            )
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
        langfuse_context.update_current_observation(
            output={
                "chunks_after": len(scored_docs),
                "scores": [round(float(s), 4) for s, _ in scored_docs],
            },
        )

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


def ensure_reranker_model_downloaded() -> str:
    """Return the local reranker path, downloading from S3 first if needed.

    Mirrors :func:`hse_prom_prog.rag.embeddings.ensure_embedding_model_downloaded`:
    when ``s3_models_bucket`` is set, the snapshot at
    ``s3://{s3_models_bucket}/{s3_models_path}/{reranker_model}/`` is
    mirrored into ``embedding_model_cache_dir`` and the local path is
    returned. The presence of ``config.json`` is the cache-hit marker —
    second call short-circuits without touching S3.

    Returns:
        Local filesystem path to the reranker directory if S3 is
        configured. Falls back to ``settings.reranker_model`` (treated
        as a HuggingFace Hub ID) when ``s3_models_bucket`` is unset.

    Raises:
        RuntimeError: If the S3 download finishes without producing the
            ``config.json`` snapshot marker, indicating an incomplete
            model upload at the configured prefix.
    """
    bucket = settings.s3_models_bucket
    if not bucket:
        logger.warning(
            "[Reranker] s3_models_bucket is empty — falling back to HuggingFace Hub ID %r",
            settings.reranker_model,
        )
        return settings.reranker_model

    cache_root = Path(settings.embedding_model_cache_dir)
    model_dir = cache_root / settings.reranker_model
    marker = model_dir / _HF_SNAPSHOT_MARKER

    if marker.exists():
        logger.info("[Reranker] Model snapshot present at %s — skipping S3 download", model_dir)
        return str(model_dir)

    prefix = f"{settings.s3_models_path.rstrip('/')}/{settings.reranker_model}".lstrip("/")
    _download_model_from_s3(bucket=bucket, prefix=prefix, local_dir=model_dir)

    if not marker.exists():
        msg = (
            f"S3 download finished but {marker} is missing — "
            f"verify s3://{bucket}/{prefix}/ contains a HuggingFace snapshot"
        )
        raise RuntimeError(msg)

    return str(model_dir)


def _build_reranker() -> Reranker:
    """Build a ``Reranker`` from application settings.

    Resolves the model snapshot via :func:`ensure_reranker_model_downloaded`
    so the local path is passed to ``CrossEncoder`` — no Hub lookup
    happens at runtime when S3 is configured.

    Returns:
        Reranker configured with the resolved model path,
        ``settings.reranker_threshold`` and ``settings.reranker_top_n``.
    """
    return Reranker(
        model_name=ensure_reranker_model_downloaded(),
        threshold=settings.reranker_threshold,
        top_n=settings.reranker_top_n,
    )


def get_reranker() -> Reranker:
    """Return the module-level ``Reranker`` singleton, creating it lazily.

    Returns:
        Cached ``Reranker`` shared across the module.
    """
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        _reranker = _build_reranker()
    return _reranker
