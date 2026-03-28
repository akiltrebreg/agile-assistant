"""Shared BM25 sparse embedding helper.

Uses fastembed's ``Qdrant/bm25`` model — the same tokenizer for both
ingestion and retrieval so that sparse vectors are consistent.
"""

import logging

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    global _model  # noqa: PLW0603
    if _model is None:
        _model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("[Sparse] BM25 model loaded (Qdrant/bm25)")
    return _model


def embed_sparse(text: str) -> SparseVector:
    """Encode a single text into a Qdrant SparseVector."""
    model = _get_model()
    result = next(iter(model.embed([text])))
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def embed_sparse_batch(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of texts into Qdrant SparseVectors."""
    model = _get_model()
    return [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in model.embed(texts)
    ]
