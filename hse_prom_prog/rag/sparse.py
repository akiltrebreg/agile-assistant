"""Sparse embedding helper: fastembed BM25 or BGE-M3 learned sparse.

Mode is controlled by ``EMBEDDING_SPARSE_MODEL`` env variable:

- ``None`` (default) — fastembed ``Qdrant/bm25`` (statistical BM25)
- ``"BAAI/bge-m3"`` — FlagEmbedding learned sparse vectors

Both modes produce ``qdrant_client.models.SparseVector`` objects that
are stored in the same Qdrant sparse vector field.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client.models import SparseVector

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)

_model: Any = None
_mode: str | None = None

_ENCODE_KWARGS: dict[str, bool] = {
    "return_sparse": True,
    "return_dense": False,
    "return_colbert_vecs": False,
}


# ── fastembed BM25 ───────────────────────────────────────────


def _init_fastembed() -> Any:
    """Load the fastembed ``Qdrant/bm25`` statistical BM25 model.

    Returns:
        Loaded ``SparseTextEmbedding`` instance ready for ``.embed()``.
    """
    from fastembed import SparseTextEmbedding  # noqa: PLC0415

    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    logger.info("[Sparse] Loaded fastembed BM25 (Qdrant/bm25)")
    return model


def _embed_fastembed(text: str) -> SparseVector:
    """Encode a single text via fastembed BM25.

    Args:
        text: Text to encode.

    Returns:
        Qdrant ``SparseVector`` with statistical BM25 weights.
    """
    result = next(iter(_model.embed([text])))
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def _embed_fastembed_batch(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of texts via fastembed BM25.

    Args:
        texts: Texts to encode.

    Returns:
        List of ``SparseVector`` objects, one per input text.
    """
    return [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in _model.embed(texts)
    ]


# ── BGE-M3 learned sparse ───────────────────────────────────


def _init_bgem3() -> Any:
    """Load the BGE-M3 learned-sparse FlagEmbedding model on CPU.

    Returns:
        Loaded ``BGEM3FlagModel`` configured for fp32 CPU inference.
    """
    from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

    model = BGEM3FlagModel(
        settings.embedding_sparse_model,
        use_fp16=False,
        device="cpu",
    )
    logger.info(
        "[Sparse] Loaded BGE-M3 learned sparse (%s)",
        settings.embedding_sparse_model,
    )
    return model


def _lexical_to_sparse(weights: dict[int, float]) -> SparseVector:
    """Convert a BGE-M3 ``lexical_weights`` dict to a Qdrant ``SparseVector``.

    Args:
        weights: Mapping of token id to learned weight.

    Returns:
        ``SparseVector`` with sorted indices and matching float values
        (empty when ``weights`` is empty).
    """
    if not weights:
        return SparseVector(indices=[], values=[])
    indices = sorted(weights.keys())
    values = [float(weights[i]) for i in indices]
    return SparseVector(indices=indices, values=values)


def _embed_bgem3(text: str) -> SparseVector:
    """Encode a single text via BGE-M3 learned sparse.

    Args:
        text: Text to encode.

    Returns:
        Qdrant ``SparseVector`` derived from the model's lexical weights.
    """
    output = _model.encode([text], **_ENCODE_KWARGS)
    return _lexical_to_sparse(output["lexical_weights"][0])


def _embed_bgem3_batch(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of texts via BGE-M3 learned sparse.

    Args:
        texts: Texts to encode.

    Returns:
        List of ``SparseVector`` objects, one per input text.
    """
    output = _model.encode(texts, **_ENCODE_KWARGS)
    return [_lexical_to_sparse(w) for w in output["lexical_weights"]]


# ── public API ───────────────────────────────────────────────


def _ensure_model() -> None:
    """Load the right sparse model on first use, switching modes lazily.

    Reloads the model only when ``settings.embedding_sparse_model``
    selects a different mode than the cached one.
    """
    global _model, _mode  # noqa: PLW0603
    target_mode = "bgem3" if settings.embedding_sparse_model else "fastembed"
    if _model is not None and _mode == target_mode:
        return
    _model = _init_bgem3() if target_mode == "bgem3" else _init_fastembed()
    _mode = target_mode


def embed_sparse(text: str) -> SparseVector:
    """Encode a single text into a Qdrant ``SparseVector``.

    Dispatches to BGE-M3 learned sparse or fastembed BM25 based on
    ``settings.embedding_sparse_model``.

    Args:
        text: Text to encode.

    Returns:
        Sparse vector compatible with the Qdrant sparse field.
    """
    _ensure_model()
    return _embed_bgem3(text) if _mode == "bgem3" else _embed_fastembed(text)


def embed_sparse_batch(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of texts into Qdrant ``SparseVector`` objects.

    Args:
        texts: Texts to encode.

    Returns:
        List of sparse vectors aligned with the input order.
    """
    _ensure_model()
    return _embed_bgem3_batch(texts) if _mode == "bgem3" else _embed_fastembed_batch(texts)
