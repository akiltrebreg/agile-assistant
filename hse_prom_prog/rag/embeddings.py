"""Shared embedding utilities: model creation, truncation, renormalization.

When ``EMBEDDING_DIMENSION`` is set, dense vectors are truncated to that
size and L2-renormalized.  This is required for Matryoshka-trained models
(e.g. ``Alibaba-NLP/gte-multilingual-base``) and can also be used as a
naive-truncation baseline for non-MRL models like ``intfloat/multilingual-e5-base``.

Both ``ingest.py`` and ``retriever.py`` call the functions here so that
document vectors and query vectors always have the same dimensionality.
"""

from __future__ import annotations

import logging

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def truncate_and_normalize(vector: list[float], dim: int) -> list[float]:
    """Truncate *vector* to *dim* dimensions and L2-renormalize.

    After truncation the vector norm changes, so renormalization is
    critical for cosine similarity to remain meaningful.
    """
    v = np.array(vector[:dim], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def get_target_dim(full_dim: int) -> int:
    """Return the effective embedding dimension.

    If ``EMBEDDING_DIMENSION`` is configured, return that value;
    otherwise return the model's native *full_dim*.
    """
    if settings.embedding_dimension is not None:
        return settings.embedding_dimension
    return full_dim


def truncate_vectors(
    vectors: list[list[float]], target_dim: int, full_dim: int
) -> list[list[float]]:
    """Truncate a batch of vectors if *target_dim* < *full_dim*.

    Returns the original list unchanged when no truncation is needed.
    """
    if target_dim >= full_dim:
        return vectors
    return [truncate_and_normalize(v, target_dim) for v in vectors]


def truncate_vector(vector: list[float], target_dim: int, full_dim: int) -> list[float]:
    """Truncate a single vector if *target_dim* < *full_dim*."""
    if target_dim >= full_dim:
        return vector
    return truncate_and_normalize(vector, target_dim)
