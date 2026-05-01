"""Shared embedding utilities: model creation, truncation, renormalization.

The embedding model snapshot is downloaded once from Yandex Cloud Object
Storage on first use (``s3://{s3_models_bucket}/{s3_models_path}/{embedding_model}/``)
and cached under ``settings.embedding_model_cache_dir``. Subsequent calls
detect the existing snapshot via the ``config.json`` marker and skip the
download. Air-gapped deployments only need S3 reachability; the
HuggingFace Hub is never contacted unless ``s3_models_bucket`` is unset.

When ``EMBEDDING_DIMENSION`` is set, dense vectors are truncated to that
size and L2-renormalized.  This is required for Matryoshka-trained models
(e.g. ``Alibaba-NLP/gte-multilingual-base``) and can also be used as a
naive-truncation baseline for non-MRL models like ``intfloat/multilingual-e5-base``.

Both ``ingest.py`` and ``retriever.py`` call the functions here so that
document vectors and query vectors always have the same dimensionality.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from hse_prom_prog.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# HuggingFace marker file — its presence means the snapshot is fully
# materialised on disk. Same idiom the ``download-model`` compose service
# uses for the LLM (docker-compose.yml:112).
_HF_SNAPSHOT_MARKER = "config.json"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding model instance.

    Ensures the model snapshot is present locally (downloads from S3 if
    missing) and passes the local path to ``HuggingFaceEmbeddings`` so
    no Hub lookup happens at runtime.

    Returns:
        Configured ``HuggingFaceEmbeddings`` running on CPU with
        normalized output vectors.
    """
    model_name_or_path = ensure_embedding_model_downloaded()
    return HuggingFaceEmbeddings(
        model_name=str(model_name_or_path),
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_embedding_model_downloaded() -> str:
    """Return the local model path, downloading from S3 first if needed.

    Returns:
        Local filesystem path to the model directory if the S3 path is
        configured. Falls back to ``settings.embedding_model`` (treated
        as a HuggingFace Hub ID) when ``s3_models_bucket`` is unset —
        useful for ad-hoc local runs without S3 credentials.

    Raises:
        RuntimeError: If the S3 download finishes without producing the
            ``config.json`` snapshot marker, indicating an incomplete
            model upload at the configured prefix.
    """
    bucket = settings.s3_models_bucket
    if not bucket:
        # Back-compat fallback: no S3 → let HF resolve the name (will hit
        # the Hub on first use). Pin the log so deploy-time misconfig is
        # visible without grepping for missing downloads.
        logger.warning(
            "[Embeddings] s3_models_bucket is empty — falling back to HuggingFace Hub ID %r",
            settings.embedding_model,
        )
        return settings.embedding_model

    cache_root = Path(settings.embedding_model_cache_dir)
    model_dir = cache_root / settings.embedding_model
    marker = model_dir / _HF_SNAPSHOT_MARKER

    if marker.exists():
        logger.info("[Embeddings] Model snapshot present at %s — skipping S3 download", model_dir)
        return str(model_dir)

    prefix = f"{settings.s3_models_path.rstrip('/')}/{settings.embedding_model}".lstrip("/")
    _download_model_from_s3(bucket=bucket, prefix=prefix, local_dir=model_dir)

    # Sanity-check: after sync the marker MUST exist — otherwise the S3
    # folder is missing config.json and HuggingFace would silently fall
    # back to a Hub lookup, defeating the air-gap guarantee.
    if not marker.exists():
        msg = (
            f"S3 download finished but {marker} is missing — "
            f"verify s3://{bucket}/{prefix}/ contains a HuggingFace snapshot"
        )
        raise RuntimeError(msg)

    return str(model_dir)


def _download_model_from_s3(*, bucket: str, prefix: str, local_dir: Path) -> None:
    """Sync every object under ``s3://{bucket}/{prefix}/`` into ``local_dir``.

    Mirrors the boto3-Yandex-Cloud convention used by ``database.load_csv``
    (creds from env, region default ``ru-central1``, custom endpoint).
    Streams files one at a time to keep memory flat for large snapshots.

    Args:
        bucket: S3 bucket holding the model snapshot.
        prefix: Key prefix for the snapshot directory inside the bucket.
        local_dir: Local destination directory (created if missing).

    Raises:
        RuntimeError: If no objects are found under the given prefix.
    """
    import boto3  # noqa: PLC0415  — heavy import, defer until first use

    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/") + "/"

    session = boto3.session.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
    )
    s3 = session.client("s3", endpoint_url=settings.s3_endpoint)

    logger.info("[Embeddings] Downloading model from s3://%s/%s ...", bucket, prefix)

    paginator = s3.get_paginator("list_objects_v2")
    file_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents: Iterable[dict] = page.get("Contents") or []
        for obj in contents:
            key = obj["Key"]
            relative = key[len(prefix) :]
            # The prefix "directory" itself shows up as a zero-byte key with
            # an empty relative path in some S3 implementations — skip it.
            if not relative or key.endswith("/"):
                continue
            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            file_count += 1

    if file_count == 0:
        msg = f"No objects found under s3://{bucket}/{prefix} — wrong bucket or prefix?"
        raise RuntimeError(msg)

    logger.info("[Embeddings] Downloaded %d file(s) to %s", file_count, local_dir)


def truncate_and_normalize(vector: list[float], dim: int) -> list[float]:
    """Truncate ``vector`` to ``dim`` dimensions and L2-renormalize.

    After truncation the vector norm changes, so renormalization is
    critical for cosine similarity to remain meaningful.

    Args:
        vector: Source embedding to truncate.
        dim: Target dimensionality (must be ``<= len(vector)``).

    Returns:
        New list of floats of length ``dim`` with unit L2 norm
        (or all zeros if the truncated prefix had zero norm).
    """
    v = np.array(vector[:dim], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def get_target_dim(full_dim: int) -> int:
    """Return the effective embedding dimension.

    If ``EMBEDDING_DIMENSION`` is configured, return that value;
    otherwise return the model's native ``full_dim``.

    Args:
        full_dim: The model's native embedding dimensionality.

    Returns:
        Effective dimensionality to use for indexing and querying.
    """
    if settings.embedding_dimension is not None:
        return settings.embedding_dimension
    return full_dim


def truncate_vectors(
    vectors: list[list[float]], target_dim: int, full_dim: int
) -> list[list[float]]:
    """Truncate a batch of vectors when ``target_dim < full_dim``.

    Returns the original list unchanged when no truncation is needed.

    Args:
        vectors: Embeddings produced by the model.
        target_dim: Effective dimensionality after truncation.
        full_dim: Native dimensionality of the model output.

    Returns:
        Either the original list (when no truncation is required) or a
        new list of truncated, L2-renormalized vectors.
    """
    if target_dim >= full_dim:
        return vectors
    return [truncate_and_normalize(v, target_dim) for v in vectors]


def truncate_vector(vector: list[float], target_dim: int, full_dim: int) -> list[float]:
    """Truncate a single vector when ``target_dim < full_dim``.

    Args:
        vector: Single embedding produced by the model.
        target_dim: Effective dimensionality after truncation.
        full_dim: Native dimensionality of the model output.

    Returns:
        The original ``vector`` unchanged, or a truncated and
        L2-renormalized copy when truncation applies.
    """
    if target_dim >= full_dim:
        return vector
    return truncate_and_normalize(vector, target_dim)
