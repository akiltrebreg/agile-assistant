"""Deterministic text-to-sparse-vector tokenizer for BM25 search.

Converts raw text into a Qdrant ``SparseVector`` (indices = token hashes,
values = term frequencies).  Designed as a lightweight, dependency-free
alternative to fastembed when a pre-trained BM25 model is not needed.

Pipeline: lowercase → split on non-alphanumeric (keeps Cyrillic) →
hash each token via ``zlib.crc32`` → count frequencies.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from qdrant_client.models import SparseVector

# Matches Latin, Cyrillic, digits, and underscores
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into tokens (Latin + Cyrillic).

    Args:
        text: Raw input text.

    Returns:
        List of lowercase tokens — runs of Latin or Cyrillic letters,
        digits, or underscores (see ``_TOKEN_RE``).
    """
    return _TOKEN_RE.findall(text.lower())


def _token_hash(token: str) -> int:
    """Compute a deterministic non-negative 32-bit hash for ``token``.

    Uses ``zlib.crc32`` so the mapping is stable across processes,
    Python versions, and platforms — required because the same token
    must collide to the same sparse-vector index in every ingest run.

    Args:
        token: Token string to hash.

    Returns:
        Non-negative 32-bit integer hash.
    """
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def tokenize_to_sparse_vector(text: str) -> SparseVector:
    """Tokenize *text* and return a Qdrant SparseVector.

    - **Deterministic**: same text always produces the same vector.
    - **Cyrillic-friendly**: keeps Russian letters alongside Latin.
    - **No stop-word removal**: BM25 IDF modifier in Qdrant already
      down-weights frequent tokens at query time.

    Args:
        text: Raw document or query text.

    Returns:
        SparseVector with CRC32 token hashes as indices and
        term frequencies as values.
    """
    tokens = _tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    freq = Counter(_token_hash(t) for t in tokens)
    indices = sorted(freq)
    values = [float(freq[i]) for i in indices]
    return SparseVector(indices=indices, values=values)


def tokenize_batch(texts: list[str]) -> list[SparseVector]:
    """Tokenize a batch of texts into ``SparseVector`` objects.

    Args:
        texts: Texts to tokenize.

    Returns:
        Sparse vectors aligned with the input order.
    """
    return [tokenize_to_sparse_vector(t) for t in texts]
