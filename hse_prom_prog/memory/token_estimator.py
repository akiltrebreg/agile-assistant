"""Char-based token estimator.

Heuristic: Russian/English text on BPE tokenisers (avibe-gptq-8bit, Qwen3)
yields ~2.5-3.5 chars per token; we use the conservative ``3`` so we
never *underestimate* token count when gating against a budget.
"""

CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    """Return a conservative upper-bound estimate of tokens in ``text``.

    Args:
        text: Input text to measure.

    Returns:
        Estimated token count, ``0`` for empty input and at least ``1``
        for any non-empty string.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)
