"""Token-budget-aware text truncation.

Used to fit long assistant replies into the conversation-history slot of
an agent prompt without cutting mid-word.
"""

from agile_assistant.memory.token_estimator import CHARS_PER_TOKEN

ELLIPSIS = " [...]"


def truncate_message(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so that the result stays within ``max_tokens``.

    Cuts at the last whitespace before the budget to avoid breaking a word,
    then appends ``[...]`` as a visual marker. If the text already fits,
    it is returned unchanged.

    Args:
        text: Input text to truncate.
        max_tokens: Hard upper bound on the resulting token count.

    Returns:
        Truncated text, possibly the original ``text`` when it already
        fits, or ``""`` when ``max_tokens`` is non-positive.
    """
    if max_tokens <= 0:
        return ""
    if not text:
        return text

    budget_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= budget_chars:
        return text

    # Reserve space for the ellipsis marker so the total still fits the budget.
    cut_at = budget_chars - len(ELLIPSIS)
    if cut_at <= 0:
        return ELLIPSIS.strip()

    head = text[:cut_at]
    last_space = head.rfind(" ")
    # Only snap to the space if it's not too far back (would waste budget).
    if last_space > cut_at // 2:
        head = head[:last_space]

    return head.rstrip() + ELLIPSIS
