"""Render a ``ConversationContext`` into a prompt-ready block.

Used by all agents that inject history into their prompts. A dedicated
helper avoids XML-tag drift across agents and keeps the role-label /
language choice in one place.
"""

from hse_prom_prog.models.memory import ConversationContext

_ROLE_LABEL = {"user": "User", "assistant": "Assistant"}


def format_history(ctx: ConversationContext | None) -> str:
    """Render ``ctx`` as a ``<conversation_history>`` block.

    Skips the ``<summary>`` sub-block when the rolling summary is empty.

    Args:
        ctx: Conversation context to render, or ``None``.

    Returns:
        XML-tagged history block, or ``""`` when ``ctx`` is ``None`` or
        has no recent turns.
    """
    if ctx is None:
        return ""

    recent = ctx.get("recent_turns") or []
    if not recent:
        return ""

    parts: list[str] = ["<conversation_history>"]
    summary = ctx.get("summary") or ""
    if summary:
        parts.append(f"<summary>{summary}</summary>")

    parts.append("<recent>")
    for turn in recent:
        label = _ROLE_LABEL.get(turn.get("role", ""), turn.get("role", ""))
        parts.append(f"{label}: {turn.get('content', '')}")
    parts.append("</recent>")
    parts.append("</conversation_history>")
    return "\n".join(parts)
