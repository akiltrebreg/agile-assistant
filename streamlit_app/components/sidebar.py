"""Sidebar: logo, API status, new-chat button, conversation history."""

from collections.abc import Callable
from datetime import UTC, datetime

import streamlit as st

from streamlit_app.api_client import APIClient

_TITLE_LIMIT = 50
_MAX_CONVERSATIONS = 20


def render_sidebar(
    client: APIClient,
    *,
    user_id: str | None = None,
    current_conversation_id: str | None = None,
    on_new_chat: Callable[[], None] | None = None,
    on_select: Callable[[str], None] | None = None,
) -> None:
    """Render the chat sidebar (logo, new-chat button, history list).

    Backwards-compatible: when the memory-layer kwargs are omitted the
    sidebar still shows the logo + "Clear chat" and nothing else — useful
    for tests or a stripped-down anonymous mode.

    Args:
        client: API client used to fetch the conversation list.
        user_id: External user id; ``None`` enables legacy mode.
        current_conversation_id: Active conversation, highlighted in the
            list. ``None`` means no conversation is currently selected.
        on_new_chat: Callback invoked when "Новый диалог" is clicked.
        on_select: Callback invoked with the conversation id when an
            entry in the history list is clicked.
    """
    with st.sidebar:
        _render_header()
        st.divider()

        if user_id:
            _render_conversation_controls(
                client,
                user_id=user_id,
                current_conversation_id=current_conversation_id,
                on_new_chat=on_new_chat,
                on_select=on_select,
            )
            return

        # Legacy mode — no memory layer engaged.
        if st.button("Очистить чат", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def _render_header() -> None:
    """Render the inline-SVG header logo at the top of the sidebar."""
    st.markdown(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 50"'
        ' style="width:100%;max-width:200px;">'
        '<rect width="200" height="50" rx="8" fill="#4A90D9"/>'
        '<text x="100" y="33" font-family="Arial,sans-serif"'
        ' font-size="20" font-weight="bold"'
        ' fill="white" text-anchor="middle">Agile AI Assistant</text>'
        "</svg>",
        unsafe_allow_html=True,
    )


def _render_conversation_controls(
    client: APIClient,
    *,
    user_id: str,
    current_conversation_id: str | None,
    on_new_chat: Callable[[], None] | None,
    on_select: Callable[[str], None] | None,
) -> None:
    """Render the "new chat" button and the conversation history list.

    Side effect: triggers ``st.rerun()`` via the fallback path when no
    ``on_new_chat`` callback is supplied.

    Args:
        client: API client used to fetch the conversation list.
        user_id: External user id whose conversations to display.
        current_conversation_id: Active conversation id (rendered with
            ``"secondary"`` style instead of ``"tertiary"``).
        on_new_chat: Callback invoked when the new-chat button is hit.
        on_select: Callback invoked with the conversation id on clicks
            in the history list.
    """
    # "Новый диалог" is the only chat-control button — same mental model
    # as ChatGPT / Claude / Gemini. Closing the previous conversation
    # (so the worker schedules summarisation + profile refresh) happens
    # inside the on_new_chat callback rather than as a separate button.
    if st.button(
        "➕ Новый диалог",
        use_container_width=True,
        type="primary",
    ):
        if on_new_chat is not None:
            on_new_chat()
        else:
            # Fallback: clear local state, leave URL alone.
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    st.divider()
    st.markdown("**История диалогов**")

    conversations = client.list_conversations(user_id, limit=_MAX_CONVERSATIONS)
    if not conversations:
        st.caption("Пока нет сохранённых диалогов.")
        return

    now = datetime.now(UTC)
    for conv in conversations:
        label = _format_conversation_label(conv, now)
        button_type = "secondary" if conv["id"] == current_conversation_id else "tertiary"
        if (
            st.button(
                label,
                key=f"conv-{conv['id']}",
                use_container_width=True,
                type=button_type,
            )
            and on_select is not None
        ):
            on_select(conv["id"])


def _format_conversation_label(conv: dict, now: datetime) -> str:
    """Build a single-line button label: truncated title plus relative time.

    Args:
        conv: Conversation dict with ``title`` and ``updated_at``.
        now: Reference time used to compute the relative-time suffix.

    Returns:
        Label like ``"Some title · 5 мин"`` (or just the title when no
        ``updated_at`` is available).
    """
    title = (conv.get("title") or "Без названия").strip()
    if len(title) > _TITLE_LIMIT:
        title = title[:_TITLE_LIMIT].rstrip() + "…"

    updated_at = _parse_datetime(conv.get("updated_at"))
    time_part = _relative_time_ru(updated_at, now) if updated_at else ""
    return f"{title} · {time_part}" if time_part else title


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string into a ``datetime``.

    Args:
        value: ISO-8601 timestamp, possibly ``None`` or malformed.

    Returns:
        Parsed ``datetime``, or ``None`` if the input is missing or not
        a valid ISO-8601 string.
    """
    if not value:
        return None
    try:
        # Python 3.11+ datetime.fromisoformat accepts trailing "Z" directly.
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _relative_time_ru(then: datetime, now: datetime) -> str:  # noqa: PLR0911
    """Return a human-readable delta in Russian.

    Examples include ``"только что"``, ``"5 мин"``, ``"вчера"`` and
    ``"3 дн."``; deltas older than a month fall back to ``DD.MM``.

    Args:
        then: Earlier moment (assumed UTC if naive).
        now: Reference (current) moment.

    Returns:
        Russian-language relative-time label.
    """
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    delta = now - then
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "только что"
    one_minute = 60
    one_hour = 3600
    one_day = 86400
    two_days = 2 * one_day
    one_week = 7 * one_day
    one_month = 30 * one_day
    if seconds < one_minute:
        return "только что"
    if seconds < one_hour:
        return f"{seconds // one_minute} мин"
    if seconds < one_day:
        return f"{seconds // one_hour} ч"
    if seconds < two_days:
        return "вчера"
    if seconds < one_week:
        return f"{seconds // one_day} дн."
    if seconds < one_month:
        return f"{seconds // one_week} нед."
    return then.strftime("%d.%m")
