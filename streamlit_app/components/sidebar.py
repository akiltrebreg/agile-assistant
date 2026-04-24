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
    """Render the sidebar.

    Backwards-compatible: when the memory-layer kwargs are omitted the
    sidebar still shows the logo + health + "Clear chat" and nothing else
    — useful for tests or a stripped-down anonymous mode.
    """
    with st.sidebar:
        _render_header(client)
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


def _render_header(client: APIClient) -> None:
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
    st.caption("LangGraph + vLLM + PostgreSQL")

    st.divider()

    if client.health():
        st.success("API: Online", icon="✅")
    else:
        st.error("API: Offline", icon="❌")


def _render_conversation_controls(
    client: APIClient,
    *,
    user_id: str,
    current_conversation_id: str | None,
    on_new_chat: Callable[[], None] | None,
    on_select: Callable[[str], None] | None,
) -> None:
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

    # "Завершить диалог" — only meaningful when there IS an active one.
    # Explicit close triggers summarisation; after that we reuse the
    # same on_new_chat callback to reset local + URL state.
    if current_conversation_id and st.button(
        "Завершить диалог",
        use_container_width=True,
    ):
        client.close_conversation(current_conversation_id)
        if on_new_chat is not None:
            on_new_chat()
        else:
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
    """Title (truncated) + relative time, on a single line for the button."""
    title = (conv.get("title") or "Без названия").strip()
    if len(title) > _TITLE_LIMIT:
        title = title[:_TITLE_LIMIT].rstrip() + "…"

    updated_at = _parse_datetime(conv.get("updated_at"))
    time_part = _relative_time_ru(updated_at, now) if updated_at else ""
    return f"{title} · {time_part}" if time_part else title


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Python 3.11+ datetime.fromisoformat accepts trailing "Z" directly.
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _relative_time_ru(then: datetime, now: datetime) -> str:  # noqa: PLR0911
    """Human-readable delta in Russian: "5 мин", "вчера", "3 дня"."""
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
