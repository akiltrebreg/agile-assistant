"""Streamlit chat UI for Agile AI Assistant.

This is the entrypoint: `streamlit run streamlit_app/app.py`

The app is a thin client — it only talks to FastAPI over HTTP.
No direct access to Celery, PostgreSQL, or vLLM.

Session model:
  * ``user_id``       — stable UUID pinned to ``st.query_params["uid"]``.
                        Survives page refresh within the same tab.
  * ``conversation_id`` — optional; pinned to ``st.query_params`` while
                        in use, cleared when the user starts a new chat.
  * ``messages``      — local copy of the transcript for fast repaint.
                        Rebuilt from the API on page load.
"""

import time

import streamlit as st

from streamlit_app.api_client import APIClient
from streamlit_app.auth import (
    current_conversation_id,
    get_or_create_user_id,
    pin_conversation_id,
)
from streamlit_app.components.result import (
    render_error,
    render_result,
    render_task_details,
    render_timeout,
)
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.config import POLL_INTERVAL_SEC, POLL_TIMEOUT_SEC

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Agile AI Assistant",
    page_icon="\U0001f916",
    layout="centered",
)

# ── Load custom CSS from nginx-served static files ───────────
st.markdown(
    '<link rel="stylesheet" href="/static/style.css">',
    unsafe_allow_html=True,
)

# ── Session state init ───────────────────────────────────────
if "client" not in st.session_state:
    st.session_state.client = APIClient()
client: APIClient = st.session_state.client

# Stable user id — written into ?uid=... so a refresh keeps identity.
if "user_id" not in st.session_state:
    st.session_state.user_id = get_or_create_user_id()
user_id: str = st.session_state.user_id


def _load_messages_from_api(conversation_id: str) -> list[dict]:
    """Fetch the full transcript and shape it for st.chat_message."""
    transcript = client.get_messages(conversation_id)
    return [{"role": msg["role"], "content": msg["content"]} for msg in transcript]


# Restore conversation after refresh: URL is the source of truth on boot.
if "conversation_id" not in st.session_state:
    restored = current_conversation_id()
    st.session_state.conversation_id = restored
    if restored:
        st.session_state.messages = _load_messages_from_api(restored)
    else:
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def _start_new_chat() -> None:
    """Clear local + URL state and force a rerun."""
    st.session_state.conversation_id = None
    st.session_state.messages = []
    pin_conversation_id(None)
    st.rerun()


def _switch_to_conversation(conversation_id: str) -> None:
    """Switch the chat to another conversation (sidebar click)."""
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = _load_messages_from_api(conversation_id)
    pin_conversation_id(conversation_id)
    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────
render_sidebar(
    client,
    user_id=user_id,
    current_conversation_id=st.session_state.conversation_id,
    on_new_chat=_start_new_chat,
    on_select=_switch_to_conversation,
)

# ── Chat title ───────────────────────────────────────────────
st.title("Agile AI Assistant")
st.caption("Задайте вопрос о Jira-задаче — например, «Расскажи о задаче AL-38787»")

# ── Render chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ───────────────────────────────────────────────
if prompt := st.chat_input("Введите запрос..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response with progress tracking
    with (
        st.chat_message("assistant"),
        st.status("Обрабатываю запрос...", expanded=True) as status_box,
    ):
        try:
            # 1. Submit task (carry memory identifiers)
            st.write("Отправляю задачу в очередь...")
            create_resp = client.submit_task(
                prompt,
                conversation_id=st.session_state.conversation_id,
                user_id=user_id,
            )
            task_id = create_resp["task_id"]
            new_conv_id = create_resp.get("conversation_id")
            # Pin the conversation so a refresh restores this chat. The
            # very first message of a fresh chat also lands here.
            if new_conv_id and new_conv_id != st.session_state.conversation_id:
                st.session_state.conversation_id = new_conv_id
                pin_conversation_id(new_conv_id)
            st.write(f"Задача создана: `{task_id}`")

            # 2. Poll for result
            st.write("Ожидаю результат...")
            deadline = time.time() + POLL_TIMEOUT_SEC
            task_data: dict = {}

            processing_shown = False
            while time.time() < deadline:
                task_data = client.get_task(task_id)
                task_status = task_data.get("status", "")

                if task_status == "COMPLETED":
                    status_box.update(
                        label="Готово",
                        state="complete",
                        expanded=False,
                    )
                    break

                if task_status == "FAILED":
                    status_box.update(
                        label="Ошибка",
                        state="error",
                        expanded=False,
                    )
                    break

                # Still processing
                if task_status == "PROCESSING" and not processing_shown:
                    st.write("Задача обрабатывается...")
                    processing_shown = True
                time.sleep(POLL_INTERVAL_SEC)
            else:
                # Timeout
                status_box.update(
                    label="Таймаут",
                    state="error",
                    expanded=False,
                )
                timeout_msg = render_timeout(task_id)
                st.warning(timeout_msg)
                st.session_state.messages.append({"role": "assistant", "content": timeout_msg})
                st.stop()

            # 3. Display result
            final_status = task_data.get("status", "")
            if final_status == "COMPLETED":
                answer = render_result(task_data)
                st.markdown(answer)
                render_task_details(task_data)
            else:
                answer = render_error(task_data)
                st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as exc:
            status_box.update(
                label="Ошибка соединения",
                state="error",
                expanded=False,
            )
            err_text = f"Не удалось связаться с API:\n\n`{exc}`"
            st.error(err_text)
            st.session_state.messages.append({"role": "assistant", "content": err_text})
