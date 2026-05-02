"""Lightweight identity for the Streamlit client.

We don't have real auth yet — for MVP we embed a stable UUID in
``st.query_params["uid"]``. It survives page refresh (the id stays in
the URL) so the same browser tab talks to the same ``user_profiles``
row across visits. Good enough to drive sidebar + profile injection
without adding a cookie-manager dependency.

When real SSO arrives, only this module needs to change — every other
component reads the result of ``get_or_create_user_id()``.
"""

import uuid

import streamlit as st


def get_or_create_user_id() -> str:
    """Return the current session's user id, creating one if needed.

    Writes the id back into ``st.query_params`` so it persists across
    page reloads. Re-calling is cheap: once the id is in the URL, the
    branch is a single dict lookup.
    """
    existing = st.query_params.get("uid")
    if existing:
        return existing

    fresh = str(uuid.uuid4())
    # Streamlit accepts both dict-style assignment and .update().
    st.query_params["uid"] = fresh
    return fresh


def current_conversation_id() -> str | None:
    """Return the conversation id pinned to the URL, or ``None``.

    Used on app-load to restore a session after refresh. Callers that
    want the in-memory state should read ``st.session_state.conversation_id``
    instead — the URL is a source of truth only at boot time.
    """
    value = st.query_params.get("conversation_id")
    return value or None


def pin_conversation_id(conversation_id: str | None) -> None:
    """Write / clear ``conversation_id`` in ``st.query_params``.

    Passing ``None`` removes it — used by the "New chat" button.
    """
    if conversation_id:
        st.query_params["conversation_id"] = conversation_id
    elif "conversation_id" in st.query_params:
        del st.query_params["conversation_id"]
