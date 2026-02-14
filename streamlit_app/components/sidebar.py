"""Sidebar component: logo, API status indicator, clear-chat button."""

import streamlit as st

from streamlit_app.api_client import APIClient


def render_sidebar(client: APIClient) -> None:
    """Render the sidebar with status and controls."""
    with st.sidebar:
        st.title("Agile AI Assistant")
        st.caption("LangGraph + vLLM + PostgreSQL")

        st.divider()

        # API health indicator
        is_healthy = client.health()
        if is_healthy:
            st.success("API: Online", icon="\u2705")
        else:
            st.error("API: Offline", icon="\u274c")

        st.divider()

        # Clear chat
        if st.button("Очистить чат", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
