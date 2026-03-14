"""Sidebar component: logo, API status indicator, clear-chat button."""

import streamlit as st

from streamlit_app.api_client import APIClient


def render_sidebar(client: APIClient) -> None:
    """Render the sidebar with status and controls."""
    with st.sidebar:
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
