"""HTTP client for FastAPI backend.

This is the only module that knows about the API. All other modules
interact with the backend through this client.
"""

import logging
from http import HTTPStatus

import requests

from streamlit_app.config import API_BASE_URL, REQUEST_TIMEOUT_SEC

logger = logging.getLogger(__name__)


class APIClient:
    """Thin wrapper around requests for calling FastAPI endpoints."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        """Initialize the API client.

        Args:
            base_url: FastAPI root URL; trailing slash is stripped.
                Defaults to the value of ``API_BASE_URL``.
        """
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        """Check if the API is reachable.

        Returns:
            True if /health returns 200, False otherwise.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                timeout=REQUEST_TIMEOUT_SEC,
            )
            return resp.ok
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def submit_task(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        """POST /tasks — create a new task.

        Args:
            query: User query string.
            conversation_id: Existing conversation id (``None`` = new).
            user_id: External user id (cookie UUID / SSO).

        Returns:
            Dict with ``task_id``, ``conversation_id``, ``status``, ``message``.

        Raises:
            requests.RequestException: On network / HTTP errors.
        """
        payload: dict = {"query": query}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if user_id:
            payload["user_id"] = user_id
        resp = requests.post(
            f"{self.base_url}/tasks",
            json=payload,
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()

    def get_task(self, task_id: str) -> dict:
        """GET /tasks/{task_id} — poll task status.

        Args:
            task_id: UUID string of the task.

        Returns:
            Full task object (task_id, query, status, result, error, timestamps).

        Raises:
            requests.RequestException: On network / HTTP errors.
        """
        resp = requests.get(
            f"{self.base_url}/tasks/{task_id}",
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Conversations (memory layer)
    # ------------------------------------------------------------------

    def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch the conversation list for a user (``GET /conversations``).

        Returns an empty list on network error so the sidebar never
        crashes the whole page — the rest of the chat still works.

        Args:
            user_id: External user id whose conversations to list.
            limit: Maximum number of conversations to return.
            offset: Pagination offset.

        Returns:
            List of conversation dicts ordered by most-recent activity,
            or ``[]`` on network failure.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/conversations",
                params={"user_id": user_id, "limit": limit, "offset": offset},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("list_conversations failed: %s", e)
            return []

    def get_messages(self, conversation_id: str, limit: int = 200) -> list[dict]:
        """Fetch the full transcript for a conversation in ascending turn order.

        Calls ``GET /conversations/{id}/messages``.

        Args:
            conversation_id: Conversation whose messages to load.
            limit: Maximum number of messages to return.

        Returns:
            Ordered list of message dicts, or ``[]`` if the conversation
            does not exist or the API is unreachable.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/conversations/{conversation_id}/messages",
                params={"limit": limit},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if resp.status_code == HTTPStatus.NOT_FOUND:
                return []
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("get_messages(%s) failed: %s", conversation_id, e)
            return []

    def close_conversation(self, conversation_id: str) -> dict | None:
        """Close a conversation and schedule its summary job.

        Calls ``POST /conversations/{id}/close``.

        Args:
            conversation_id: Conversation to close.

        Returns:
            Backend response dict on success, or ``None`` on network
            failure so the UI can keep going.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/conversations/{conversation_id}/close",
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning("close_conversation(%s) failed: %s", conversation_id, e)
            return None
