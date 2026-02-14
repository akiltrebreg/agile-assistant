"""HTTP client for FastAPI backend.

This is the only module that knows about the API. All other modules
interact with the backend through this client.
"""

import logging

import requests

from streamlit_app.config import API_BASE_URL, REQUEST_TIMEOUT_SEC

logger = logging.getLogger(__name__)


class APIClient:
    """Thin wrapper around requests for calling FastAPI endpoints."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
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

    def submit_task(self, query: str) -> dict:
        """POST /tasks — create a new task.

        Args:
            query: User query string.

        Returns:
            Dict with task_id, status, message.

        Raises:
            requests.RequestException: On network / HTTP errors.
        """
        resp = requests.post(
            f"{self.base_url}/tasks",
            json={"query": query},
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
