"""Configuration for Streamlit app.

All settings are read from environment variables with sensible defaults.
Inside Docker: API_BASE_URL=http://api:8080
Local dev:     API_BASE_URL=http://localhost:8080
"""

import os

# Application mode
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# FastAPI base URL (no trailing slash)
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8080")

# Polling settings
POLL_INTERVAL_SEC: float = float(os.getenv("POLL_INTERVAL_SEC", "2.0"))
POLL_TIMEOUT_SEC: float = float(os.getenv("POLL_TIMEOUT_SEC", "120.0"))

# HTTP request timeout
REQUEST_TIMEOUT_SEC: float = float(os.getenv("REQUEST_TIMEOUT_SEC", "10.0"))
