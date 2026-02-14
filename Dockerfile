# Multi-stage Dockerfile for HSE Prom Prog

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY hse_prom_prog ./hse_prom_prog

# Copy Streamlit app and config
COPY streamlit_app ./streamlit_app
COPY .streamlit ./.streamlit

# Copy static assets (served by nginx in production)
COPY static ./static

# Copy Alembic migrations
COPY alembic ./alembic
COPY alembic.ini ./

# Copy README (referenced by pyproject.toml)
COPY README.md ./

# Install the package
RUN poetry install --no-interaction --no-ansi

# Default command
CMD ["python", "-m", "hse_prom_prog.main", "Привет! Выведи данные по задаче ABC-123"]
