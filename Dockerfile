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
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies, replace GPU PyTorch with CPU-only, clean cache — all in one layer
# (GPU inference runs in a separate vLLM container)
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root && \
    pip uninstall -y \
        torch torchvision torchaudio \
        nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
        nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
        triton 2>/dev/null; \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip /root/.cache/pypoetry

# Copy application code
COPY hse_prom_prog ./hse_prom_prog

# Copy Streamlit app and config
COPY streamlit_app ./streamlit_app
COPY .streamlit ./.streamlit

# Copy static assets (served by nginx in production, Streamlit static serving in K8s)
COPY static ./static
RUN ln -s /app/static /app/streamlit_app/static

# Knowledge base is downloaded from S3 at ingestion time (see hse_prom_prog/rag/ingest.py)

# Copy database CSV data (for k8s postgres-load-data Job)
COPY database/data ./database/data

# Copy eval module (RAGAS evaluation)
COPY eval ./eval

# Copy Alembic migrations
COPY alembic ./alembic
COPY alembic.ini ./

# Copy README (referenced by pyproject.toml)
COPY README.md ./

# Install the package itself (--only-root skips dependencies, preserves CPU PyTorch)
RUN poetry install --no-interaction --no-ansi --only-root

# Default command
CMD ["python", "-m", "hse_prom_prog.main", "Привет! Выведи данные по задаче ABC-123"]
