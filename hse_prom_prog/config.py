"""Configuration module for HSE Prom Prog.

This module contains Pydantic settings for configuring the application,
including vLLM API endpoints and model parameters.
"""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings.

    Attributes:
        vllm_base_url: Base URL for vLLM OpenAI-compatible API endpoint.
        vllm_model: Model name to use with vLLM.
        vllm_api_key: API key for vLLM (not required for local deployments).
        vllm_temperature: Temperature parameter for LLM generation.
        vllm_max_tokens: Maximum tokens for LLM responses.
        postgres_host: PostgreSQL host address.
        postgres_port: PostgreSQL port number.
        postgres_user: PostgreSQL user name.
        postgres_password: PostgreSQL password.
        postgres_db: PostgreSQL database name.
        log_level: Logging level for the application.
        redis_host: Redis host address.
        redis_port: Redis port number.
        redis_db: Redis database number.
        redis_password: Redis password (optional).
        celery_broker_url: Celery broker URL (optional, defaults to Redis URL).
        celery_task_track_started: Whether to track task start times.
        celery_task_time_limit: Hard time limit for tasks in seconds.
        celery_task_soft_time_limit: Soft time limit for tasks in seconds.
        fastapi_host: FastAPI server bind address.
        fastapi_port: FastAPI server port number.
        fastapi_workers: Number of Uvicorn worker processes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # vLLM Configuration
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for vLLM OpenAI-compatible endpoint",
    )
    vllm_model: str = Field(
        default="/models/avibe-gptq-8bit",
        description="Model name to use with vLLM (local path or HuggingFace ID)",
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM (use 'EMPTY' for local deployments)",
    )
    vllm_temperature: float = Field(
        default=0.00,  # 0.05
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    vllm_max_tokens: int = Field(
        default=600,
        ge=1,
        le=4096,
        description="Maximum tokens for LLM responses",
    )
    vllm_repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Repetition penalty to avoid generation loops",
    )

    # SQL LLM (arctic-7b text2sql, separate from main vLLM)
    sql_vllm_base_url: str = Field(
        default="http://localhost:8001/v1",
        description="Base URL for text2sql vLLM endpoint (arctic-7b)",
    )
    sql_vllm_model: str = Field(
        default="/models/qwen3-8b-awq-4bit",
        description="Text2SQL model name (Qwen3-8B-AWQ)",
    )

    # PostgreSQL Configuration
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host address",
    )
    postgres_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL port number",
    )
    postgres_user: str = Field(
        default="hse_user",
        description="PostgreSQL user name",
    )
    postgres_password: str = Field(
        default="hse_password",
        description="PostgreSQL password",
    )
    postgres_db: str = Field(
        default="hse_jira_db",
        description="PostgreSQL database name",
    )

    # S3 Knowledge Base (Yandex Cloud Object Storage)
    s3_kb_bucket: str | None = Field(
        default="knowledge-base",
        description="S3 bucket for knowledge base. None = use local knowledge_base/ dir.",
    )
    s3_kb_path: str = Field(
        default="knowledge_base",
        description="Path prefix inside S3 bucket (e.g. 'knowledge_base')",
    )
    s3_data_bucket: str | None = Field(
        default="database-agile",
        description="S3 bucket for CSV data. None = use local database/data/ dir.",
    )
    s3_data_path: str = Field(
        default="data",
        description="Path prefix inside S3 data bucket",
    )
    s3_endpoint: str = Field(
        default="https://storage.yandexcloud.net",
        description="S3 endpoint URL (Yandex Cloud Object Storage)",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector store URL",
    )
    qdrant_collection_name: str = Field(
        default="business_docs",
        description="Qdrant collection name for RAG documents",
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        description="HuggingFace embedding model for RAG",
    )
    embedding_sparse_model: str | None = Field(
        default=None,
        description="Sparse embedding model (e.g. 'BAAI/bge-m3'). None = fastembed BM25.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        description="Truncate embeddings to this dimension (Matryoshka). None = full model dim.",
    )

    @field_validator("embedding_dimension")
    @classmethod
    def _check_embedding_dimension(cls, v: int | None) -> int | None:
        allowed = {64, 128, 256, 512, 768, 1024}
        if v is not None and v not in allowed:
            msg = f"embedding_dimension must be one of {sorted(allowed)}, got {v}"
            raise ValueError(msg)
        return v

    retriever_top_k: int = Field(
        default=4,
        ge=1,
        description="Final number of chunks after reranking",
    )
    retriever_initial_k: int = Field(
        default=20,
        ge=1,
        description="Number of chunks fetched before reranking",
    )

    # Chunking Configuration
    chunk_size: int = Field(
        default=500,
        ge=100,
        description="Chunk size for RecursiveCharacterTextSplitter",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Chunk overlap for RecursiveCharacterTextSplitter",
    )
    max_context_chars: int = Field(
        default=4000,
        ge=500,
        description="Max characters of context passed to LLM",
    )

    # Search Configuration
    search_type: Literal["dense", "sparse", "hybrid"] = Field(
        default="dense",
        description="Retrieval mode: 'dense' (cosine), 'sparse' (BM25), 'hybrid' (RRF fusion)",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF fusion parameter k (used only when search_type='hybrid')",
    )

    # Reranker Configuration
    reranker_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking stage",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking",
    )
    reranker_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum reranker score to keep a document",
    )
    reranker_top_n: int = Field(
        default=4,
        ge=1,
        description="Max documents to keep after reranking",
    )

    # Input Guardrail (TopicGuard) — two-zone threshold
    guardrail_enabled: bool = Field(
        default=True,
        description="Enable input topic guardrail (off-topic filter before Supervisor)",
    )
    guardrail_hard_block_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Hard block: below this NLI on-topic probability the query is "
            "treated as clearly off-topic and blocked without reaching Supervisor."
        ),
    )
    guardrail_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Confident-pass threshold on NLI probability. Queries in "
            "[hard_block, threshold) pass through as low_confidence — "
            "Supervisor acts as second-line filter."
        ),
    )

    # Application mode
    debug: bool = Field(
        default=False,
        description="Debug mode (enables hot-reload, verbose logging)",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Redis Configuration
    redis_host: str = Field(
        default="localhost",
        description="Redis host address",
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port number",
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number",
    )
    redis_password: str | None = Field(
        default=None,
        description="Redis password (optional)",
    )

    # Celery Configuration
    celery_broker_url: str | None = Field(
        default=None,
        description="Celery broker URL (auto-generated from Redis if not provided)",
    )
    celery_task_track_started: bool = Field(
        default=True,
        description="Track when tasks are started",
    )
    celery_task_time_limit: int = Field(
        default=600,  # 10 minutes
        ge=1,
        description="Hard time limit for tasks (seconds)",
    )
    celery_task_soft_time_limit: int = Field(
        default=300,  # 5 minutes
        ge=1,
        description="Soft time limit for tasks (seconds)",
    )

    # FastAPI Configuration
    fastapi_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host",
    )
    fastapi_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="FastAPI server port",
    )
    fastapi_workers: int = Field(
        default=1,
        ge=1,
        description="Number of Uvicorn workers",
    )
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or '*' for all",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list.

        Returns:
            List of allowed origins. ['*'] means all origins.
        """
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL.

        Returns:
            PostgreSQL connection string in SQLAlchemy format.
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL.

        Returns:
            Redis connection string.
        """
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:"
                f"{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_broker(self) -> str:
        """Get Celery broker URL.

        Returns:
            Celery broker connection string (defaults to Redis URL).
        """
        return self.celery_broker_url or self.redis_url


# Global settings instance
settings = Settings()
