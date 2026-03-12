"""Configuration module for HSE Prom Prog.

This module contains Pydantic settings for configuring the application,
including vLLM API endpoints and model parameters.
"""

from pydantic import Field
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
        default="Qwen/Qwen2.5-3B-Instruct",
        description="Model name to use with vLLM",
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM (use 'EMPTY' for local deployments)",
    )
    vllm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    vllm_max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens for LLM responses",
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
