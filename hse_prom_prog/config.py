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


# Global settings instance
settings = Settings()
