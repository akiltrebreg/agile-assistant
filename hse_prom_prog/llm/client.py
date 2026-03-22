"""LLM client module for interacting with vLLM via OpenAI-compatible API.

This module provides a client for communicating with vLLM using the
OpenAI SDK, with proper error handling and logging.
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with vLLM through OpenAI-compatible API.

    This class wraps the LangChain ChatOpenAI client configured for vLLM,
    providing methods for text generation with error handling.

    Attributes:
        client: LangChain ChatOpenAI client instance configured for vLLM.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the LLM client with vLLM configuration.

        Args:
            base_url: Base URL for vLLM API. Defaults to settings value.
            model: Model name to use. Defaults to settings value.
            api_key: API key for authentication. Defaults to settings value.
            temperature: Temperature for generation. Defaults to settings value.
            max_tokens: Maximum tokens for responses. Defaults to settings value.
        """
        self.base_url = base_url or settings.vllm_base_url
        self.model = model or settings.vllm_model
        self.api_key = api_key or settings.vllm_api_key
        self.temperature = temperature or settings.vllm_temperature
        self.max_tokens = max_tokens or settings.vllm_max_tokens

        logger.info(f"Initializing LLM client with base_url={self.base_url}, model={self.model}")

        self.client = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={
                "repetition_penalty": settings.vllm_repetition_penalty,
            },
        )

    def invoke(self, prompt: str) -> str:
        """Generate a response from the LLM for the given prompt.

        Args:
            prompt: The input prompt for the LLM.

        Returns:
            The generated text response from the LLM.

        Raises:
            Exception: If there's an error communicating with the LLM API.
        """
        try:
            logger.debug(f"Invoking LLM with prompt: {prompt[:100]}...")
            response = self.client.invoke(prompt)
            result = response.content if hasattr(response, "content") else str(response)
            logger.debug(f"LLM response: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise


def get_llm_client(**kwargs: Any) -> LLMClient:
    """Factory function to create and return an LLM client instance.

    Args:
        **kwargs: Optional keyword arguments to override default settings.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(**kwargs)
