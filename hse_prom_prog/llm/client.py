"""LLM client module for interacting with vLLM via OpenAI-compatible API.

This module provides a client for communicating with vLLM using the
OpenAI SDK, with proper error handling and logging.
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from hse_prom_prog.config import settings
from hse_prom_prog.tracing import langfuse_context, observe

logger = logging.getLogger(__name__)


def _extract_usage(response: Any) -> dict[str, int] | None:
    """Pull token counts off a LangChain message in either supported format.

    Newer LangChain exposes ``usage_metadata`` (input/output/total) directly
    on the message; older versions stash an OpenAI-shape ``token_usage``
    dict under ``response_metadata``. We accept either and normalise to
    the Langfuse usage shape.
    """
    um = getattr(response, "usage_metadata", None)
    if um:
        return {
            "input": um.get("input_tokens", 0),
            "output": um.get("output_tokens", 0),
            "total": um.get("total_tokens", 0),
        }
    legacy = (getattr(response, "response_metadata", None) or {}).get("token_usage") or {}
    if legacy:
        return {
            "input": legacy.get("prompt_tokens", 0),
            "output": legacy.get("completion_tokens", 0),
            "total": legacy.get("total_tokens", 0),
        }
    return None


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

    @observe(as_type="generation", name="llm_call")
    def invoke(
        self,
        prompt: str,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response from the LLM for the given prompt.

        Args:
            prompt: The input prompt for the LLM.
            response_format: Optional OpenAI-compatible response_format
                (e.g. JSON schema). When set, vLLM uses guided decoding
                to force the output to match the schema exactly.
            max_tokens: Per-call override for completion length. Useful
                for classifiers that emit short structured output — avoids
                wasting budget on the default 600.

        Returns:
            The generated text response from the LLM.

        Raises:
            Exception: If there's an error communicating with the LLM API.
        """
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        # Best-effort tracing: build the generation metadata up front so a
        # crash before .invoke() still leaves a useful Langfuse record.
        langfuse_context.update_current_observation(
            model=self.model,
            input=prompt,
            model_parameters={
                "temperature": self.temperature,
                "max_tokens": effective_max_tokens,
                "response_format": "json_schema" if response_format else None,
            },
        )

        try:
            logger.debug(f"Invoking LLM with prompt: {prompt[:100]}...")
            bind_kwargs: dict[str, Any] = {}
            if response_format is not None:
                bind_kwargs["response_format"] = response_format
            if max_tokens is not None:
                bind_kwargs["max_tokens"] = max_tokens
            client = self.client.bind(**bind_kwargs) if bind_kwargs else self.client
            response = client.invoke(prompt)
            result = response.content if hasattr(response, "content") else str(response)
            logger.debug(f"LLM response: {result[:100]}...")

            usage = _extract_usage(response)
            update: dict[str, Any] = {"output": result}
            if usage is not None:
                update["usage"] = usage
            langfuse_context.update_current_observation(**update)
            return result
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=f"{type(e).__name__}: {e}",
            )
            raise


def get_llm_client(**kwargs: Any) -> LLMClient:
    """Factory function to create and return an LLM client instance.

    Args:
        **kwargs: Optional keyword arguments to override default settings.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(**kwargs)
