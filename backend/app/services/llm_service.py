"""LLM service for Gemini chat completions with streaming."""

import logging
from typing import AsyncIterator, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import settings
from app.core.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)


class LLMMessage(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class LLMConfig(BaseModel):
    """Configuration for LLM generation."""

    model: str = "gemini-2.5-flash-lite"
    temperature: float = Field(default=0.3, ge=0, le=1)
    max_output_tokens: int = Field(default=2048, ge=1, le=8192)


class LLMService:
    """
    Wrapper around Langchain's ChatGoogleGenerativeAI.

    Provides streaming chat completions using Gemini.
    """

    def __init__(
        self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None
    ):
        """
        Initialize LLM service.

        Args:
            api_key: Gemini API key. If None, reads from config.
            config: LLM configuration. If None, uses defaults.

        Raises:
            ConfigurationError: If API key not available.
        """
        # Lazy import to avoid loading settings at module import time
        if api_key is None:
            from app.config import settings

            api_key = settings.gemini_api_key

        if not api_key:
            raise ConfigurationError(
                "Gemini API key is required. Please set GEMINI_API_KEY in your .env file."
            )

        self._api_key = api_key
        self._config = config or LLMConfig()

        # Initialize ChatGoogleGenerativeAI
        try:
            self._llm = ChatGoogleGenerativeAI(
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_output_tokens,
                api_key=self._api_key,
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize ChatGoogleGenerativeAI: {e}"
            ) from e

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming response.

        Args:
            prompt: User prompt (includes context and question)
            system_prompt: Optional system instructions

        Yields:
            Token strings as they are generated

        Raises:
            LLMError: If generation fails
        """
        try:
            logger.debug("Starting LLM streaming generation")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            logger.debug(f"System prompt: {'Yes' if system_prompt else 'No'}")

            # Build message list
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            logger.debug(f"Calling Gemini API with {len(messages)} messages...")

            # Stream response
            chunk_count = 0
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    chunk_count += 1
                    yield chunk.content

            logger.debug(f"LLM streaming completed: {chunk_count} chunks received")

        except Exception as e:
            logger.error(f"❌ LLM streaming generation failed: {e}", exc_info=True)
            raise LLMError(
                "Failed to generate streaming response from LLM", original_error=e
            ) from e

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate complete response (non-streaming).

        Useful for testing or when streaming not needed.

        Args:
            prompt: User prompt (includes context and question)
            system_prompt: Optional system instructions

        Returns:
            Complete response string

        Raises:
            LLMError: If generation fails
        """
        try:
            # Build message list
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            # Generate response
            response = await self._llm.ainvoke(messages)

            return response.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise LLMError(
                "Failed to generate response from LLM", original_error=e
            ) from e


def get_llm_service() -> LLMService:
    """
    FastAPI dependency function to get an LLM service instance.

    Returns:
        LLMService instance

    Raises:
        ConfigurationError: If API key is not configured
    """
    return LLMService()