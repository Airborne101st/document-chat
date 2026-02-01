"""Embedding service for generating text embeddings using Google Gemini."""

from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.exceptions import ConfigurationError


class EmbeddingService:
    """Service for generating embeddings using Google Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            api_key: Optional API key override. If not provided, uses settings.gemini_api_key

        Raises:
            ConfigurationError: If API key is not provided and not found in settings
        """
        # Lazy import to avoid loading settings at module import time
        if api_key is None:
            from app.config import settings
            api_key = settings.gemini_api_key

        self._api_key = api_key

        if not self._api_key:
            raise ConfigurationError(
                "Gemini API key is required. Please set GEMINI_API_KEY in your .env file."
            )

        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self._api_key
        )

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Get the underlying LangChain embeddings instance.

        Returns:
            GoogleGenerativeAIEmbeddings instance that can be passed to ChromaDB
        """
        return self._embeddings

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple text strings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one for each input text
        """
        return self._embeddings.embed_documents(texts)


def get_embedding_service() -> EmbeddingService:
    """
    FastAPI dependency function to get an embedding service instance.

    Returns:
        EmbeddingService instance

    Raises:
        ConfigurationError: If API key is not configured
    """
    return EmbeddingService()
