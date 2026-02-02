"""Tests for the embedding service."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ConfigurationError
from app.services.embedding_service import EmbeddingService, get_embedding_service


class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    @patch("app.config.settings")
    def test_init_with_settings_api_key(self, mock_settings, mock_embeddings_class):
        """Test initialization with API key from settings."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        service = EmbeddingService()

        mock_embeddings_class.assert_called_once_with(
            model="models/embedding-001",
            google_api_key="test-api-key"
        )
        assert service._api_key == "test-api-key"
        assert service._embeddings == mock_embeddings_instance

    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    def test_init_with_override_api_key(self, mock_embeddings_class):
        """Test initialization with API key override."""
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        service = EmbeddingService(api_key="override-key")

        mock_embeddings_class.assert_called_once_with(
            model="models/embedding-001",
            google_api_key="override-key"
        )
        assert service._api_key == "override-key"

    @patch("app.config.settings")
    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    def test_init_missing_api_key_raises_error(self, mock_embeddings_class, mock_settings):
        """Test that missing API key raises ConfigurationError."""
        mock_settings.gemini_api_key = ""

        with pytest.raises(ConfigurationError) as exc_info:
            EmbeddingService()

        assert "Gemini API key is required" in str(exc_info.value)
        assert ".env file" in str(exc_info.value)
        mock_embeddings_class.assert_not_called()

    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    @patch("app.config.settings")
    def test_embeddings_property(self, mock_settings, mock_embeddings_class):
        """Test that embeddings property returns the LangChain instance."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        service = EmbeddingService()
        result = service.embeddings

        assert result == mock_embeddings_instance

    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    @patch("app.config.settings")
    def test_embed_text(self, mock_settings, mock_embeddings_class):
        """Test embedding a single text."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Mock embed_query to return a fake embedding
        fake_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embeddings_instance.embed_query.return_value = fake_embedding

        service = EmbeddingService()
        result = service.embed_text("Hello world")

        mock_embeddings_instance.embed_query.assert_called_once_with("Hello world")
        assert result == fake_embedding
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    @patch("app.config.settings")
    def test_embed_texts(self, mock_settings, mock_embeddings_class):
        """Test embedding multiple texts."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        # Mock embed_documents to return fake embeddings
        fake_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        mock_embeddings_instance.embed_documents.return_value = fake_embeddings

        service = EmbeddingService()
        texts = ["Hello", "world", "test"]
        result = service.embed_texts(texts)

        mock_embeddings_instance.embed_documents.assert_called_once_with(texts)
        assert result == fake_embeddings
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(isinstance(x, float) for embedding in result for x in embedding)

    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    @patch("app.config.settings")
    def test_get_embedding_service_dependency(self, mock_settings, mock_embeddings_class):
        """Test the FastAPI dependency function."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        service = get_embedding_service()

        assert isinstance(service, EmbeddingService)
        mock_embeddings_class.assert_called_once_with(
            model="models/embedding-001",
            google_api_key="test-api-key"
        )

    @patch("app.config.settings")
    @patch("app.services.embedding_service.GoogleGenerativeAIEmbeddings")
    def test_get_embedding_service_dependency_missing_key(self, mock_embeddings_class, mock_settings):
        """Test that dependency function raises ConfigurationError when API key is missing."""
        mock_settings.gemini_api_key = ""

        with pytest.raises(ConfigurationError) as exc_info:
            get_embedding_service()

        assert "Gemini API key is required" in str(exc_info.value)
