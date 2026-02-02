"""Tests for the vector store service."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.exceptions import CollectionNotFoundError
from app.services.vector_store_service import (
    RetrievedChunk,
    VectorStoreService,
    get_vector_store_service,
)


class TestRetrievedChunk:
    """Test suite for RetrievedChunk Pydantic model."""

    def test_valid_chunk(self):
        """Test creating a valid retrieved chunk."""
        chunk = RetrievedChunk(
            content="Sample text content",
            page_number=5,
            chunk_index=12,
            relevance_score=0.87,
            source="document.pdf",
        )

        assert chunk.content == "Sample text content"
        assert chunk.page_number == 5
        assert chunk.chunk_index == 12
        assert chunk.relevance_score == 0.87
        assert chunk.source == "document.pdf"

    def test_page_number_validation(self):
        """Test that page_number must be >= 1."""
        with pytest.raises(ValueError):
            RetrievedChunk(
                content="Text",
                page_number=0,  # Invalid
                chunk_index=0,
                relevance_score=0.5,
                source="doc.pdf",
            )

    def test_chunk_index_validation(self):
        """Test that chunk_index must be >= 0."""
        with pytest.raises(ValueError):
            RetrievedChunk(
                content="Text",
                page_number=1,
                chunk_index=-1,  # Invalid
                relevance_score=0.5,
                source="doc.pdf",
            )

    def test_relevance_score_bounds(self):
        """Test that relevance_score must be between 0 and 1."""
        # Test lower bound
        with pytest.raises(ValueError):
            RetrievedChunk(
                content="Text",
                page_number=1,
                chunk_index=0,
                relevance_score=-0.1,  # Invalid
                source="doc.pdf",
            )

        # Test upper bound
        with pytest.raises(ValueError):
            RetrievedChunk(
                content="Text",
                page_number=1,
                chunk_index=0,
                relevance_score=1.5,  # Invalid
                source="doc.pdf",
            )


class TestVectorStoreService:
    """Test suite for VectorStoreService."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = MagicMock()
        service.embeddings = MagicMock()
        return service

    @pytest.fixture
    def vector_store_service(self, mock_embedding_service):
        """Create a VectorStoreService instance with mocked dependencies."""
        return VectorStoreService(
            embedding_service=mock_embedding_service, persist_directory=None
        )

    def test_initialization(self, mock_embedding_service):
        """Test service initialization."""
        service = VectorStoreService(
            embedding_service=mock_embedding_service,
            persist_directory="/tmp/chroma",
        )

        assert service._embedding_service == mock_embedding_service
        assert service._persist_directory == "/tmp/chroma"
        assert service._collections == {}

    @patch("app.services.vector_store_service.Chroma")
    def test_create_collection(
        self, mock_chroma_class, vector_store_service, mock_embedding_service
    ):
        """Test creating a new collection."""
        # Setup mock
        mock_vectorstore = MagicMock()
        mock_chroma_class.from_documents.return_value = mock_vectorstore

        # Create test documents
        documents = [
            Document(
                page_content="Test content 1",
                metadata={"page": 1, "chunk_index": 0, "filename": "test.pdf"},
            ),
            Document(
                page_content="Test content 2",
                metadata={"page": 2, "chunk_index": 1, "filename": "test.pdf"},
            ),
        ]

        # Create collection
        count = vector_store_service.create_collection("test_collection", documents)

        # Verify
        assert count == 2
        mock_chroma_class.from_documents.assert_called_once_with(
            documents=documents,
            embedding=mock_embedding_service.embeddings,
            collection_name="test_collection",
            persist_directory=None,
        )
        assert "test_collection" in vector_store_service._collections
        assert vector_store_service._collections["test_collection"] == mock_vectorstore

    def test_create_collection_already_exists(self, vector_store_service):
        """Test that creating an existing collection raises ValueError."""
        # Add a collection to the dict
        vector_store_service._collections["existing"] = MagicMock()

        with pytest.raises(ValueError, match="already exists"):
            vector_store_service.create_collection("existing", [])

    def test_get_collection_success(self, vector_store_service):
        """Test getting an existing collection."""
        mock_collection = MagicMock()
        vector_store_service._collections["test"] = mock_collection

        result = vector_store_service.get_collection("test")

        assert result == mock_collection

    def test_get_collection_not_found(self, vector_store_service):
        """Test getting a non-existent collection raises error."""
        with pytest.raises(CollectionNotFoundError) as exc_info:
            vector_store_service.get_collection("nonexistent")

        assert exc_info.value.collection_name == "nonexistent"
        assert "nonexistent" in str(exc_info.value)

    @patch("app.config.settings")
    def test_query_with_default_top_k(
        self, mock_settings, vector_store_service, mock_embedding_service
    ):
        """Test querying a collection with default top_k from settings."""
        mock_settings.top_k_chunks = 5

        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Result 1",
                    metadata={
                        "page": 1,
                        "chunk_index": 0,
                        "filename": "doc.pdf",
                    },
                ),
                0.95,
            ),
            (
                Document(
                    page_content="Result 2",
                    metadata={
                        "page": 2,
                        "chunk_index": 1,
                        "filename": "doc.pdf",
                    },
                ),
                0.87,
            ),
        ]
        vector_store_service._collections["test"] = mock_collection

        # Query
        results = vector_store_service.query("test", "test query")

        # Verify
        mock_collection.similarity_search_with_relevance_scores.assert_called_once_with(
            "test query", k=3
        )
        assert len(results) == 2
        assert all(isinstance(chunk, RetrievedChunk) for chunk in results)

        # Check first result
        assert results[0].content == "Result 1"
        assert results[0].page_number == 1
        assert results[0].chunk_index == 0
        assert results[0].relevance_score == 0.95
        assert results[0].source == "doc.pdf"

    def test_query_with_custom_top_k(self, vector_store_service):
        """Test querying with custom top_k value."""
        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = []
        vector_store_service._collections["test"] = mock_collection

        # Query with custom top_k
        vector_store_service.query("test", "test query", top_k=10)

        # Verify custom top_k was used
        mock_collection.similarity_search_with_relevance_scores.assert_called_once_with(
            "test query", k=10
        )

    def test_query_missing_metadata(self, vector_store_service):
        """Test querying handles missing metadata gracefully."""
        # Setup mock collection with document missing some metadata
        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Result with minimal metadata",
                    metadata={},  # Missing all metadata
                ),
                0.75,
            )
        ]
        vector_store_service._collections["test"] = mock_collection

        # Query
        results = vector_store_service.query("test", "test query", top_k=1)

        # Verify defaults are used
        assert len(results) == 1
        assert results[0].content == "Result with minimal metadata"
        assert results[0].page_number == 1  # Default
        assert results[0].chunk_index == 0  # Default
        assert results[0].relevance_score == 0.75
        assert results[0].source == "unknown"  # Default

    def test_query_with_source_fallback(self, vector_store_service):
        """Test querying uses 'source' metadata if 'filename' is missing."""
        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Content",
                    metadata={"source": "alternative.txt"},  # Has 'source' not 'filename'
                ),
                0.8,
            )
        ]
        vector_store_service._collections["test"] = mock_collection

        results = vector_store_service.query("test", "query", top_k=1)

        assert results[0].source == "alternative.txt"

    def test_query_collection_not_found(self, vector_store_service):
        """Test querying a non-existent collection raises error."""
        with pytest.raises(CollectionNotFoundError):
            vector_store_service.query("nonexistent", "test query")

    def test_delete_collection_success(self, vector_store_service):
        """Test deleting an existing collection."""
        mock_collection = MagicMock()
        vector_store_service._collections["test"] = mock_collection

        result = vector_store_service.delete_collection("test")

        assert result is True
        mock_collection.delete_collection.assert_called_once()
        assert "test" not in vector_store_service._collections

    def test_delete_collection_not_found(self, vector_store_service):
        """Test deleting a non-existent collection returns False."""
        result = vector_store_service.delete_collection("nonexistent")

        assert result is False

    def test_list_collections(self, vector_store_service):
        """Test listing all collections."""
        # Add some collections
        vector_store_service._collections["coll1"] = MagicMock()
        vector_store_service._collections["coll2"] = MagicMock()
        vector_store_service._collections["coll3"] = MagicMock()

        collections = vector_store_service.list_collections()

        assert len(collections) == 3
        assert "coll1" in collections
        assert "coll2" in collections
        assert "coll3" in collections

    def test_list_collections_empty(self, vector_store_service):
        """Test listing collections when none exist."""
        collections = vector_store_service.list_collections()

        assert collections == []


class TestDependencyFunction:
    """Test suite for FastAPI dependency function."""

    @patch("app.services.vector_store_service.settings")
    @patch("app.services.vector_store_service.get_embedding_service")
    def test_get_vector_store_service(self, mock_get_embedding, mock_settings):
        """Test the FastAPI dependency function."""
        mock_settings.chroma_persist_dir = "/tmp/chroma"
        mock_embedding_service = MagicMock()
        mock_get_embedding.return_value = mock_embedding_service

        service = get_vector_store_service(embedding_service=mock_embedding_service)

        assert isinstance(service, VectorStoreService)
        assert service._embedding_service == mock_embedding_service
        assert service._persist_directory == "/tmp/chroma"