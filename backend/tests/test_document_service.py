"""Tests for the document service."""

import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import UploadFile

from app.core.exceptions import (
    FileTooLargeError,
    PageLimitExceededError,
    ProcessingError,
    UnsupportedFileTypeError,
)
from app.models.schemas import DocumentUploadResponse
from app.services.document_service import (
    DocumentService,
    ProcessingResult,
    get_document_service,
)


class TestProcessingResult:
    """Test suite for ProcessingResult model."""

    def test_valid_result(self):
        """Test creating a valid processing result."""
        result = ProcessingResult(
            filename="test.pdf",
            page_count=10,
            chunk_count=25,
            collection_name="session_abc123",
            processing_time_ms=1234.56,
        )

        assert result.filename == "test.pdf"
        assert result.page_count == 10
        assert result.chunk_count == 25
        assert result.collection_name == "session_abc123"
        assert result.processing_time_ms == 1234.56


class TestDocumentService:
    """Test suite for DocumentService."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.allowed_extensions = ["pdf", "txt", "docx"]
        settings.max_file_size_mb = 20
        settings.max_page_limit = 50
        settings.chunk_size_tokens = 500
        settings.chunk_overlap_tokens = 100
        return settings

    @pytest.fixture
    def mock_document_processor(self):
        """Create mock document processor."""
        return MagicMock()

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store service."""
        return MagicMock()

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        manager = MagicMock()
        manager.is_active = False
        manager.collection_name = None
        return manager

    @pytest.fixture
    def document_service(
        self,
        mock_document_processor,
        mock_vector_store,
        mock_session_manager,
        mock_settings,
    ):
        """Create DocumentService instance with mocked dependencies."""
        return DocumentService(
            document_processor=mock_document_processor,
            vector_store_service=mock_vector_store,
            session_manager=mock_session_manager,
            settings=mock_settings,
        )

    def test_initialization(
        self,
        mock_document_processor,
        mock_vector_store,
        mock_session_manager,
        mock_settings,
    ):
        """Test service initialization."""
        service = DocumentService(
            document_processor=mock_document_processor,
            vector_store_service=mock_vector_store,
            session_manager=mock_session_manager,
            settings=mock_settings,
        )

        assert service._document_processor == mock_document_processor
        assert service._vector_store == mock_vector_store
        assert service._session_manager == mock_session_manager
        assert service._settings == mock_settings

    def test_validate_file_success(self, document_service):
        """Test successful file validation."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.size = 10 * 1024 * 1024  # 10MB

        # Should not raise
        document_service._validate_file(file)

    def test_validate_file_unsupported_extension(self, document_service):
        """Test validation fails for unsupported extension."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.xyz"
        file.size = 1024

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            document_service._validate_file(file)

        assert exc_info.value.extension == "xyz"
        assert exc_info.value.allowed == ["pdf", "txt", "docx"]

    def test_validate_file_too_large(self, document_service):
        """Test validation fails for file that's too large."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.size = 25 * 1024 * 1024  # 25MB (exceeds 20MB limit)

        with pytest.raises(FileTooLargeError) as exc_info:
            document_service._validate_file(file)

        assert exc_info.value.size_mb > 20
        assert exc_info.value.max_size_mb == 20

    @pytest.mark.asyncio
    async def test_save_temp_file(self, document_service):
        """Test saving uploaded file to temp location."""
        # Create mock upload file
        content = b"Test file content"
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.read = AsyncMock(return_value=content)

        # Save file
        temp_path = await document_service._save_temp_file(file)

        # Verify file was created
        assert temp_path.exists()
        assert temp_path.suffix == ".pdf"

        # Verify content
        with open(temp_path, "rb") as f:
            saved_content = f.read()
        assert saved_content == content

        # Cleanup
        temp_path.unlink()

    def test_cleanup_temp_file(self, document_service):
        """Test cleaning up temp file."""
        # Create a real temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Verify it exists
        assert tmp_path.exists()

        # Cleanup
        document_service._cleanup_temp_file(tmp_path)

        # Verify it's deleted
        assert not tmp_path.exists()

    def test_cleanup_temp_file_not_exists(self, document_service):
        """Test cleanup handles non-existent file gracefully."""
        non_existent = Path("/tmp/does_not_exist_12345.pdf")

        # Should not raise
        document_service._cleanup_temp_file(non_existent)

    def test_cleanup_old_session_no_active_session(
        self, document_service, mock_session_manager, mock_vector_store
    ):
        """Test cleanup when no session is active."""
        mock_session_manager.is_active = False

        document_service._cleanup_old_session()

        # Should not try to delete collection or clear session
        mock_vector_store.delete_collection.assert_not_called()

    def test_cleanup_old_session_with_active_session(
        self, document_service, mock_session_manager, mock_vector_store
    ):
        """Test cleanup deletes old collection and clears session."""
        mock_session_manager.is_active = True
        mock_session_manager.collection_name = "old_collection"

        document_service._cleanup_old_session()

        # Should delete old collection and clear session
        mock_vector_store.delete_collection.assert_called_once_with("old_collection")
        mock_session_manager.clear_session.assert_called_once()

    def test_cleanup_old_session_deletion_fails(
        self, document_service, mock_session_manager, mock_vector_store
    ):
        """Test cleanup handles deletion failure gracefully."""
        mock_session_manager.is_active = True
        mock_session_manager.collection_name = "old_collection"
        mock_vector_store.delete_collection.side_effect = Exception("Delete failed")

        # Should not raise, just log warning
        document_service._cleanup_old_session()

        # Session should still be cleared
        mock_session_manager.clear_session.assert_called_once()

    def test_estimate_page_count_with_page_metadata(self, document_service):
        """Test page count estimation using page metadata."""
        chunks = [
            MagicMock(metadata={"page": 1}),
            MagicMock(metadata={"page": 2}),
            MagicMock(metadata={"page": 3}),
            MagicMock(metadata={"page": 3}),  # Multiple chunks on same page
        ]

        page_count = document_service._estimate_page_count(chunks)

        assert page_count == 3

    def test_estimate_page_count_without_metadata(self, document_service):
        """Test page count estimation without page metadata."""
        chunks = [MagicMock(metadata={}) for _ in range(12)]

        page_count = document_service._estimate_page_count(chunks)

        # Should estimate ~3 chunks per page: 12 chunks / 3 = 4 pages
        assert page_count == 4

    def test_estimate_page_count_empty(self, document_service):
        """Test page count estimation with no chunks."""
        page_count = document_service._estimate_page_count([])

        assert page_count == 1  # Default to 1 page

    @pytest.mark.asyncio
    async def test_process_upload_success(
        self,
        document_service,
        mock_document_processor,
        mock_vector_store,
        mock_session_manager,
    ):
        """Test successful document upload processing."""
        # Mock file
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.size = 1024
        file.read = AsyncMock(return_value=b"test content")

        # Mock processed document
        mock_chunks = [
            MagicMock(metadata={"page": 1}),
            MagicMock(metadata={"page": 2}),
        ]
        mock_processed = MagicMock()
        mock_processed.filename = "test.pdf"
        mock_processed.file_type = "pdf"
        mock_processed.total_chunks = 2
        mock_processed.chunks = mock_chunks

        mock_document_processor.process_document.return_value = mock_processed

        # Process upload
        response = await document_service.process_upload(file)

        # Verify response
        assert isinstance(response, DocumentUploadResponse)
        assert response.filename == "test.pdf"
        assert response.file_type == "pdf"
        assert response.total_chunks == 2

        # Verify pipeline steps
        mock_document_processor.process_document.assert_called_once()
        mock_vector_store.create_collection.assert_called_once()
        mock_session_manager.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_upload_page_limit_exceeded(
        self, document_service, mock_document_processor, mock_settings
    ):
        """Test upload fails when page limit is exceeded."""
        mock_settings.max_page_limit = 5

        # Mock file
        file = MagicMock(spec=UploadFile)
        file.filename = "large.pdf"
        file.size = 1024
        file.read = AsyncMock(return_value=b"test content")

        # Mock processed document with too many pages
        mock_chunks = [MagicMock(metadata={"page": i}) for i in range(1, 11)]  # 10 pages
        mock_processed = MagicMock()
        mock_processed.chunks = mock_chunks

        mock_document_processor.process_document.return_value = mock_processed

        # Should raise error
        with pytest.raises(PageLimitExceededError) as exc_info:
            await document_service.process_upload(file)

        assert exc_info.value.page_count == 10
        assert exc_info.value.max_pages == 5

    @pytest.mark.asyncio
    async def test_process_upload_processing_error(
        self, document_service, mock_document_processor
    ):
        """Test upload handles processing errors."""
        # Mock file
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.size = 1024
        file.read = AsyncMock(return_value=b"test content")

        # Mock processing failure
        from app.core.document_processor import DocumentProcessingError

        mock_document_processor.process_document.side_effect = DocumentProcessingError(
            "Failed to parse PDF"
        )

        # Should wrap in ProcessingError
        with pytest.raises(ProcessingError) as exc_info:
            await document_service.process_upload(file)

        assert "Failed to process document" in str(exc_info.value)
        assert exc_info.value.original_error is not None

    @pytest.mark.asyncio
    async def test_process_upload_cleans_up_temp_file(
        self, document_service, mock_document_processor
    ):
        """Test that temp file is cleaned up even on error."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.pdf"
        file.size = 1024
        file.read = AsyncMock(return_value=b"test content")

        # Mock processing failure
        mock_document_processor.process_document.side_effect = Exception("Error")

        # Patch cleanup method to track calls
        with patch.object(
            document_service, "_cleanup_temp_file"
        ) as mock_cleanup:
            try:
                await document_service.process_upload(file)
            except ProcessingError:
                pass

            # Verify cleanup was called
            mock_cleanup.assert_called_once()


class TestDependencyFunction:
    """Test suite for FastAPI dependency function."""

    @patch("app.services.document_service.DocumentProcessor")
    def test_get_document_service(self, mock_processor_class):
        """Test the FastAPI dependency function."""
        # Mock dependencies
        mock_session_manager = MagicMock()
        mock_vector_store = MagicMock()
        mock_settings = MagicMock()
        mock_settings.chunk_size_tokens = 500
        mock_settings.chunk_overlap_tokens = 100

        # Mock processor instance
        mock_processor_instance = MagicMock()
        mock_processor_class.return_value = mock_processor_instance

        # Get service
        service = get_document_service(
            session_manager=mock_session_manager,
            vector_store_service=mock_vector_store,
            settings=mock_settings,
        )

        # Verify processor was initialized with correct settings
        mock_processor_class.assert_called_once_with(
            chunk_size=500, chunk_overlap=100
        )

        # Verify service was created
        assert isinstance(service, DocumentService)
        assert service._document_processor == mock_processor_instance
        assert service._vector_store == mock_vector_store
        assert service._session_manager == mock_session_manager
        assert service._settings == mock_settings