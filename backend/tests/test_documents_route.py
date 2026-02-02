"""Tests for the document upload endpoint."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from app.api.routes.documents import router
from app.core.exceptions import (
    FileTooLargeError,
    PageLimitExceededError,
    ProcessingError,
    UnsupportedFileTypeError,
)
from app.models.schemas import DocumentUploadResponse


@pytest.fixture
def app():
    """Create a test FastAPI app with the documents router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_document_service():
    """Create a mock document service."""
    return MagicMock()


class TestUploadDocumentEndpoint:
    """Test suite for document upload endpoint."""

    def test_upload_document_success(self, client, mock_document_service):
        """Test successful document upload."""
        # Mock successful processing
        mock_response = DocumentUploadResponse(
            filename="test.pdf",
            file_type="pdf",
            total_chunks=25,
            message="Document uploaded and processed successfully",
        )
        mock_document_service.process_upload = AsyncMock(return_value=mock_response)

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Create test file
        file_content = b"Test PDF content"
        files = {"file": ("test.pdf", BytesIO(file_content), "application/pdf")}

        # Make request
        response = client.post("/documents", files=files)

        # Verify response
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        assert data["filename"] == "test.pdf"
        assert data["file_type"] == "pdf"
        assert data["total_chunks"] == 25
        assert "successfully" in data["message"]

        # Verify service was called
        mock_document_service.process_upload.assert_called_once()

    def test_upload_document_unsupported_file_type(self, client, mock_document_service):
        """Test upload with unsupported file type."""
        # Mock unsupported file type error
        mock_document_service.process_upload = AsyncMock(
            side_effect=UnsupportedFileTypeError(
                extension="xyz", allowed=["pdf", "txt", "docx"]
            )
        )

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Create test file with unsupported extension
        files = {"file": ("test.xyz", BytesIO(b"content"), "application/octet-stream")}

        # Make request
        response = client.post("/documents", files=files)

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "xyz" in data["detail"]
        assert "not supported" in data["detail"]

    def test_upload_document_file_too_large(self, client, mock_document_service):
        """Test upload with file that's too large."""
        # Mock file too large error
        mock_document_service.process_upload = AsyncMock(
            side_effect=FileTooLargeError(size_mb=25.5, max_size_mb=20)
        )

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Create test file
        files = {"file": ("large.pdf", BytesIO(b"x" * 1000000), "application/pdf")}

        # Make request
        response = client.post("/documents", files=files)

        # Verify error response
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        data = response.json()
        assert "detail" in data
        assert "25.5" in data["detail"]
        assert "exceeds" in data["detail"]

    def test_upload_document_page_limit_exceeded(self, client, mock_document_service):
        """Test upload with document that has too many pages."""
        # Mock page limit exceeded error
        mock_document_service.process_upload = AsyncMock(
            side_effect=PageLimitExceededError(page_count=75, max_pages=50)
        )

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Create test file
        files = {"file": ("long.pdf", BytesIO(b"content"), "application/pdf")}

        # Make request
        response = client.post("/documents", files=files)

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "75" in data["detail"]
        assert "pages" in data["detail"]

    def test_upload_document_processing_error(self, client, mock_document_service):
        """Test upload with processing error."""
        # Mock processing error
        mock_document_service.process_upload = AsyncMock(
            side_effect=ProcessingError("Failed to extract text from PDF")
        )

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Create test file
        files = {"file": ("corrupt.pdf", BytesIO(b"invalid"), "application/pdf")}

        # Make request
        response = client.post("/documents", files=files)

        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert "Failed to extract text" in data["detail"]

    def test_upload_document_no_file(self, client):
        """Test upload without providing a file."""
        # Make request without file
        response = client.post("/documents")

        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_document_different_file_types(self, client, mock_document_service):
        """Test uploading different supported file types."""
        mock_response = DocumentUploadResponse(
            filename="test.txt",
            file_type="txt",
            total_chunks=10,
            message="Document uploaded and processed successfully",
        )
        mock_document_service.process_upload = AsyncMock(return_value=mock_response)

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Test TXT file
        files = {"file": ("test.txt", BytesIO(b"Text content"), "text/plain")}
        response = client.post("/documents", files=files)
        assert response.status_code == status.HTTP_201_CREATED

        # Test DOCX file
        mock_response.filename = "test.docx"
        mock_response.file_type = "docx"
        files = {
            "file": (
                "test.docx",
                BytesIO(b"DOCX content"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        }
        response = client.post("/documents", files=files)
        assert response.status_code == status.HTTP_201_CREATED

    def test_upload_document_replaces_previous(self, client, mock_document_service):
        """Test that uploading a new document replaces the previous one."""
        # Mock successful processing
        mock_response1 = DocumentUploadResponse(
            filename="first.pdf", file_type="pdf", total_chunks=20, message="Success"
        )
        mock_response2 = DocumentUploadResponse(
            filename="second.pdf", file_type="pdf", total_chunks=30, message="Success"
        )

        # Override dependency
        from app.api.routes.documents import get_document_service

        client.app.dependency_overrides[get_document_service] = (
            lambda: mock_document_service
        )

        # Upload first document
        mock_document_service.process_upload = AsyncMock(return_value=mock_response1)
        files = {"file": ("first.pdf", BytesIO(b"content1"), "application/pdf")}
        response1 = client.post("/documents", files=files)
        assert response1.status_code == status.HTTP_201_CREATED
        assert response1.json()["filename"] == "first.pdf"

        # Upload second document
        mock_document_service.process_upload = AsyncMock(return_value=mock_response2)
        files = {"file": ("second.pdf", BytesIO(b"content2"), "application/pdf")}
        response2 = client.post("/documents", files=files)
        assert response2.status_code == status.HTTP_201_CREATED
        assert response2.json()["filename"] == "second.pdf"

    def test_upload_document_openapi_metadata(self, app):
        """Test that upload endpoint has proper OpenAPI metadata."""
        openapi_schema = app.openapi()

        # Check endpoint exists in schema
        assert "/documents" in openapi_schema["paths"]
        upload_endpoint = openapi_schema["paths"]["/documents"]["post"]

        # Check metadata
        assert upload_endpoint["summary"] == "Upload a document"
        assert "documents" in upload_endpoint["tags"]
        assert "PDF, DOCX, or TXT" in upload_endpoint["description"]

        # Check responses
        assert "201" in upload_endpoint["responses"]
        assert "400" in upload_endpoint["responses"]
        assert "413" in upload_endpoint["responses"]
        assert "422" in upload_endpoint["responses"]

        # Check response descriptions
        assert (
            upload_endpoint["responses"]["201"]["description"]
            == "Document processed successfully"
        )
        assert upload_endpoint["responses"]["400"]["description"] == "Invalid file"
        assert upload_endpoint["responses"]["413"]["description"] == "File too large"
        assert (
            upload_endpoint["responses"]["422"]["description"] == "Processing failed"
        )

    def test_upload_document_request_body_schema(self, app):
        """Test that request body has proper schema."""
        openapi_schema = app.openapi()
        upload_endpoint = openapi_schema["paths"]["/documents"]["post"]

        # Check request body
        assert "requestBody" in upload_endpoint
        request_body = upload_endpoint["requestBody"]
        assert "multipart/form-data" in request_body["content"]

        # Check file parameter
        schema = request_body["content"]["multipart/form-data"]["schema"]
        # Schema may have properties directly or use a $ref
        assert "properties" in schema or "$ref" in schema


class TestErrorResponseModel:
    """Test suite for ErrorResponse model."""

    def test_error_response_creation(self):
        """Test creating an ErrorResponse."""
        from app.models.schemas import ErrorResponse

        error = ErrorResponse(error="Test error", detail="Something went wrong")

        assert error.error == "Test error"
        assert error.detail == "Something went wrong"

    def test_error_response_serialization(self):
        """Test ErrorResponse serialization."""
        from app.models.schemas import ErrorResponse

        error = ErrorResponse(error="Not found", detail="File not found")
        data = error.model_dump()

        assert data["error"] == "Not found"
        assert data["detail"] == "File not found"

    def test_error_response_json_schema(self):
        """Test ErrorResponse has proper JSON schema example."""
        from app.models.schemas import ErrorResponse

        schema = ErrorResponse.model_json_schema()
        assert "example" in schema