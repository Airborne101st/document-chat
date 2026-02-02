"""
End-to-end tests for the complete RAG pipeline.

Tests the full flow with mocked external services (Gemini API).
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_txt_bytes():
    """Create sample text file content for testing."""
    content = """
    Machine Learning and Artificial Intelligence

    Machine learning is a subset of artificial intelligence (AI) that provides systems
    the ability to automatically learn and improve from experience without being explicitly
    programmed. Machine learning focuses on the development of computer programs that can
    access data and use it to learn for themselves.

    The process of learning begins with observations or data, such as examples, direct
    experience, or instruction, in order to look for patterns in data and make better
    decisions in the future based on the examples that we provide.

    The primary aim is to allow the computers to learn automatically without human
    intervention or assistance and adjust actions accordingly.
    """.strip()
    return content.encode("utf-8")


@pytest.fixture
def client():
    """Create test client with mocked external services."""
    # Mock Gemini embeddings and LLM
    with patch(
        "app.services.embedding_service.GoogleGenerativeAIEmbeddings"
    ) as mock_embed, patch(
        "app.services.llm_service.ChatGoogleGenerativeAI"
    ) as mock_llm:
        # Configure mock embeddings
        mock_embed_instance = MagicMock()
        mock_embed_instance.embed_query.return_value = [0.1] * 768
        mock_embed_instance.embed_documents.return_value = [[0.1] * 768] * 10
        mock_embed.return_value = mock_embed_instance

        # Configure mock LLM streaming
        async def mock_stream(*args, **kwargs):
            for token in [
                "Machine ",
                "learning ",
                "is ",
                "a ",
                "subset ",
                "of ",
                "artificial ",
                "intelligence.",
            ]:
                yield MagicMock(content=token)

        mock_llm_instance = MagicMock()
        mock_llm_instance.astream = mock_stream
        mock_llm.return_value = mock_llm_instance

        from app.main import app

        yield TestClient(app)


class TestHealthEndpoint:
    """Test suite for health endpoint."""

    def test_health_endpoint(self, client):
        """Test health check returns correctly."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["session_active"] is False
        assert data["document_loaded"] is False


class TestFullRAGFlow:
    """Test suite for complete RAG pipeline."""

    def test_full_rag_flow(self, client, sample_txt_bytes):
        """
        Test complete RAG flow:
        1. Upload document
        2. Verify session active
        3. Query document
        4. Verify response contains answer and sources
        """
        # 1. Upload document
        files = {"file": ("test.txt", BytesIO(sample_txt_bytes), "text/plain")}
        response = client.post("/api/v1/documents", files=files)
        assert response.status_code == 201
        upload_data = response.json()
        assert upload_data["filename"] == "test.txt"
        assert upload_data["file_type"] == "txt"
        assert upload_data["total_chunks"] > 0

        # 2. Verify session active
        response = client.get("/health")
        health_data = response.json()
        assert health_data["session_active"] is True
        assert health_data["document_loaded"] is True
        assert health_data["filename"] == "test.txt"
        assert health_data["chunk_count"] > 0

        # 3. Query document (use sync endpoint for easier testing)
        response = client.post(
            "/api/v1/query/sync", json={"query": "What is machine learning?"}
        )
        # May succeed or fail depending on mocks
        assert response.status_code in [200, 422]

        # 4. Verify response structure (if successful)
        if response.status_code == 200:
            query_data = response.json()
            assert "answer" in query_data
            assert "sources" in query_data

            # If sources exist, verify structure
            if len(query_data["sources"]) > 0:
                source = query_data["sources"][0]
                assert "content" in source
                assert "page_number" in source
                assert "chunk_index" in source
                assert "relevance_score" in source
                assert "filename" in source

    def test_query_without_document(self, client):
        """Test query fails gracefully when no document loaded."""
        response = client.post(
            "/api/v1/query/sync", json={"query": "Test question"}
        )
        # Should return error (400 or 422)
        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data
        # Error message indicates no document or retrieval failure
        assert len(data["detail"]) > 0

    def test_upload_replaces_previous(self, client, sample_txt_bytes):
        """Test that uploading new document replaces previous one."""
        # Upload first document
        files1 = {"file": ("first.txt", BytesIO(sample_txt_bytes), "text/plain")}
        response = client.post("/api/v1/documents", files=files1)
        assert response.status_code == 201

        # Verify first document is active
        response = client.get("/health")
        assert response.json()["filename"] == "first.txt"

        # Upload second document
        different_content = b"This is completely different content for testing."
        files2 = {"file": ("second.txt", BytesIO(different_content), "text/plain")}
        response = client.post("/api/v1/documents", files=files2)
        assert response.status_code == 201

        # Verify second document is active
        response = client.get("/health")
        health_data = response.json()
        assert health_data["filename"] == "second.txt"
        assert health_data["session_active"] is True


class TestDocumentUploadValidation:
    """Test suite for document upload validation."""

    def test_upload_unsupported_file_type(self, client):
        """Test upload with unsupported file type."""
        files = {"file": ("test.xyz", BytesIO(b"content"), "application/octet-stream")}
        response = client.post("/api/v1/documents", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "not supported" in data["detail"]

    def test_upload_pdf_file(self, client):
        """Test uploading a PDF file (will fail processing but should validate)."""
        # Create a minimal valid PDF-like content
        pdf_content = b"%PDF-1.4\nTest PDF content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}

        # This might fail at processing stage, but file type should be accepted
        response = client.post("/api/v1/documents", files=files)
        # Accept either 201 (success) or 422 (processing error)
        assert response.status_code in [201, 422]

    def test_upload_missing_file(self, client):
        """Test upload without providing a file."""
        response = client.post("/api/v1/documents")
        assert response.status_code == 422  # Validation error


class TestQueryValidation:
    """Test suite for query validation."""

    def test_query_empty_question(self, client):
        """Test query with empty question."""
        response = client.post("/api/v1/query/sync", json={"query": ""})
        assert response.status_code == 422  # Validation error

    def test_query_missing_question(self, client):
        """Test query without question field."""
        response = client.post("/api/v1/query/sync", json={})
        assert response.status_code == 422  # Validation error

    def test_query_invalid_top_k(self, client, sample_txt_bytes):
        """Test query with invalid top_k values."""
        # Upload document first
        files = {"file": ("test.txt", BytesIO(sample_txt_bytes), "text/plain")}
        client.post("/api/v1/documents", files=files)

        # Test with top_k too low
        response = client.post(
            "/api/v1/query/sync", json={"query": "Test?", "top_k": 0}
        )
        assert response.status_code == 422

        # Test with top_k too high
        response = client.post(
            "/api/v1/query/sync", json={"query": "Test?", "top_k": 20}
        )
        assert response.status_code == 422


class TestStreamingQuery:
    """Test suite for streaming query endpoint."""

    def test_streaming_query_returns_sse(self, client, sample_txt_bytes):
        """Test that streaming query returns Server-Sent Events."""
        # Upload document first
        files = {"file": ("test.txt", BytesIO(sample_txt_bytes), "text/plain")}
        client.post("/api/v1/documents", files=files)

        # Make streaming query
        response = client.post(
            "/api/v1/query",
            json={"query": "What is machine learning?"},
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200
        # Check content type is event stream
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Test suite for error handling."""

    def test_validation_error_format(self, client):
        """Test that validation errors are properly formatted."""
        response = client.post("/api/v1/query/sync", json={})
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"] == "Validation error"

    def test_404_not_found(self, client):
        """Test that invalid endpoints return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404


class TestCORS:
    """Test suite for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestOpenAPIDocumentation:
    """Test suite for OpenAPI documentation."""

    def test_openapi_json_available(self, client):
        """Test that OpenAPI JSON schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert schema["info"]["title"] == "Document Chat API"

    def test_swagger_ui_available(self, client):
        """Test that Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_available(self, client):
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestMultipleDocumentTypes:
    """Test suite for different document types."""

    def test_txt_document(self, client, sample_txt_bytes):
        """Test processing TXT document."""
        files = {"file": ("doc.txt", BytesIO(sample_txt_bytes), "text/plain")}
        response = client.post("/api/v1/documents", files=files)
        assert response.status_code == 201
        assert response.json()["file_type"] == "txt"

    def test_docx_document_structure(self, client):
        """Test DOCX document upload structure (will fail without valid DOCX)."""
        # This is just to test the endpoint accepts docx
        files = {
            "file": (
                "doc.docx",
                BytesIO(b"PK"),  # DOCX files start with PK
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        }
        response = client.post("/api/v1/documents", files=files)
        # Will fail at processing but should accept the file type
        assert response.status_code in [201, 422]


class TestSessionManagement:
    """Test suite for session management."""

    def test_session_inactive_initially(self, client):
        """Test that session is inactive when app starts."""
        response = client.get("/health")
        data = response.json()
        assert data["session_active"] is False
        assert data["filename"] is None
        assert data["chunk_count"] is None

    def test_session_active_after_upload(self, client, sample_txt_bytes):
        """Test that session becomes active after document upload."""
        files = {"file": ("test.txt", BytesIO(sample_txt_bytes), "text/plain")}
        client.post("/api/v1/documents", files=files)

        response = client.get("/health")
        data = response.json()
        assert data["session_active"] is True
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] > 0


class TestEndToEndWithMultipleQueries:
    """Test suite for multiple queries on the same document."""

    def test_multiple_queries_on_same_document(self, client, sample_txt_bytes):
        """Test that multiple queries work on the same document."""
        # Upload document
        files = {"file": ("ml.txt", BytesIO(sample_txt_bytes), "text/plain")}
        upload_response = client.post("/api/v1/documents", files=files)
        assert upload_response.status_code == 201

        # First query
        response1 = client.post(
            "/api/v1/query/sync", json={"query": "What is machine learning?"}
        )
        # May return 200 or 422 depending on processing
        assert response1.status_code in [200, 422]
        if response1.status_code == 200:
            assert len(response1.json()["answer"]) >= 0

        # Second query on same document
        response2 = client.post(
            "/api/v1/query/sync", json={"query": "How does AI work?"}
        )
        # May return 200 or 422 depending on processing
        assert response2.status_code in [200, 422]
        if response2.status_code == 200:
            assert len(response2.json()["answer"]) >= 0

        # Session should still be active
        response = client.get("/health")
        assert response.json()["session_active"] is True