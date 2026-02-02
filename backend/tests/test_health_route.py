"""Tests for the health check endpoint."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.health import router
from app.core.session import SessionState
from app.models.schemas import HealthResponse


@pytest.fixture
def app():
    """Create a test FastAPI app with the health router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    return MagicMock()


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check_no_session(self, client, mock_session_manager):
        """Test health check when no document is loaded."""
        # Mock inactive session state
        mock_state = SessionState(
            is_active=False,
            filename=None,
            page_count=0,
            chunk_count=0,
            created_at=None,
            collection_name=None,
        )
        mock_session_manager.get_state.return_value = mock_state

        # Override dependency
        from app.api.routes.health import get_session_manager

        client.app.dependency_overrides[get_session_manager] = (
            lambda: mock_session_manager
        )

        # Make request
        response = client.get("/health")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["session_active"] is False
        assert data["document_loaded"] is False
        assert data["filename"] is None
        assert data["page_count"] is None
        assert data["chunk_count"] is None

    def test_health_check_with_active_session(self, client, mock_session_manager):
        """Test health check when a document is loaded."""
        # Mock active session state
        mock_state = SessionState(
            is_active=True,
            filename="test_document.pdf",
            page_count=15,
            chunk_count=42,
            created_at=datetime.utcnow(),
            collection_name="session_abc123",
        )
        mock_session_manager.get_state.return_value = mock_state

        # Override dependency
        from app.api.routes.health import get_session_manager

        client.app.dependency_overrides[get_session_manager] = (
            lambda: mock_session_manager
        )

        # Make request
        response = client.get("/health")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["session_active"] is True
        assert data["document_loaded"] is True
        assert data["filename"] == "test_document.pdf"
        assert data["page_count"] == 15
        assert data["chunk_count"] == 42

    def test_health_check_response_model(self, client, mock_session_manager):
        """Test that response conforms to HealthResponse model."""
        # Mock session state
        mock_state = SessionState(
            is_active=True,
            filename="doc.pdf",
            page_count=10,
            chunk_count=25,
            created_at=datetime.utcnow(),
            collection_name="session_123",
        )
        mock_session_manager.get_state.return_value = mock_state

        # Override dependency
        from app.api.routes.health import get_session_manager

        client.app.dependency_overrides[get_session_manager] = (
            lambda: mock_session_manager
        )

        # Make request
        response = client.get("/health")

        # Verify response can be parsed as HealthResponse
        assert response.status_code == 200
        health_response = HealthResponse(**response.json())

        assert isinstance(health_response, HealthResponse)
        assert health_response.status == "healthy"
        assert health_response.version == "0.1.0"

    def test_health_check_calls_session_manager(self, client, mock_session_manager):
        """Test that health check calls session manager get_state."""
        # Mock session state
        mock_state = SessionState()
        mock_session_manager.get_state.return_value = mock_state

        # Override dependency
        from app.api.routes.health import get_session_manager

        client.app.dependency_overrides[get_session_manager] = (
            lambda: mock_session_manager
        )

        # Make request
        response = client.get("/health")

        # Verify session manager was called
        assert response.status_code == 200
        mock_session_manager.get_state.assert_called_once()

    def test_health_check_partial_session_data(self, client, mock_session_manager):
        """Test health check with partial session data."""
        # Mock session with default/zero values
        mock_state = SessionState(
            is_active=True,
            filename="doc.txt",
            page_count=0,  # Zero page count (default)
            chunk_count=10,
            created_at=datetime.utcnow(),
            collection_name="session_xyz",
        )
        mock_session_manager.get_state.return_value = mock_state

        # Override dependency
        from app.api.routes.health import get_session_manager

        client.app.dependency_overrides[get_session_manager] = (
            lambda: mock_session_manager
        )

        # Make request
        response = client.get("/health")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["session_active"] is True
        assert data["filename"] == "doc.txt"
        assert data["page_count"] == 0
        assert data["chunk_count"] == 10

    def test_health_check_openapi_metadata(self, app):
        """Test that health endpoint has proper OpenAPI metadata."""
        openapi_schema = app.openapi()

        # Check endpoint exists in schema
        assert "/health" in openapi_schema["paths"]
        health_endpoint = openapi_schema["paths"]["/health"]["get"]

        # Check metadata
        assert health_endpoint["summary"] == "Health check"
        assert "health" in health_endpoint["tags"]
        assert "API health and session status" in health_endpoint["description"]

        # Check response schema
        assert "200" in health_endpoint["responses"]
        response_schema = health_endpoint["responses"]["200"]
        assert "application/json" in response_schema["content"]


class TestHealthResponseModel:
    """Test suite for HealthResponse model."""

    def test_health_response_minimal(self):
        """Test creating HealthResponse with minimal data."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            session_active=False,
            document_loaded=False,
        )

        assert response.status == "healthy"
        assert response.version == "0.1.0"
        assert response.session_active is False
        assert response.document_loaded is False
        assert response.filename is None
        assert response.page_count is None
        assert response.chunk_count is None

    def test_health_response_full(self):
        """Test creating HealthResponse with all fields."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            session_active=True,
            document_loaded=True,
            filename="test.pdf",
            page_count=20,
            chunk_count=50,
        )

        assert response.status == "healthy"
        assert response.version == "0.1.0"
        assert response.session_active is True
        assert response.document_loaded is True
        assert response.filename == "test.pdf"
        assert response.page_count == 20
        assert response.chunk_count == 50

    def test_health_response_defaults(self):
        """Test HealthResponse default values."""
        response = HealthResponse(session_active=False, document_loaded=False)

        assert response.status == "healthy"
        assert response.version == "0.1.0"

    def test_health_response_serialization(self):
        """Test HealthResponse can be serialized to JSON."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            session_active=True,
            document_loaded=True,
            filename="doc.pdf",
            page_count=10,
            chunk_count=25,
        )

        # Serialize to dict
        data = response.model_dump()

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["session_active"] is True
        assert data["filename"] == "doc.pdf"

        # Serialize to JSON
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "healthy" in json_str