"""Tests for the main FastAPI application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestApplicationSetup:
    """Test suite for application setup and configuration."""

    @patch("app.main.settings")
    def test_app_creation(self, mock_settings):
        """Test that app is created with correct configuration."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        assert app.title == "Document Chat API"
        assert app.version == "0.1.0"
        assert "RAG-based" in app.description

    @patch("app.main.settings")
    def test_app_has_docs(self, mock_settings):
        """Test that docs endpoints are configured."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    @patch("app.main.settings")
    def test_cors_middleware_configured(self, mock_settings):
        """Test that CORS middleware is configured."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        # Check that CORS middleware is added
        # Middleware may be wrapped, so check for non-empty middleware list
        assert len(app.user_middleware) > 0

    @patch("app.main.settings")
    def test_routers_included(self, mock_settings):
        """Test that all routers are included."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        # Get all routes
        routes = [route.path for route in app.routes]

        # Check health route
        assert "/health" in routes

        # Check documents routes
        assert "/api/v1/documents" in routes

        # Check query routes
        assert "/api/v1/query" in routes


class TestLifespan:
    """Test suite for application lifespan."""

    @patch("builtins.print")
    @patch("app.main.settings")
    def test_lifespan_startup_with_api_key(self, mock_settings, mock_print):
        """Test lifespan startup with API key configured."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        # Create client to trigger lifespan
        with TestClient(app):
            pass

        # Check startup message was printed
        mock_print.assert_any_call("Starting document-chat API v0.1.0")

    @patch("builtins.print")
    @patch("app.main.settings")
    def test_lifespan_startup_without_api_key(self, mock_settings, mock_print):
        """Test lifespan startup without API key shows warning."""
        mock_settings.gemini_api_key = ""
        mock_settings.debug = False

        from app.main import app

        # Create client to trigger lifespan
        with TestClient(app):
            pass

        # Check warning message was printed
        mock_print.assert_any_call(
            "WARNING: GEMINI_API_KEY not set. API calls will fail."
        )

    @patch("builtins.print")
    @patch("app.main.settings")
    def test_lifespan_shutdown(self, mock_settings, mock_print):
        """Test lifespan shutdown message."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        # Create and close client to trigger shutdown
        with TestClient(app):
            pass

        # Check shutdown message was printed
        mock_print.assert_any_call("Shutting down document-chat API")


class TestExceptionHandlers:
    """Test suite for global exception handlers."""

    @patch("app.main.settings")
    def test_validation_error_handler(self, mock_settings):
        """Test validation error handler."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)

        # Make request with invalid data (empty query)
        response = client.post("/api/v1/query", json={})

        # Check error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "error" in data
        assert data["error"] == "Validation error"
        assert "detail" in data

    @pytest.mark.asyncio
    @patch("app.main.settings")
    async def test_general_exception_handler_debug_false(self, mock_settings):
        """Test general exception handler with debug=False."""
        from app.main import general_exception_handler
        from fastapi import Request

        mock_settings.debug = False

        # Create a mock request
        request = MagicMock(spec=Request)
        exc = ValueError("Test error")

        # Call the handler directly
        response = await general_exception_handler(request, exc)

        # Check error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        import json
        data = json.loads(response.body.decode())
        assert "error" in data
        assert data["error"] == "Internal server error"
        assert data["detail"] is None  # Should be None when debug=False

    @pytest.mark.asyncio
    @patch("app.main.settings")
    async def test_general_exception_handler_debug_true(self, mock_settings):
        """Test general exception handler with debug=True."""
        from app.main import general_exception_handler
        from fastapi import Request

        mock_settings.debug = True

        # Create a mock request
        request = MagicMock(spec=Request)
        exc = ValueError("Test error message")

        # Call the handler directly
        response = await general_exception_handler(request, exc)

        # Check error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        import json
        data = json.loads(response.body.decode())
        assert "error" in data
        assert data["error"] == "Internal server error"
        assert data["detail"] is not None  # Should have detail when debug=True
        assert "Test error message" in data["detail"]


class TestCORS:
    """Test suite for CORS configuration."""

    @patch("app.main.settings")
    def test_cors_headers(self, mock_settings):
        """Test that CORS headers are set correctly."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)

        # Make OPTIONS request (preflight)
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    @patch("app.main.settings")
    def test_cors_allows_credentials(self, mock_settings):
        """Test that CORS allows credentials."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)

        # Make request with origin
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})

        # Check credentials header
        assert "access-control-allow-credentials" in response.headers
        assert response.headers["access-control-allow-credentials"] == "true"


class TestOpenAPISchema:
    """Test suite for OpenAPI schema."""

    @patch("app.main.settings")
    def test_openapi_schema_generated(self, mock_settings):
        """Test that OpenAPI schema is generated correctly."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)
        response = client.get("/openapi.json")

        assert response.status_code == status.HTTP_200_OK
        schema = response.json()

        # Check basic schema properties
        assert schema["info"]["title"] == "Document Chat API"
        assert schema["info"]["version"] == "0.1.0"

    @patch("app.main.settings")
    def test_openapi_has_all_endpoints(self, mock_settings):
        """Test that OpenAPI schema includes all endpoints."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)
        response = client.get("/openapi.json")

        schema = response.json()
        paths = schema["paths"]

        # Check health endpoint
        assert "/health" in paths

        # Check documents endpoint
        assert "/api/v1/documents" in paths

        # Check query endpoint
        assert "/api/v1/query" in paths

    @patch("app.main.settings")
    def test_docs_accessible(self, mock_settings):
        """Test that Swagger docs are accessible."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)
        response = client.get("/docs")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]

    @patch("app.main.settings")
    def test_redoc_accessible(self, mock_settings):
        """Test that ReDoc is accessible."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.debug = False

        from app.main import app

        client = TestClient(app)
        response = client.get("/redoc")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]


class TestErrorResponseModel:
    """Test suite for updated ErrorResponse model."""

    def test_error_response_with_detail(self):
        """Test ErrorResponse with both error and detail."""
        from app.models.schemas import ErrorResponse

        error = ErrorResponse(error="Validation error", detail="Field is required")

        assert error.error == "Validation error"
        assert error.detail == "Field is required"

    def test_error_response_without_detail(self):
        """Test ErrorResponse with only error."""
        from app.models.schemas import ErrorResponse

        error = ErrorResponse(error="Internal error")

        assert error.error == "Internal error"
        assert error.detail is None

    def test_error_response_serialization(self):
        """Test ErrorResponse serialization."""
        from app.models.schemas import ErrorResponse

        error = ErrorResponse(error="Test error", detail="Details here")
        data = error.model_dump()

        assert data["error"] == "Test error"
        assert data["detail"] == "Details here"