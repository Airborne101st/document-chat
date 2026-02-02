"""Tests for the query endpoint."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from app.api.routes.query import router
from app.core.exceptions import NoActiveSessionError, QueryError
from app.models.schemas import QueryResponse, Source
from app.services.query_service import StreamingQueryResponse


@pytest.fixture
def app():
    """Create a test FastAPI app with the query router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_query_service():
    """Create a mock query service."""
    return MagicMock()


class TestQueryDocumentStreamingEndpoint:
    """Test suite for streaming query endpoint."""

    def test_query_streaming_success(self, client, mock_query_service):
        """Test successful streaming query."""

        # Mock streaming response
        async def mock_stream(question):
            # Yield tokens
            yield StreamingQueryResponse(token="Hello")
            yield StreamingQueryResponse(token=" ")
            yield StreamingQueryResponse(token="world")

            # Yield sources
            sources = [
                Source(
                    content="Test content",
                    page_number=1,
                    chunk_index=0,
                    relevance_score=0.9,
                    filename="test.pdf",
                )
            ]
            yield StreamingQueryResponse(sources=sources)

            # Yield done
            yield StreamingQueryResponse(is_done=True)

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post(
            "/query", json={"query": "What is AI?"}, headers={"Accept": "text/event-stream"}
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        events = self._parse_sse_events(response.text)

        # Check tokens
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 3
        assert json.loads(token_events[0]["data"])["token"] == "Hello"
        assert json.loads(token_events[1]["data"])["token"] == " "
        assert json.loads(token_events[2]["data"])["token"] == "world"

        # Check sources
        source_events = [e for e in events if e["event"] == "sources"]
        assert len(source_events) == 1
        sources_data = json.loads(source_events[0]["data"])
        assert len(sources_data["sources"]) == 1
        assert sources_data["sources"][0]["content"] == "Test content"

        # Check done
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert json.loads(done_events[0]["data"])["status"] == "complete"

    def test_query_streaming_no_active_session(self, client, mock_query_service):
        """Test streaming query with no active session."""

        # Mock NoActiveSessionError
        async def mock_stream(question):
            raise NoActiveSessionError()
            yield  # Make it a generator

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query", json={"query": "What is AI?"})

        # Verify response
        assert response.status_code == status.HTTP_200_OK

        # Parse SSE events
        events = self._parse_sse_events(response.text)

        # Check error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        error_data = json.loads(error_events[0]["data"])
        assert "No document loaded" in error_data["error"]

    def test_query_streaming_query_error(self, client, mock_query_service):
        """Test streaming query with QueryError."""

        # Mock QueryError
        async def mock_stream(question):
            raise QueryError("Failed to retrieve context")
            yield  # Make it a generator

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query", json={"query": "What is AI?"})

        # Verify response
        assert response.status_code == status.HTTP_200_OK

        # Parse SSE events
        events = self._parse_sse_events(response.text)

        # Check error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        error_data = json.loads(error_events[0]["data"])
        assert "Failed to retrieve context" in error_data["error"]

    def test_query_streaming_unexpected_error(self, client, mock_query_service):
        """Test streaming query with unexpected error."""

        # Mock unexpected error
        async def mock_stream(question):
            raise Exception("Unexpected error")
            yield  # Make it a generator

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query", json={"query": "What is AI?"})

        # Verify response
        assert response.status_code == status.HTTP_200_OK

        # Parse SSE events
        events = self._parse_sse_events(response.text)

        # Check error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        error_data = json.loads(error_events[0]["data"])
        assert "unexpected error occurred" in error_data["error"]

    def test_query_streaming_error_in_response(self, client, mock_query_service):
        """Test streaming query with error in response object."""

        # Mock streaming response with error
        async def mock_stream(question):
            yield StreamingQueryResponse(error="Something went wrong")

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query", json={"query": "What is AI?"})

        # Verify response
        assert response.status_code == status.HTTP_200_OK

        # Parse SSE events
        events = self._parse_sse_events(response.text)

        # Check error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        error_data = json.loads(error_events[0]["data"])
        assert "Something went wrong" in error_data["error"]

    def test_query_invalid_request(self, client):
        """Test query with invalid request body."""
        # Make request with missing query field
        response = client.post("/query", json={})

        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def _parse_sse_events(self, text):
        """Parse SSE events from response text."""
        events = []
        lines = text.strip().split("\n")
        current_event = {}

        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line[6:].strip()
            elif line.startswith("data:"):
                current_event["data"] = line[5:].strip()
                events.append(current_event)
                current_event = {}

        return events


class TestQueryDocumentSyncEndpoint:
    """Test suite for synchronous query endpoint."""

    def test_query_sync_success(self, client, mock_query_service):
        """Test successful synchronous query."""

        # Mock streaming response
        async def mock_stream(question):
            yield StreamingQueryResponse(token="AI")
            yield StreamingQueryResponse(token=" is")
            yield StreamingQueryResponse(token=" technology")

            sources = [
                Source(
                    content="AI definition",
                    page_number=1,
                    chunk_index=0,
                    relevance_score=0.95,
                    filename="ai.pdf",
                )
            ]
            yield StreamingQueryResponse(sources=sources)
            yield StreamingQueryResponse(is_done=True)

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query/sync", json={"query": "What is AI?"})

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check answer
        assert data["answer"] == "AI is technology"

        # Check sources
        assert len(data["sources"]) == 1
        assert data["sources"][0]["content"] == "AI definition"
        assert data["sources"][0]["page_number"] == 1
        assert data["sources"][0]["relevance_score"] == 0.95

    def test_query_sync_no_active_session(self, client, mock_query_service):
        """Test sync query with no active session."""

        # Mock NoActiveSessionError
        async def mock_stream(question):
            raise NoActiveSessionError()
            yield  # Make it a generator

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query/sync", json={"query": "What is AI?"})

        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "No document loaded" in data["detail"]

    def test_query_sync_query_error(self, client, mock_query_service):
        """Test sync query with QueryError."""

        # Mock QueryError
        async def mock_stream(question):
            raise QueryError("LLM generation failed")
            yield  # Make it a generator

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query/sync", json={"query": "What is AI?"})

        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert "LLM generation failed" in data["detail"]

    def test_query_sync_empty_response(self, client, mock_query_service):
        """Test sync query with empty response."""

        # Mock empty stream
        async def mock_stream(question):
            yield StreamingQueryResponse(is_done=True)

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query/sync", json={"query": "What is AI?"})

        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check empty answer and sources
        assert data["answer"] == ""
        assert data["sources"] == []

    def test_query_sync_not_in_schema(self, app):
        """Test that sync endpoint is not included in OpenAPI schema."""
        openapi_schema = app.openapi()

        # Check that /query/sync is not in schema (include_in_schema=False)
        assert "/query/sync" not in openapi_schema["paths"]

    def test_query_sync_response_model(self, client, mock_query_service):
        """Test that sync response conforms to QueryResponse model."""

        # Mock streaming response
        async def mock_stream(question):
            yield StreamingQueryResponse(token="Answer")

            sources = [
                Source(
                    content="Content",
                    page_number=1,
                    chunk_index=0,
                    relevance_score=0.8,
                    filename="doc.pdf",
                )
            ]
            yield StreamingQueryResponse(sources=sources)
            yield StreamingQueryResponse(is_done=True)

        mock_query_service.query_stream = mock_stream

        # Override dependency
        from app.api.routes.query import get_query_service

        client.app.dependency_overrides[get_query_service] = (
            lambda: mock_query_service
        )

        # Make request
        response = client.post("/query/sync", json={"query": "Test?"})

        # Verify response can be parsed as QueryResponse
        assert response.status_code == status.HTTP_200_OK
        query_response = QueryResponse(**response.json())

        assert isinstance(query_response, QueryResponse)
        assert query_response.answer == "Answer"
        assert len(query_response.sources) == 1


class TestQueryEndpointMetadata:
    """Test suite for query endpoint metadata."""

    def test_query_endpoint_openapi_metadata(self, app):
        """Test that query endpoint has proper OpenAPI metadata."""
        openapi_schema = app.openapi()

        # Check endpoint exists in schema
        assert "/query" in openapi_schema["paths"]
        query_endpoint = openapi_schema["paths"]["/query"]["post"]

        # Check metadata
        assert query_endpoint["summary"] == "Query the document"
        assert "query" in query_endpoint["tags"]
        assert "streaming response" in query_endpoint["description"]

        # Check responses
        assert "200" in query_endpoint["responses"]
        assert "400" in query_endpoint["responses"]
        assert "422" in query_endpoint["responses"]

    def test_query_request_validation(self, client):
        """Test query request validation."""
        # Empty query
        response = client.post("/query", json={"query": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid top_k (too low)
        response = client.post("/query", json={"query": "Test?", "top_k": 0})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid top_k (too high)
        response = client.post("/query", json={"query": "Test?", "top_k": 20})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY