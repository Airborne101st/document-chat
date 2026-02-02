"""Tests for the query service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import NoActiveSessionError, QueryError
from app.models.schemas import Source
from app.services.query_service import (
    QueryContext,
    QueryService,
    StreamingQueryResponse,
    get_query_service,
)
from app.services.vector_store_service import RetrievedChunk


class TestQueryContext:
    """Test suite for QueryContext model."""

    def test_valid_query_context(self):
        """Test creating a valid query context."""
        chunks = [
            RetrievedChunk(
                content="Test content",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                source="test.pdf",
            )
        ]

        context = QueryContext(chunks=chunks, formatted_context="Formatted text")

        assert len(context.chunks) == 1
        assert context.formatted_context == "Formatted text"


class TestStreamingQueryResponse:
    """Test suite for StreamingQueryResponse model."""

    def test_token_response(self):
        """Test creating a token response."""
        response = StreamingQueryResponse(token="Hello")

        assert response.token == "Hello"
        assert response.sources is None
        assert response.is_done is False
        assert response.error is None

    def test_sources_response(self):
        """Test creating a sources response."""
        sources = [
            Source(
                content="Content",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                filename="test.pdf",
            )
        ]

        response = StreamingQueryResponse(sources=sources)

        assert response.token is None
        assert len(response.sources) == 1
        assert response.is_done is False

    def test_done_response(self):
        """Test creating a done signal response."""
        response = StreamingQueryResponse(is_done=True)

        assert response.token is None
        assert response.sources is None
        assert response.is_done is True

    def test_error_response(self):
        """Test creating an error response."""
        response = StreamingQueryResponse(error="Something went wrong")

        assert response.token is None
        assert response.error == "Something went wrong"


class TestQueryService:
    """Test suite for QueryService."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store service."""
        return MagicMock()

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service."""
        return MagicMock()

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        manager = MagicMock()
        manager.is_active = True
        manager.collection_name = "test_collection"
        return manager

    @pytest.fixture
    def query_service(self, mock_vector_store, mock_llm, mock_session_manager):
        """Create QueryService instance with mocked dependencies."""
        return QueryService(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm,
            session_manager=mock_session_manager,
            top_k=3,
        )

    def test_initialization(
        self, mock_vector_store, mock_llm, mock_session_manager
    ):
        """Test service initialization."""
        service = QueryService(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm,
            session_manager=mock_session_manager,
            top_k=5,
        )

        assert service._vector_store == mock_vector_store
        assert service._llm == mock_llm
        assert service._session_manager == mock_session_manager
        assert service._top_k == 5

    def test_retrieve_context(self, query_service, mock_vector_store):
        """Test retrieving and formatting context."""
        # Mock retrieved chunks
        chunks = [
            RetrievedChunk(
                content="First chunk content",
                page_number=1,
                chunk_index=0,
                relevance_score=0.95,
                source="test.pdf",
            ),
            RetrievedChunk(
                content="Second chunk content",
                page_number=2,
                chunk_index=1,
                relevance_score=0.87,
                source="test.pdf",
            ),
        ]
        mock_vector_store.query.return_value = chunks

        # Retrieve context
        context = query_service._retrieve_context("What is AI?", "test_collection")

        # Verify
        mock_vector_store.query.assert_called_once_with(
            collection_name="test_collection", query_text="What is AI?", top_k=3
        )
        assert len(context.chunks) == 2
        assert "First chunk content" in context.formatted_context
        assert "Second chunk content" in context.formatted_context
        assert "Page 1" in context.formatted_context
        assert "Page 2" in context.formatted_context
        assert "Score: 0.95" in context.formatted_context
        assert "Score: 0.87" in context.formatted_context

    def test_build_prompt(self, query_service):
        """Test building prompt with context."""
        chunks = [
            RetrievedChunk(
                content="AI is intelligence demonstrated by machines",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                source="test.pdf",
            )
        ]
        context = QueryContext(
            chunks=chunks,
            formatted_context="[Chunk 1]\nAI is intelligence demonstrated by machines",
        )

        prompt = query_service._build_prompt("What is AI?", context)

        assert "Context from document:" in prompt
        assert "AI is intelligence demonstrated by machines" in prompt
        assert "Question: What is AI?" in prompt
        assert "Answer:" in prompt

    def test_chunks_to_sources(self, query_service):
        """Test converting chunks to sources."""
        chunks = [
            RetrievedChunk(
                content="Content 1",
                page_number=1,
                chunk_index=0,
                relevance_score=0.95,
                source="test.pdf",
            ),
            RetrievedChunk(
                content="Content 2",
                page_number=2,
                chunk_index=1,
                relevance_score=0.87,
                source="test.pdf",
            ),
        ]

        sources = query_service._chunks_to_sources(chunks)

        assert len(sources) == 2
        assert all(isinstance(s, Source) for s in sources)
        assert sources[0].content == "Content 1"
        assert sources[0].page_number == 1
        assert sources[0].chunk_index == 0
        assert sources[0].relevance_score == 0.95
        assert sources[0].filename == "test.pdf"
        assert sources[1].content == "Content 2"

    @pytest.mark.asyncio
    async def test_query_stream_no_active_session(
        self, query_service, mock_session_manager
    ):
        """Test query fails when no session is active."""
        mock_session_manager.is_active = False

        with pytest.raises(NoActiveSessionError) as exc_info:
            async for _ in query_service.query_stream("What is AI?"):
                pass

        assert "No document loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_stream_no_collection_name(
        self, query_service, mock_session_manager
    ):
        """Test query fails when collection name is None."""
        mock_session_manager.is_active = True
        mock_session_manager.collection_name = None

        with pytest.raises(NoActiveSessionError):
            async for _ in query_service.query_stream("What is AI?"):
                pass

    @pytest.mark.asyncio
    async def test_query_stream_success(
        self, query_service, mock_vector_store, mock_llm
    ):
        """Test successful streaming query."""
        # Mock retrieved chunks
        chunks = [
            RetrievedChunk(
                content="AI is artificial intelligence",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                source="test.pdf",
            )
        ]
        mock_vector_store.query.return_value = chunks

        # Mock LLM streaming
        async def mock_stream(prompt, system_prompt):
            tokens = ["AI", " is", " a", " technology"]
            for token in tokens:
                yield token

        mock_llm.generate_stream = mock_stream

        # Execute query
        responses = []
        async for response in query_service.query_stream("What is AI?"):
            responses.append(response)

        # Verify responses
        # Should have: 4 tokens + 1 sources + 1 done
        assert len(responses) == 6

        # Check tokens
        assert responses[0].token == "AI"
        assert responses[1].token == " is"
        assert responses[2].token == " a"
        assert responses[3].token == " technology"

        # Check sources
        assert responses[4].sources is not None
        assert len(responses[4].sources) == 1
        assert responses[4].sources[0].content == "AI is artificial intelligence"

        # Check done signal
        assert responses[5].is_done is True

    @pytest.mark.asyncio
    async def test_query_stream_retrieval_error(
        self, query_service, mock_vector_store
    ):
        """Test query handles retrieval errors."""
        mock_vector_store.query.side_effect = Exception("Vector store error")

        with pytest.raises(QueryError) as exc_info:
            async for _ in query_service.query_stream("What is AI?"):
                pass

        assert "Failed to retrieve relevant context" in str(exc_info.value)
        assert exc_info.value.original_error is not None

    @pytest.mark.asyncio
    async def test_query_stream_llm_error(
        self, query_service, mock_vector_store, mock_llm
    ):
        """Test query handles LLM errors."""
        # Mock successful retrieval
        chunks = [
            RetrievedChunk(
                content="Test",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                source="test.pdf",
            )
        ]
        mock_vector_store.query.return_value = chunks

        # Mock LLM error
        from app.core.exceptions import LLMError

        async def mock_stream_error(prompt, system_prompt):
            raise LLMError("LLM generation failed")
            yield  # Make it a generator

        mock_llm.generate_stream = mock_stream_error

        with pytest.raises(QueryError) as exc_info:
            async for _ in query_service.query_stream("What is AI?"):
                pass

        assert "Failed to generate response from LLM" in str(exc_info.value)
        assert isinstance(exc_info.value.original_error, LLMError)

    @pytest.mark.asyncio
    async def test_query_stream_context_formatting(
        self, query_service, mock_vector_store, mock_llm
    ):
        """Test that context is properly formatted in the prompt."""
        # Mock retrieved chunks
        chunks = [
            RetrievedChunk(
                content="First chunk",
                page_number=1,
                chunk_index=0,
                relevance_score=0.95,
                source="doc.pdf",
            ),
            RetrievedChunk(
                content="Second chunk",
                page_number=2,
                chunk_index=1,
                relevance_score=0.87,
                source="doc.pdf",
            ),
        ]
        mock_vector_store.query.return_value = chunks

        # Track the prompt that was sent to LLM
        captured_prompt = None

        async def mock_stream(prompt, system_prompt):
            nonlocal captured_prompt
            captured_prompt = prompt
            yield "Answer"

        mock_llm.generate_stream = mock_stream

        # Execute query
        async for _ in query_service.query_stream("Test question"):
            pass

        # Verify prompt formatting
        assert captured_prompt is not None
        assert "First chunk" in captured_prompt
        assert "Second chunk" in captured_prompt
        assert "Page 1" in captured_prompt
        assert "Page 2" in captured_prompt
        assert "Test question" in captured_prompt

    @pytest.mark.asyncio
    async def test_query_stream_uses_system_prompt(
        self, query_service, mock_vector_store, mock_llm
    ):
        """Test that system prompt is used."""
        # Mock successful retrieval
        chunks = [
            RetrievedChunk(
                content="Test",
                page_number=1,
                chunk_index=0,
                relevance_score=0.9,
                source="test.pdf",
            )
        ]
        mock_vector_store.query.return_value = chunks

        # Track system prompt
        captured_system_prompt = None

        async def mock_stream(prompt, system_prompt):
            nonlocal captured_system_prompt
            captured_system_prompt = system_prompt
            yield "Answer"

        mock_llm.generate_stream = mock_stream

        # Execute query
        async for _ in query_service.query_stream("Question"):
            pass

        # Verify system prompt was used
        assert captured_system_prompt is not None
        assert "helpful assistant" in captured_system_prompt
        assert "based on the provided document context" in captured_system_prompt


class TestDependencyFunction:
    """Test suite for FastAPI dependency function."""

    @patch("app.services.query_service.QueryService")
    def test_get_query_service(self, mock_service_class):
        """Test the FastAPI dependency function."""
        # Mock dependencies
        mock_vector_store = MagicMock()
        mock_llm = MagicMock()
        mock_session_manager = MagicMock()
        mock_settings = MagicMock()
        mock_settings.top_k_chunks = 5

        # Mock service instance
        mock_service_instance = MagicMock()
        mock_service_class.return_value = mock_service_instance

        # Get service
        service = get_query_service(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm,
            session_manager=mock_session_manager,
            settings=mock_settings,
        )

        # Verify QueryService was initialized with correct params
        mock_service_class.assert_called_once_with(
            vector_store_service=mock_vector_store,
            llm_service=mock_llm,
            session_manager=mock_session_manager,
            top_k=5,
        )
        assert service == mock_service_instance