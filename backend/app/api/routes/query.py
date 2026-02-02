"""Query endpoint with Server-Sent Events streaming."""

import json
import logging

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.core.exceptions import NoActiveSessionError, QueryError
from app.models.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.services.query_service import QueryService, get_query_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    summary="Query the document",
    description="Ask a question about the uploaded document (streaming response)",
    responses={
        200: {"description": "Streaming response with answer and sources"},
        400: {"model": ErrorResponse, "description": "No document loaded"},
        422: {"model": ErrorResponse, "description": "Query processing failed"},
    },
)
async def query_document(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    """
    Query the loaded document with a question.

    Returns Server-Sent Events stream with:
    - event: "token" - Individual tokens as they're generated
    - event: "sources" - Retrieved source chunks with page numbers
    - event: "done" - Signals completion
    - event: "error" - Error information if something fails

    Args:
        request: Query request with question and optional top_k
        query_service: Injected query service

    Returns:
        EventSourceResponse streaming the answer and sources
    """

    async def event_generator():
        """Generate SSE events from the query stream."""
        logger.info(f"Starting SSE stream for query: '{request.query[:100]}'")
        event_count = 0
        try:
            async for response in query_service.query_stream(request.query):
                event_count += 1

                if response.error:
                    logger.error(f"Query error in stream: {response.error}")
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": response.error}),
                    }
                    return

                if response.token is not None:
                    logger.debug(f"Sending token event #{event_count}")
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": response.token}),
                    }

                if response.sources is not None:
                    logger.info(f"Sending {len(response.sources)} sources")
                    yield {
                        "event": "sources",
                        "data": json.dumps(
                            {"sources": [s.model_dump() for s in response.sources]}
                        ),
                    }

                if response.is_done:
                    logger.info(f"Query stream completed ({event_count} events total)")
                    yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except NoActiveSessionError as e:
            logger.error(f"NoActiveSessionError in SSE stream: {e}")
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        except QueryError as e:
            logger.error(f"QueryError in SSE stream: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        except Exception as e:
            logger.error(f"Unexpected error in SSE stream: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": "An unexpected error occurred"}),
            }

    return EventSourceResponse(event_generator())


@router.post(
    "/sync",
    response_model=QueryResponse,
    summary="Query document (non-streaming)",
    description="Synchronous version for testing",
    include_in_schema=False,  # Hide from docs, just for testing
)
async def query_document_sync(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """
    Non-streaming query for testing purposes.

    Collects all tokens from the stream and returns a complete response.

    Args:
        request: Query request with question
        query_service: Injected query service

    Returns:
        QueryResponse with complete answer and sources

    Raises:
        HTTPException: If no document is loaded or query fails
    """
    # Collect all tokens and sources from stream
    tokens = []
    sources = []

    try:
        async for response in query_service.query_stream(request.query):
            if response.token:
                tokens.append(response.token)
            if response.sources:
                sources = response.sources
    except NoActiveSessionError as e:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except QueryError as e:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )

    return QueryResponse(answer="".join(tokens), sources=sources)