"""Health check endpoint."""

from fastapi import APIRouter, Depends

from app.core.session import SessionManager, get_session_manager
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and session status",
)
async def health_check(
    session_manager: SessionManager = Depends(get_session_manager),
) -> HealthResponse:
    """
    Returns API health status and current session info.

    This endpoint can be used to:
    - Verify the API is running
    - Check if a document is currently loaded
    - Get information about the loaded document

    Returns:
        HealthResponse with API status and session information
    """
    state = session_manager.get_state()

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        session_active=state.is_active,
        document_loaded=state.is_active,
        filename=state.filename,
        page_count=state.page_count if state.is_active else None,
        chunk_count=state.chunk_count if state.is_active else None,
    )