"""Main FastAPI application."""

import logging
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, query
from app.config import get_settings
from app.core.logging_config import setup_logging
from app.models.schemas import ErrorResponse

# Initialize logging first
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
    - Log startup message
    - Validate configuration (API key present)

    Shutdown:
    - Cleanup any resources
    - Log shutdown message
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Starting Document Chat API v0.1.0")
    logger.info("=" * 80)
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"ChromaDB persist directory: {settings.chroma_persist_dir}")
    logger.info(f"Max file size: {settings.max_file_size_mb}MB")
    logger.info(f"Max page limit: {settings.max_page_limit}")
    logger.info(f"Chunk size: {settings.chunk_size_tokens} tokens")
    logger.info(f"Top-K chunks: {settings.top_k_chunks}")

    # Validate required config
    if not settings.gemini_api_key:
        logger.warning("⚠️  GEMINI_API_KEY not set. API calls will fail!")
    else:
        logger.info(f"✓ Gemini API key configured ({settings.gemini_api_key[:10]}...)")

    logger.info("Application startup complete")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("Shutting down Document Chat API")
    logger.info("=" * 80)


app = FastAPI(
    title="Document Chat API",
    description="RAG-based Document Q&A API using Google Gemini",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses with timing."""
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    if request.query_params:
        logger.debug(f"  Query params: {dict(request.query_params)}")

    try:
        response = await call_next(request)

        # Calculate duration
        duration = (time.time() - start_time) * 1000

        # Log response
        status_emoji = "✓" if response.status_code < 400 else "✗"
        logger.info(
            f"← {status_emoji} {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration:.2f}ms)"
        )

        return response
    except Exception as exc:
        duration = (time.time() - start_time) * 1000
        logger.error(
            f"← ✗ {request.method} {request.url.path} "
            f"→ ERROR ({duration:.2f}ms): {exc}"
        )
        raise


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error", detail=str(exc.errors())
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected errors."""
    # Log the full exception with traceback
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )

    if settings.debug:
        # Include full traceback in debug mode
        tb = traceback.format_exc()
        logger.debug(f"Traceback:\n{tb}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else None,
        ).model_dump(),
    )