"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document.

    Attributes:
        filename: Name of the uploaded file.
        file_type: File extension.
        total_chunks: Number of chunks created.
        message: Success message.
    """

    filename: str
    file_type: str
    total_chunks: int
    message: str = "Document uploaded and processed successfully"


class QueryRequest(BaseModel):
    """Request schema for querying documents.

    Attributes:
        query: User's question or query text.
        top_k: Number of relevant chunks to retrieve (optional).
    """

    query: str = Field(..., min_length=1, description="User's question")
    top_k: int | None = Field(None, ge=1, le=10, description="Number of chunks to retrieve")


class Source(BaseModel):
    """Source chunk used to generate the answer.

    Attributes:
        content: The text content of the chunk.
        page_number: Page number in the document.
        chunk_index: Index of the chunk in the sequence.
        relevance_score: Similarity score (0-1).
        filename: Source document filename.
    """

    content: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    relevance_score: float = Field(..., ge=0, le=1)
    filename: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "The document states that...",
                "page_number": 3,
                "chunk_index": 7,
                "relevance_score": 0.89,
                "filename": "report.pdf",
            }
        }
    )


class QueryResponse(BaseModel):
    """Response schema for query results.

    Attributes:
        answer: Generated answer from the LLM.
        sources: List of source documents used.
    """

    answer: str
    sources: list[Source] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response schema.

    Attributes:
        error: Error type or category.
        detail: Detailed error message (optional).
    """

    error: str
    detail: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"error": "Validation error", "detail": "Invalid input"}
        }
    )


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status.
        version: API version.
        session_active: Whether a session is currently active.
        document_loaded: Whether a document is currently loaded.
        filename: Name of loaded document (if any).
        page_count: Number of pages in loaded document (if any).
        chunk_count: Number of chunks in loaded document (if any).
    """

    status: str = "healthy"
    version: str = "0.1.0"
    session_active: bool = False
    document_loaded: bool = False
    filename: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None