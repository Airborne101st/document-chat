"""Document upload endpoint."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.exceptions import (
    FileTooLargeError,
    PageLimitExceededError,
    ProcessingError,
    UnsupportedFileTypeError,
)
from app.models.schemas import DocumentUploadResponse, ErrorResponse
from app.services.document_service import DocumentService, get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    description="Upload a PDF, DOCX, or TXT file for Q&A",
    responses={
        201: {"description": "Document processed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Processing failed"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, or TXT)"),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """
    Upload and process a document for Q&A.

    This endpoint:
    - Validates file type (PDF, DOCX, TXT) and size
    - Extracts text and splits into chunks
    - Generates embeddings and stores in vector database
    - Replaces any previously loaded document

    Args:
        file: Uploaded document file
        document_service: Injected document service

    Returns:
        DocumentUploadResponse with processing statistics

    Raises:
        HTTPException:
            - 400: Unsupported file type or page limit exceeded
            - 413: File too large
            - 422: Processing failed
    """
    try:
        return await document_service.process_upload(file)
    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e)
        ) from e
    except PageLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except ProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from e