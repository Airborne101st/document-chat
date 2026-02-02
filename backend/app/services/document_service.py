"""Document service that orchestrates the full upload pipeline."""

import logging
import tempfile
import time
from pathlib import Path

from fastapi import Depends, UploadFile
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.core.document_processor import DocumentProcessor, DocumentProcessingError
from app.core.exceptions import (
    FileTooLargeError,
    PageLimitExceededError,
    ProcessingError,
    UnsupportedFileTypeError,
)
from app.core.session import SessionManager, get_session_manager
from app.models.schemas import DocumentUploadResponse
from app.services.vector_store_service import (
    VectorStoreService,
    get_vector_store_service,
)

logger = logging.getLogger(__name__)


class ProcessingResult(BaseModel):
    """Internal result from document processing."""

    filename: str
    page_count: int
    chunk_count: int
    collection_name: str
    processing_time_ms: float


class DocumentService:
    """
    Orchestrates document upload pipeline:
    validate → save temp → process → embed → store → cleanup
    """

    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_store_service: VectorStoreService,
        session_manager: SessionManager,
        settings: Settings,
    ):
        """
        Initialize with required dependencies.

        Args:
            document_processor: For loading and chunking documents
            vector_store_service: For creating and managing vector collections
            session_manager: For tracking session state
            settings: Application settings
        """
        self._document_processor = document_processor
        self._vector_store = vector_store_service
        self._session_manager = session_manager
        self._settings = settings

    async def process_upload(self, file: UploadFile) -> DocumentUploadResponse:
        """
        Full document processing pipeline.

        Steps:
        1. Validate file (extension, size)
        2. Save to temp location
        3. Clear previous session if exists (delete old collection)
        4. Load and chunk document using DocumentProcessor
        5. Create new Chroma collection with chunks
        6. Create new session
        7. Cleanup temp file
        8. Return response with stats

        Args:
            file: Uploaded file from FastAPI

        Returns:
            DocumentUploadResponse with processing stats

        Raises:
            UnsupportedFileTypeError: Invalid extension
            FileTooLargeError: Exceeds size limit
            PageLimitExceededError: Too many pages
            ProcessingError: General processing failure
        """
        start_time = time.time()
        temp_file_path = None

        try:
            # Step 1: Validate file
            self._validate_file(file)

            # Step 2: Save to temp location
            temp_file_path = await self._save_temp_file(file)

            # Step 3: Clear previous session if exists
            self._cleanup_old_session()

            # Step 4: Load and chunk document
            try:
                processed_doc = self._document_processor.process_document(
                    temp_file_path
                )
            except ValueError as e:
                # DocumentProcessor raises ValueError for unsupported extensions
                raise UnsupportedFileTypeError(
                    extension=temp_file_path.suffix.lstrip(".").lower(),
                    allowed=self._settings.allowed_extensions,
                ) from e
            except DocumentProcessingError as e:
                raise ProcessingError(
                    f"Failed to process document: {str(e)}", original_error=e
                ) from e

            # Check page limit (for PDFs, each page is a document)
            # For other formats, we'll count chunks as a proxy
            page_count = self._estimate_page_count(processed_doc.chunks)
            if page_count > self._settings.max_page_limit:
                raise PageLimitExceededError(
                    page_count=page_count, max_pages=self._settings.max_page_limit
                )

            # Step 5: Create new session first to get collection name
            # Use original filename from upload, not temp file name
            original_filename = file.filename or processed_doc.filename
            collection_name = self._session_manager.create_session(
                filename=original_filename,
                page_count=page_count,
                chunk_count=processed_doc.total_chunks,
            )
            logger.info(f"Created session with collection: {collection_name}")

            # Step 6: Create Chroma collection with the session's collection name
            try:
                logger.info(f"Creating vector store collection: {collection_name}")
                self._vector_store.create_collection(
                    collection_name=collection_name, documents=processed_doc.chunks
                )
                logger.info(f"✓ Vector store collection created: {collection_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create vector store collection: {e}", exc_info=True)
                # Clean up session since collection creation failed
                self._session_manager.clear_session()
                raise ProcessingError(
                    "Failed to create vector store collection", original_error=e
                ) from e

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Document processed successfully: {original_filename}, "
                f"{processed_doc.total_chunks} chunks, {processing_time_ms:.0f}ms"
            )

            # Step 8: Return response
            return DocumentUploadResponse(
                filename=original_filename,
                file_type=processed_doc.file_type,
                total_chunks=processed_doc.total_chunks,
                message="Document uploaded and processed successfully",
            )

        except (
            UnsupportedFileTypeError,
            FileTooLargeError,
            PageLimitExceededError,
            ProcessingError,
        ):
            # Re-raise expected errors
            raise

        except Exception as e:
            # Wrap unexpected errors
            logger.error(f"Unexpected error during document upload: {e}", exc_info=True)
            raise ProcessingError(
                "An unexpected error occurred during document processing",
                original_error=e,
            ) from e

        finally:
            # Step 7: Always cleanup temp file
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)

    def _validate_file(self, file: UploadFile) -> None:
        """
        Validate file extension and size.

        Args:
            file: Uploaded file to validate

        Raises:
            UnsupportedFileTypeError: If extension not in allowed list
            FileTooLargeError: If file exceeds max_file_size_mb
        """
        # Validate extension
        if file.filename:
            extension = Path(file.filename).suffix.lstrip(".").lower()
            if extension not in self._settings.allowed_extensions:
                raise UnsupportedFileTypeError(
                    extension=extension, allowed=self._settings.allowed_extensions
                )

        # Validate size
        if file.size:
            size_mb = file.size / (1024 * 1024)
            if size_mb > self._settings.max_file_size_mb:
                raise FileTooLargeError(
                    size_mb=size_mb, max_size_mb=self._settings.max_file_size_mb
                )

    async def _save_temp_file(self, file: UploadFile) -> Path:
        """
        Save uploaded file to temp directory.

        Args:
            file: Uploaded file to save

        Returns:
            Path to saved temp file
        """
        # Get file extension for temp file
        suffix = Path(file.filename).suffix if file.filename else ".tmp"

        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Read and write file content
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        logger.debug(f"Saved temp file: {tmp_path}")
        return tmp_path

    def _cleanup_temp_file(self, file_path: Path) -> None:
        """
        Remove temp file. Logs warning if file doesn't exist.

        Args:
            file_path: Path to temp file to delete
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
            else:
                logger.warning(f"Temp file not found for cleanup: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

    def _cleanup_old_session(self) -> None:
        """
        If session exists, clear it and delete old collection.
        Called before processing new document.
        """
        if self._session_manager.is_active:
            old_collection = self._session_manager.collection_name
            if old_collection:
                try:
                    self._vector_store.delete_collection(old_collection)
                    logger.info(f"Deleted old collection: {old_collection}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete old collection {old_collection}: {e}"
                    )

            # Clear session state
            self._session_manager.clear_session()
            logger.info("Cleared previous session")

    def _estimate_page_count(self, chunks: list) -> int:
        """
        Estimate page count from chunks.

        For PDFs, uses the 'page' metadata field.
        For other formats, estimates based on chunk count.

        Args:
            chunks: List of document chunks

        Returns:
            Estimated page count
        """
        if not chunks:
            return 1

        # Try to get max page number from metadata
        max_page = 0
        for chunk in chunks:
            if hasattr(chunk, "metadata") and "page" in chunk.metadata:
                max_page = max(max_page, chunk.metadata["page"])

        # If we found page metadata, use it
        if max_page > 0:
            return max_page

        # Otherwise, estimate: assume ~3 chunks per page (rough estimate)
        estimated_pages = max(1, len(chunks) // 3)
        return estimated_pages


def get_document_service(
    session_manager: SessionManager = Depends(get_session_manager),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """
    FastAPI dependency function to get a document service instance.

    Args:
        session_manager: Injected session manager
        vector_store_service: Injected vector store service
        settings: Injected application settings

    Returns:
        DocumentService instance
    """
    document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    return DocumentService(
        document_processor=document_processor,
        vector_store_service=vector_store_service,
        session_manager=session_manager,
        settings=settings,
    )