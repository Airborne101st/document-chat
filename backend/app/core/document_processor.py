"""Document processing using LangChain for loading and chunking documents.

This module wraps LangChain's document loaders and text splitters to provide
a clean interface for the RAG application.
"""

from pathlib import Path
from typing import ClassVar

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.config import settings


class DocumentProcessingError(Exception):
    """Raised when document processing fails."""

    pass


class ProcessedDocument(BaseModel):
    """Represents a processed document with chunks.

    Attributes:
        filename: Original filename.
        file_type: File extension (pdf, docx, txt).
        total_chunks: Number of chunks created.
        chunks: List of text chunks with metadata.
    """

    filename: str
    file_type: str
    total_chunks: int
    chunks: list[Document]

    model_config = {"arbitrary_types_allowed": True}


class DocumentProcessor:
    """Processes documents using LangChain loaders and splitters.

    Handles loading of PDF, DOCX, and TXT files and splits them into chunks
    suitable for embedding and retrieval.
    """

    # Map file extensions to loader classes
    _LOADERS: ClassVar[dict[str, type]] = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
    }

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Size of text chunks in tokens. If None, uses config value.
            chunk_overlap: Overlap between chunks in tokens. If None, uses config value.
        """
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_tokens

        # Initialize text splitter with tiktoken encoding
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",  # tiktoken encoding model
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def process_document(self, file_path: Path | str) -> ProcessedDocument:
        """Process a document file: load and chunk it.

        Args:
            file_path: Path to the document file.

        Returns:
            ProcessedDocument containing chunks and metadata.

        Raises:
            ValueError: If file extension is not supported.
            DocumentProcessingError: If loading or processing fails.
            FileNotFoundError: If file does not exist.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Get file extension
        file_extension = path.suffix.lstrip(".").lower()

        if file_extension not in self._LOADERS:
            supported = ", ".join(self._LOADERS.keys())
            raise ValueError(
                f"Unsupported file extension: '{file_extension}'. "
                f"Supported extensions: {supported}"
            )

        # Load document
        try:
            documents = self._load_document(path, file_extension)
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to load document: {e}"
            ) from e

        # Split into chunks
        try:
            chunks = self.text_splitter.split_documents(documents)
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to split document into chunks: {e}"
            ) from e

        # Enrich metadata
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "filename": path.name,
                    "file_type": file_extension,
                    "source": str(path),
                }
            )

        return ProcessedDocument(
            filename=path.name,
            file_type=file_extension,
            total_chunks=len(chunks),
            chunks=chunks,
        )

    def _load_document(self, path: Path, file_extension: str) -> list[Document]:
        """Load a document using the appropriate LangChain loader.

        Args:
            path: Path to the document.
            file_extension: File extension (pdf, docx, txt).

        Returns:
            List of LangChain Document objects.

        Raises:
            DocumentProcessingError: If loading fails.
        """
        loader_class = self._LOADERS[file_extension]
        loader = loader_class(str(path))

        try:
            documents = loader.load()

            # Handle empty documents
            if not documents:
                # Return an empty document with metadata
                return [
                    Document(
                        page_content="",
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "file_type": file_extension,
                        },
                    )
                ]

            return documents

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to load {file_extension.upper()} file: {e}"
            ) from e

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (pdf, docx, txt).
        """
        return list(cls._LOADERS.keys())

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """Chunk raw text into documents.

        Useful for processing text that's already been extracted.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of Document chunks.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.text_splitter.split_documents([doc])


def process_document(
    file_path: Path | str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> ProcessedDocument:
    """Convenience function to process a document.

    Args:
        file_path: Path to the document file.
        chunk_size: Optional chunk size override.
        chunk_overlap: Optional chunk overlap override.

    Returns:
        ProcessedDocument with chunks and metadata.

    Raises:
        ValueError: If file extension is not supported.
        DocumentProcessingError: If processing fails.
        FileNotFoundError: If file does not exist.

    Example:
        >>> result = process_document("document.pdf")
        >>> print(f"Created {result.total_chunks} chunks")
        >>> for chunk in result.chunks:
        ...     print(chunk.page_content[:100])
    """
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return processor.process_document(file_path)