"""Custom exceptions for the application."""

__all__ = [
    "ConfigurationError",
    "CollectionNotFoundError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "PageLimitExceededError",
    "ProcessingError",
    "LLMError",
    "NoActiveSessionError",
    "QueryError",
]


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


class CollectionNotFoundError(Exception):
    """Raised when a Chroma collection is not found."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' not found")


class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, extension: str, allowed: list[str]):
        self.extension = extension
        self.allowed = allowed
        super().__init__(
            f"File type '.{extension}' is not supported. Allowed types: {', '.join(allowed)}"
        )


class FileTooLargeError(Exception):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, size_mb: float, max_size_mb: int):
        self.size_mb = size_mb
        self.max_size_mb = max_size_mb
        super().__init__(
            f"File size {size_mb:.1f}MB exceeds limit of {max_size_mb}MB"
        )


class PageLimitExceededError(Exception):
    """Raised when document has too many pages."""

    def __init__(self, page_count: int, max_pages: int):
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(
            f"Document has {page_count} pages, exceeds limit of {max_pages} pages"
        )


class ProcessingError(Exception):
    """General document processing failure."""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class LLMError(Exception):
    """Raised when LLM generation fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class NoActiveSessionError(Exception):
    """Raised when query attempted without loaded document."""

    def __init__(self):
        super().__init__("No document loaded. Please upload a document first.")


class QueryError(Exception):
    """Raised when query processing fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)
