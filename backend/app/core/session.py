"""Session management for document chat application.

This module provides a singleton SessionManager to track the current document
session state. In v1, only one document can be active at a time.
"""

import uuid
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class SessionState(BaseModel):
    """Current session state for the loaded document.

    Attributes:
        is_active: Whether a document is currently loaded.
        filename: Name of the loaded document file.
        page_count: Number of pages in the document.
        chunk_count: Number of chunks created from the document.
        created_at: Timestamp when the session was created.
        collection_name: ChromaDB collection name for this session.
    """

    is_active: bool = False
    filename: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    created_at: datetime | None = None
    collection_name: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_active": True,
                "filename": "report.pdf",
                "page_count": 12,
                "chunk_count": 34,
                "created_at": "2024-01-15T10:30:00",
                "collection_name": "session_abc123",
            }
        }
    )


class SessionManager:
    """Manages global session state.

    This is a singleton class that maintains state for the currently loaded document.
    For v1: Single document at a time, no persistence.
    New upload replaces previous session.
    """

    _instance: ClassVar["SessionManager | None"] = None
    _state: SessionState

    def __new__(cls) -> "SessionManager":
        """Ensure only one instance exists (singleton pattern).

        Returns:
            The singleton SessionManager instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._state = SessionState()
        return cls._instance

    def create_session(
        self, filename: str, page_count: int, chunk_count: int
    ) -> str:
        """Create a new session for a document.

        If a session already exists, it will be replaced. The old collection name
        is returned for cleanup purposes.

        Args:
            filename: Name of the document file.
            page_count: Number of pages in the document.
            chunk_count: Number of chunks created.

        Returns:
            The generated collection name for ChromaDB.
        """
        # Generate unique collection name
        collection_name = f"session_{uuid.uuid4().hex[:12]}"

        # Update session state
        self._state = SessionState(
            is_active=True,
            filename=filename,
            page_count=page_count,
            chunk_count=chunk_count,
            created_at=datetime.utcnow(),
            collection_name=collection_name,
        )

        return collection_name

    def get_state(self) -> SessionState:
        """Get a copy of the current session state.

        Returns:
            A copy of the SessionState model.
        """
        return self._state.model_copy()

    def clear_session(self) -> str | None:
        """Clear the current session.

        Resets the session state to default values.

        Returns:
            The old collection name for cleanup, or None if no session was active.
        """
        old_collection_name = self._state.collection_name

        # Reset to default state
        self._state = SessionState()

        return old_collection_name

    @property
    def is_active(self) -> bool:
        """Check if a document is currently loaded.

        Returns:
            True if a session is active, False otherwise.
        """
        return self._state.is_active

    @property
    def collection_name(self) -> str | None:
        """Get the current ChromaDB collection name.

        Returns:
            The collection name, or None if no session is active.
        """
        return self._state.collection_name

    def reset_for_testing(self) -> None:
        """Reset the singleton instance to default state.

        WARNING: This should only be used in tests to ensure clean state
        between test runs.
        """
        self._state = SessionState()


def get_session_manager() -> SessionManager:
    """FastAPI dependency to get the SessionManager instance.

    Returns:
        The singleton SessionManager instance.

    Example:
        ```python
        @app.get("/session")
        def get_session(manager: SessionManager = Depends(get_session_manager)):
            return manager.get_state()
        ```
    """
    return SessionManager()