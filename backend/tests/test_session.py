"""Unit tests for session manager."""

from datetime import datetime

import pytest

from app.core.session import SessionManager, SessionState, get_session_manager


@pytest.fixture(autouse=True)
def reset_session():
    """Reset session manager before each test."""
    manager = SessionManager()
    manager.reset_for_testing()
    yield
    # Clean up after test
    manager.reset_for_testing()


class TestSessionState:
    """Tests for SessionState Pydantic model."""

    def test_default_session_state(self):
        """Test default session state values."""
        state = SessionState()

        assert state.is_active is False
        assert state.filename is None
        assert state.page_count == 0
        assert state.chunk_count == 0
        assert state.created_at is None
        assert state.collection_name is None

    def test_session_state_with_values(self):
        """Test creating session state with values."""
        now = datetime.utcnow()
        state = SessionState(
            is_active=True,
            filename="test.pdf",
            page_count=10,
            chunk_count=25,
            created_at=now,
            collection_name="session_abc123",
        )

        assert state.is_active is True
        assert state.filename == "test.pdf"
        assert state.page_count == 10
        assert state.chunk_count == 25
        assert state.created_at == now
        assert state.collection_name == "session_abc123"

    def test_session_state_model_copy(self):
        """Test copying session state."""
        state = SessionState(is_active=True, filename="test.pdf")
        copied = state.model_copy()

        assert copied.is_active == state.is_active
        assert copied.filename == state.filename
        assert copied is not state  # Different instances


class TestSessionManager:
    """Tests for SessionManager singleton class."""

    def test_singleton_behavior(self):
        """Test that SessionManager returns the same instance."""
        manager1 = SessionManager()
        manager2 = SessionManager()

        assert manager1 is manager2

    def test_initial_state(self):
        """Test initial session state is inactive."""
        manager = SessionManager()

        assert manager.is_active is False
        assert manager.collection_name is None

        state = manager.get_state()
        assert state.is_active is False
        assert state.filename is None

    def test_create_session(self):
        """Test creating a new session."""
        manager = SessionManager()

        collection_name = manager.create_session(
            filename="report.pdf", page_count=12, chunk_count=34
        )

        # Check collection name is generated
        assert collection_name is not None
        assert collection_name.startswith("session_")
        assert len(collection_name) == 20  # "session_" + 12 hex chars

        # Check state is updated
        assert manager.is_active is True
        assert manager.collection_name == collection_name

        state = manager.get_state()
        assert state.is_active is True
        assert state.filename == "report.pdf"
        assert state.page_count == 12
        assert state.chunk_count == 34
        assert state.created_at is not None
        assert isinstance(state.created_at, datetime)
        assert state.collection_name == collection_name

    def test_create_session_returns_unique_names(self):
        """Test that each session gets a unique collection name."""
        manager = SessionManager()

        collection1 = manager.create_session("doc1.pdf", 5, 10)
        collection2 = manager.create_session("doc2.pdf", 8, 15)

        assert collection1 != collection2
        assert collection1.startswith("session_")
        assert collection2.startswith("session_")

    def test_get_state_returns_copy(self):
        """Test that get_state returns a copy, not the original."""
        manager = SessionManager()
        manager.create_session("test.pdf", 5, 10)

        state1 = manager.get_state()
        state2 = manager.get_state()

        assert state1 is not state2  # Different instances
        assert state1.filename == state2.filename  # Same values
        assert state1.collection_name == state2.collection_name

    def test_clear_session(self):
        """Test clearing the session."""
        manager = SessionManager()

        # Create a session first
        collection_name = manager.create_session("test.pdf", 5, 10)
        assert manager.is_active is True

        # Clear the session
        old_collection = manager.clear_session()

        # Should return the old collection name
        assert old_collection == collection_name

        # State should be reset
        assert manager.is_active is False
        assert manager.collection_name is None

        state = manager.get_state()
        assert state.is_active is False
        assert state.filename is None
        assert state.page_count == 0
        assert state.chunk_count == 0
        assert state.created_at is None
        assert state.collection_name is None

    def test_clear_session_when_no_session(self):
        """Test clearing when no session exists."""
        manager = SessionManager()

        # Clear without creating a session
        old_collection = manager.clear_session()

        # Should return None
        assert old_collection is None
        assert manager.is_active is False

    def test_new_session_replaces_old_session(self):
        """Test that creating a new session replaces the old one."""
        manager = SessionManager()

        # Create first session
        collection1 = manager.create_session("doc1.pdf", 5, 10)
        state1 = manager.get_state()

        # Create second session (should replace first)
        collection2 = manager.create_session("doc2.pdf", 8, 15)
        state2 = manager.get_state()

        # Collections should be different
        assert collection1 != collection2

        # State should reflect the new session
        assert state2.filename == "doc2.pdf"
        assert state2.page_count == 8
        assert state2.chunk_count == 15
        assert state2.collection_name == collection2

        # Old session data should be gone
        assert manager.collection_name == collection2
        assert manager.get_state().filename == "doc2.pdf"

    def test_is_active_property(self):
        """Test the is_active property."""
        manager = SessionManager()

        assert manager.is_active is False

        manager.create_session("test.pdf", 5, 10)
        assert manager.is_active is True

        manager.clear_session()
        assert manager.is_active is False

    def test_collection_name_property(self):
        """Test the collection_name property."""
        manager = SessionManager()

        assert manager.collection_name is None

        collection = manager.create_session("test.pdf", 5, 10)
        assert manager.collection_name == collection
        assert manager.collection_name.startswith("session_")

        manager.clear_session()
        assert manager.collection_name is None

    def test_reset_for_testing(self):
        """Test the reset_for_testing method."""
        manager = SessionManager()

        # Create a session
        manager.create_session("test.pdf", 5, 10)
        assert manager.is_active is True

        # Reset
        manager.reset_for_testing()

        # Should be back to default state
        assert manager.is_active is False
        assert manager.collection_name is None

        state = manager.get_state()
        assert state.is_active is False
        assert state.filename is None


class TestGetSessionManager:
    """Tests for the FastAPI dependency function."""

    def test_get_session_manager_returns_singleton(self):
        """Test that get_session_manager returns the singleton instance."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2
        assert isinstance(manager1, SessionManager)

    def test_get_session_manager_state_persists(self):
        """Test that state persists across get_session_manager calls."""
        manager1 = get_session_manager()
        collection = manager1.create_session("test.pdf", 5, 10)

        manager2 = get_session_manager()

        assert manager2.is_active is True
        assert manager2.collection_name == collection
        assert manager2.get_state().filename == "test.pdf"


class TestSessionManagerIntegration:
    """Integration tests for typical usage patterns."""

    def test_typical_document_upload_workflow(self):
        """Test a typical workflow of uploading a document."""
        manager = SessionManager()

        # Initially no session
        assert not manager.is_active

        # Upload document creates session
        collection = manager.create_session("report.pdf", 12, 34)

        assert manager.is_active
        assert manager.collection_name is not None

        state = manager.get_state()
        assert state.filename == "report.pdf"
        assert state.page_count == 12
        assert state.chunk_count == 34

        # Clear session when done
        old_collection = manager.clear_session()
        assert old_collection == collection
        assert not manager.is_active

    def test_replacing_document_workflow(self):
        """Test uploading a new document while one is loaded."""
        manager = SessionManager()

        # Upload first document
        collection1 = manager.create_session("doc1.pdf", 5, 10)
        assert manager.get_state().filename == "doc1.pdf"

        # Upload second document (replaces first)
        collection2 = manager.create_session("doc2.pdf", 8, 15)

        # Should have new session
        assert collection2 != collection1
        assert manager.get_state().filename == "doc2.pdf"
        assert manager.get_state().chunk_count == 15

        # Old collection name is lost (would need to be cleaned up before replace)
        # This is expected behavior for v1