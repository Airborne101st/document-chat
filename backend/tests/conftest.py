"""Pytest configuration and fixtures."""

import os

import pytest

# Set environment variables BEFORE any imports
# This must happen at module level, not in a fixture
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-testing")


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables before any tests run."""
    # Environment variables are already set at module level
    yield
