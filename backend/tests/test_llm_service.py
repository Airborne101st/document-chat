"""Tests for the LLM service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import ConfigurationError, LLMError
from app.services.llm_service import (
    LLMConfig,
    LLMMessage,
    LLMService,
    get_llm_service,
)


class TestLLMMessage:
    """Test suite for LLMMessage model."""

    def test_valid_user_message(self):
        """Test creating a valid user message."""
        msg = LLMMessage(role="user", content="What is AI?")

        assert msg.role == "user"
        assert msg.content == "What is AI?"

    def test_valid_assistant_message(self):
        """Test creating a valid assistant message."""
        msg = LLMMessage(role="assistant", content="AI stands for...")

        assert msg.role == "assistant"
        assert msg.content == "AI stands for..."

    def test_valid_system_message(self):
        """Test creating a valid system message."""
        msg = LLMMessage(role="system", content="You are a helpful assistant.")

        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_invalid_role(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValueError):
            LLMMessage(role="invalid", content="Test")


class TestLLMConfig:
    """Test suite for LLMConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.model == "gemini-1.5-flash"
        assert config.temperature == 0.3
        assert config.max_output_tokens == 2048

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = LLMConfig(
            model="gemini-1.5-pro", temperature=0.7, max_output_tokens=4096
        )

        assert config.model == "gemini-1.5-pro"
        assert config.temperature == 0.7
        assert config.max_output_tokens == 4096

    def test_temperature_bounds(self):
        """Test that temperature is bounded between 0 and 1."""
        # Lower bound
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        # Upper bound
        with pytest.raises(ValueError):
            LLMConfig(temperature=1.5)

        # Valid bounds
        config_min = LLMConfig(temperature=0.0)
        config_max = LLMConfig(temperature=1.0)
        assert config_min.temperature == 0.0
        assert config_max.temperature == 1.0

    def test_max_output_tokens_bounds(self):
        """Test that max_output_tokens has valid bounds."""
        # Lower bound
        with pytest.raises(ValueError):
            LLMConfig(max_output_tokens=0)

        # Upper bound
        with pytest.raises(ValueError):
            LLMConfig(max_output_tokens=10000)

        # Valid bounds
        config_min = LLMConfig(max_output_tokens=1)
        config_max = LLMConfig(max_output_tokens=8192)
        assert config_min.max_output_tokens == 1
        assert config_max.max_output_tokens == 8192


class TestLLMService:
    """Test suite for LLMService."""

    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    @patch("app.config.settings")
    def test_init_with_settings_api_key(self, mock_settings, mock_chat_class):
        """Test initialization with API key from settings."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance

        service = LLMService()

        mock_chat_class.assert_called_once_with(
            model="gemini-1.5-flash",
            temperature=0.3,
            max_tokens=2048,
            api_key="test-api-key",
        )
        assert service._api_key == "test-api-key"
        assert service._llm == mock_llm_instance

    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    def test_init_with_override_api_key(self, mock_chat_class):
        """Test initialization with API key override."""
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance

        service = LLMService(api_key="override-key")

        mock_chat_class.assert_called_once_with(
            model="gemini-1.5-flash",
            temperature=0.3,
            max_tokens=2048,
            api_key="override-key",
        )
        assert service._api_key == "override-key"

    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    def test_init_with_custom_config(self, mock_chat_class):
        """Test initialization with custom config."""
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance

        config = LLMConfig(
            model="gemini-1.5-pro", temperature=0.7, max_output_tokens=4096
        )
        service = LLMService(api_key="test-key", config=config)

        mock_chat_class.assert_called_once_with(
            model="gemini-1.5-pro",
            temperature=0.7,
            max_tokens=4096,
            api_key="test-key",
        )

    @patch("app.config.settings")
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    def test_init_missing_api_key_raises_error(self, mock_chat_class, mock_settings):
        """Test that missing API key raises ConfigurationError."""
        mock_settings.gemini_api_key = ""

        with pytest.raises(ConfigurationError) as exc_info:
            LLMService()

        assert "Gemini API key is required" in str(exc_info.value)
        mock_chat_class.assert_not_called()

    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    def test_init_chat_model_failure(self, mock_chat_class):
        """Test that ChatGoogleGenerativeAI initialization failure is handled."""
        mock_chat_class.side_effect = Exception("API connection failed")

        with pytest.raises(ConfigurationError) as exc_info:
            LLMService(api_key="test-key")

        assert "Failed to initialize ChatGoogleGenerativeAI" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_stream_success(self, mock_chat_class):
        """Test streaming generation success."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock streaming chunks
        async def mock_astream(messages):
            chunks = [
                MagicMock(content="Hello"),
                MagicMock(content=" "),
                MagicMock(content="world"),
                MagicMock(content="!"),
            ]
            for chunk in chunks:
                yield chunk

        mock_llm.astream = mock_astream

        # Create service and stream
        service = LLMService(api_key="test-key")
        result = []
        async for token in service.generate_stream("Test prompt"):
            result.append(token)

        # Verify
        assert result == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_stream_with_system_prompt(self, mock_chat_class):
        """Test streaming with system prompt."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock streaming chunks
        async def mock_astream(messages):
            # Verify messages include system prompt
            assert len(messages) == 2
            assert messages[0].content == "You are helpful"
            assert messages[1].content == "Test prompt"

            chunks = [MagicMock(content="Response")]
            for chunk in chunks:
                yield chunk

        mock_llm.astream = mock_astream

        # Create service and stream
        service = LLMService(api_key="test-key")
        result = []
        async for token in service.generate_stream(
            "Test prompt", system_prompt="You are helpful"
        ):
            result.append(token)

        assert result == ["Response"]

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_stream_empty_chunks(self, mock_chat_class):
        """Test streaming handles empty chunks."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock streaming chunks with some empty content
        async def mock_astream(messages):
            chunks = [
                MagicMock(content="Hello"),
                MagicMock(content=""),  # Empty chunk
                MagicMock(content="world"),
                MagicMock(content=None),  # None content
            ]
            for chunk in chunks:
                yield chunk

        mock_llm.astream = mock_astream

        # Create service and stream
        service = LLMService(api_key="test-key")
        result = []
        async for token in service.generate_stream("Test prompt"):
            result.append(token)

        # Empty and None chunks should be filtered out
        assert result == ["Hello", "world"]

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_stream_failure(self, mock_chat_class):
        """Test streaming generation failure handling."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock streaming failure
        async def mock_astream(messages):
            raise Exception("API error")
            yield  # Make it a generator

        mock_llm.astream = mock_astream

        # Create service and attempt to stream
        service = LLMService(api_key="test-key")

        with pytest.raises(LLMError) as exc_info:
            async for token in service.generate_stream("Test prompt"):
                pass

        assert "Failed to generate streaming response" in str(exc_info.value)
        assert exc_info.value.original_error is not None

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_success(self, mock_chat_class):
        """Test non-streaming generation success."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock ainvoke response
        mock_response = MagicMock()
        mock_response.content = "This is a complete response."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Create service and generate
        service = LLMService(api_key="test-key")
        result = await service.generate("Test prompt")

        # Verify
        assert result == "This is a complete response."
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_with_system_prompt(self, mock_chat_class):
        """Test non-streaming with system prompt."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock ainvoke
        async def mock_ainvoke(messages):
            # Verify messages include system prompt
            assert len(messages) == 2
            assert messages[0].content == "Be concise"
            assert messages[1].content == "What is AI?"

            mock_response = MagicMock()
            mock_response.content = "AI is artificial intelligence."
            return mock_response

        mock_llm.ainvoke = mock_ainvoke

        # Create service and generate
        service = LLMService(api_key="test-key")
        result = await service.generate("What is AI?", system_prompt="Be concise")

        assert result == "AI is artificial intelligence."

    @pytest.mark.asyncio
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    async def test_generate_failure(self, mock_chat_class):
        """Test non-streaming generation failure handling."""
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_chat_class.return_value = mock_llm

        # Mock ainvoke failure
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        # Create service and attempt to generate
        service = LLMService(api_key="test-key")

        with pytest.raises(LLMError) as exc_info:
            await service.generate("Test prompt")

        assert "Failed to generate response from LLM" in str(exc_info.value)
        assert exc_info.value.original_error is not None


class TestDependencyFunction:
    """Test suite for FastAPI dependency function."""

    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    @patch("app.config.settings")
    def test_get_llm_service(self, mock_settings, mock_chat_class):
        """Test the FastAPI dependency function."""
        mock_settings.gemini_api_key = "test-api-key"
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance

        service = get_llm_service()

        assert isinstance(service, LLMService)
        mock_chat_class.assert_called_once()

    @patch("app.config.settings")
    @patch("app.services.llm_service.ChatGoogleGenerativeAI")
    def test_get_llm_service_missing_key(self, mock_chat_class, mock_settings):
        """Test dependency function raises ConfigurationError when API key is missing."""
        mock_settings.gemini_api_key = ""

        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_service()

        assert "Gemini API key is required" in str(exc_info.value)