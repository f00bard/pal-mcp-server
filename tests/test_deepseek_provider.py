"""Tests for DeepSeek provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.deepseek import DeepSeekModelProvider
from providers.shared import ProviderType


class TestDeepSeekProvider:
    """Test DeepSeek provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        # Clear restriction service cache before each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        # Clear restriction service cache after each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = DeepSeekModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.DEEPSEEK
        assert provider.base_url == "https://api.deepseek.com"

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        provider = DeepSeekModelProvider("test-key", base_url="https://custom.deepseek.com")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.deepseek.com"

    def test_model_validation(self):
        """Test model name validation."""
        provider = DeepSeekModelProvider("test-key")

        # Test valid models
        assert provider.validate_model_name("deepseek-v4-pro") is True
        assert provider.validate_model_name("deepseek") is True
        assert provider.validate_model_name("deepseek-pro") is True
        assert provider.validate_model_name("deepseek-v4-flash") is True
        assert provider.validate_model_name("deepseek-flash") is True
        assert provider.validate_model_name("deepseek-chat") is True
        assert provider.validate_model_name("deepseek-reasoner") is True

        # Test invalid model
        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False
        assert provider.validate_model_name("grok-4") is False
        assert provider.validate_model_name("deepseek-v3") is False

    def test_resolve_model_name(self):
        """Test model name resolution."""
        provider = DeepSeekModelProvider("test-key")

        # Test shorthand resolution
        assert provider._resolve_model_name("deepseek") == "deepseek-v4-pro"
        assert provider._resolve_model_name("deepseek-pro") == "deepseek-v4-pro"
        assert provider._resolve_model_name("deepseek-flash") == "deepseek-v4-flash"

        # Test full name passthrough
        assert provider._resolve_model_name("deepseek-v4-pro") == "deepseek-v4-pro"
        assert provider._resolve_model_name("deepseek-v4-flash") == "deepseek-v4-flash"
        assert provider._resolve_model_name("deepseek-chat") == "deepseek-chat"
        assert provider._resolve_model_name("deepseek-reasoner") == "deepseek-reasoner"

    def test_get_capabilities_v4_pro(self):
        """Test getting model capabilities for DeepSeek V4 Pro."""
        provider = DeepSeekModelProvider("test-key")

        capabilities = provider.get_capabilities("deepseek-v4-pro")
        assert capabilities.model_name == "deepseek-v4-pro"
        assert capabilities.friendly_name == "DeepSeek (V4 Pro)"
        assert capabilities.context_window == 1_000_000
        assert capabilities.max_output_tokens == 384_000
        assert capabilities.provider == ProviderType.DEEPSEEK
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is False

        # Test temperature range
        assert capabilities.temperature_constraint.min_temp == 0.0
        assert capabilities.temperature_constraint.max_temp == 2.0
        assert capabilities.temperature_constraint.default_temp == 0.3

    def test_get_capabilities_v4_flash(self):
        """Test getting model capabilities for DeepSeek V4 Flash."""
        provider = DeepSeekModelProvider("test-key")

        capabilities = provider.get_capabilities("deepseek-flash")
        assert capabilities.model_name == "deepseek-v4-flash"
        assert capabilities.friendly_name == "DeepSeek (V4 Flash)"
        assert capabilities.context_window == 1_000_000
        assert capabilities.provider == ProviderType.DEEPSEEK
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is False

    def test_get_capabilities_legacy_chat(self):
        """Legacy deepseek-chat is non-thinking."""
        provider = DeepSeekModelProvider("test-key")

        capabilities = provider.get_capabilities("deepseek-chat")
        assert capabilities.model_name == "deepseek-chat"
        assert capabilities.supports_extended_thinking is False

    def test_get_capabilities_with_shorthand(self):
        """Test getting model capabilities with shorthand."""
        provider = DeepSeekModelProvider("test-key")

        capabilities = provider.get_capabilities("deepseek")
        assert capabilities.model_name == "deepseek-v4-pro"  # Should resolve to full name
        assert capabilities.context_window == 1_000_000

        capabilities_flash = provider.get_capabilities("deepseek-flash")
        assert capabilities_flash.model_name == "deepseek-v4-flash"  # Should resolve to full name

    def test_unsupported_model_capabilities(self):
        """Test error handling for unsupported models."""
        provider = DeepSeekModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider deepseek"):
            provider.get_capabilities("invalid-model")

    def test_extended_thinking_flags(self):
        """DeepSeek V4 capabilities should expose extended thinking support correctly."""
        provider = DeepSeekModelProvider("test-key")

        thinking_aliases = [
            "deepseek-v4-pro",
            "deepseek",
            "deepseek-pro",
            "deepseek-v4-flash",
            "deepseek-flash",
            "deepseek-reasoner",
        ]
        for alias in thinking_aliases:
            assert provider.get_capabilities(alias).supports_extended_thinking is True

    def test_provider_type(self):
        """Test provider type identification."""
        provider = DeepSeekModelProvider("test-key")
        assert provider.get_provider_type() == ProviderType.DEEPSEEK

    @patch.dict(os.environ, {"DEEPSEEK_ALLOWED_MODELS": "deepseek-v4-pro"})
    def test_model_restrictions(self):
        """Test model restrictions functionality."""
        # Clear cached restriction service
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = DeepSeekModelProvider("test-key")

        # deepseek-v4-pro should be allowed (including aliases)
        assert provider.validate_model_name("deepseek-v4-pro") is True
        assert provider.validate_model_name("deepseek") is True
        assert provider.validate_model_name("deepseek-pro") is True

        # deepseek-v4-flash should be blocked by restrictions
        assert provider.validate_model_name("deepseek-v4-flash") is False
        assert provider.validate_model_name("deepseek-flash") is False

    @patch.dict(os.environ, {"DEEPSEEK_ALLOWED_MODELS": "deepseek-v4-flash"})
    def test_multiple_model_restrictions(self):
        """Restrictions should allow aliases for DeepSeek V4 Flash."""
        # Clear cached restriction service
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = DeepSeekModelProvider("test-key")

        # Canonical name and alias should be allowed
        assert provider.validate_model_name("deepseek-v4-flash") is True
        assert provider.validate_model_name("deepseek-flash") is True

        # deepseek-v4-pro should NOT be allowed
        assert provider.validate_model_name("deepseek-v4-pro") is False
        assert provider.validate_model_name("deepseek") is False

    @patch.dict(os.environ, {"DEEPSEEK_ALLOWED_MODELS": "deepseek,deepseek-v4-pro,deepseek-flash,deepseek-v4-flash"})
    def test_both_shorthand_and_full_name_allowed(self):
        """Test that aliases and canonical names can be allowed together."""
        # Clear cached restriction service
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = DeepSeekModelProvider("test-key")

        # Both shorthand and full name should be allowed when explicitly listed
        assert provider.validate_model_name("deepseek") is True  # Alias explicitly allowed
        assert provider.validate_model_name("deepseek-v4-pro") is True  # Canonical name explicitly allowed
        assert provider.validate_model_name("deepseek-flash") is True  # Alias explicitly allowed
        assert provider.validate_model_name("deepseek-v4-flash") is True  # Canonical name explicitly allowed

    @patch.dict(os.environ, {"DEEPSEEK_ALLOWED_MODELS": ""})
    def test_empty_restrictions_allows_all(self):
        """Test that empty restrictions allow all models."""
        # Clear cached restriction service
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = DeepSeekModelProvider("test-key")

        assert provider.validate_model_name("deepseek-v4-pro") is True
        assert provider.validate_model_name("deepseek-v4-flash") is True
        assert provider.validate_model_name("deepseek") is True
        assert provider.validate_model_name("deepseek-pro") is True
        assert provider.validate_model_name("deepseek-flash") is True

    def test_friendly_name(self):
        """Test friendly name constant."""
        provider = DeepSeekModelProvider("test-key")
        assert provider.FRIENDLY_NAME == "DeepSeek"

        capabilities = provider.get_capabilities("deepseek-v4-pro")
        assert capabilities.friendly_name == "DeepSeek (V4 Pro)"

    def test_supported_models_structure(self):
        """Test that MODEL_CAPABILITIES has the correct structure."""
        provider = DeepSeekModelProvider("test-key")

        # Check that all expected base models are present
        assert "deepseek-v4-pro" in provider.MODEL_CAPABILITIES
        assert "deepseek-v4-flash" in provider.MODEL_CAPABILITIES
        assert "deepseek-chat" in provider.MODEL_CAPABILITIES
        assert "deepseek-reasoner" in provider.MODEL_CAPABILITIES

        # Check model configs have required fields
        from providers.shared import ModelCapabilities

        pro_config = provider.MODEL_CAPABILITIES["deepseek-v4-pro"]
        assert isinstance(pro_config, ModelCapabilities)
        assert hasattr(pro_config, "context_window")
        assert hasattr(pro_config, "supports_extended_thinking")
        assert hasattr(pro_config, "aliases")
        assert pro_config.context_window == 1_000_000
        assert pro_config.supports_extended_thinking is True

        # Check aliases are correctly structured
        assert "deepseek" in pro_config.aliases
        assert "deepseek-pro" in pro_config.aliases

        flash_config = provider.MODEL_CAPABILITIES["deepseek-v4-flash"]
        assert flash_config.context_window == 1_000_000
        assert flash_config.supports_extended_thinking is True
        assert "deepseek-flash" in flash_config.aliases

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_resolves_alias_before_api_call(self, mock_openai_class):
        """Test that generate_content resolves aliases before making API calls.

        This is the CRITICAL test that ensures aliases like 'deepseek' get resolved
        to 'deepseek-v4-pro' before being sent to the DeepSeek API.
        """
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "deepseek-v4-pro"  # API returns the resolved model name
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        provider = DeepSeekModelProvider("test-key")

        # Call generate_content with alias 'deepseek'
        result = provider.generate_content(
            prompt="Test prompt", model_name="deepseek", temperature=0.7  # This should resolve to "deepseek-v4-pro"
        )

        # Verify the API was called with the RESOLVED model name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # CRITICAL ASSERTION: The API should receive "deepseek-v4-pro", not "deepseek"
        assert (
            call_kwargs["model"] == "deepseek-v4-pro"
        ), f"Expected 'deepseek-v4-pro' but API received '{call_kwargs['model']}'"

        # Verify other parameters
        assert call_kwargs["temperature"] == 0.7
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test prompt"

        # Verify response
        assert result.content == "Test response"
        assert result.model_name == "deepseek-v4-pro"  # Should be the resolved name

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_other_aliases(self, mock_openai_class):
        """Test other alias resolutions in generate_content."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = DeepSeekModelProvider("test-key")

        # Test deepseek-pro -> deepseek-v4-pro
        mock_response.model = "deepseek-v4-pro"
        provider.generate_content(prompt="Test", model_name="deepseek-pro", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-v4-pro"

        # Test deepseek-flash -> deepseek-v4-flash
        mock_response.model = "deepseek-v4-flash"
        provider.generate_content(prompt="Test", model_name="deepseek-flash", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-v4-flash"

        # Test deepseek-chat -> deepseek-chat (legacy passthrough)
        mock_response.model = "deepseek-chat"
        provider.generate_content(prompt="Test", model_name="deepseek-chat", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"
