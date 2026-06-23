"""DeepSeek model provider implementation."""

import logging
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .openai_compatible import OpenAICompatibleProvider
from .registries.deepseek import DeepSeekModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class DeepSeekModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for DeepSeek models exposed over an OpenAI-style API.

    Publishes capability metadata for the officially supported deployments and
    maps tool-category preferences to the appropriate DeepSeek model. Thinking
    is controlled by the model defaults (MVP scope): the V4 models think by
    default, and ``deepseek-chat`` is the non-thinking option.
    """

    FRIENDLY_NAME = "DeepSeek"

    REGISTRY_CLASS = DeepSeekModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Canonical model identifiers used for category routing.
    PRIMARY_MODEL = "deepseek-v4-pro"
    FALLBACK_MODEL = "deepseek-v4-flash"

    def __init__(self, api_key: str, **kwargs):
        """Initialize DeepSeek provider with API key."""
        # Set DeepSeek base URL
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.DEEPSEEK

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get DeepSeek's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer DeepSeek V4 Pro for advanced reasoning tasks.
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer DeepSeek V4 Flash for speed.
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            return allowed_models[0]

        else:  # BALANCED or default
            # Prefer DeepSeek V4 Pro for balanced use.
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]


# Load registry data at import time
DeepSeekModelProvider._ensure_registry()
