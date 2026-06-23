"""OpenRouter provider implementation."""

import logging
from typing import Optional

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .registries.openrouter import OpenRouterModelRegistry
from .shared import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)


class OpenRouterProvider(OpenAICompatibleProvider):
    """Client for OpenRouter's multi-model aggregation service.

    Role
        Surface OpenRouter’s dynamic catalogue through the same interface as
        native providers so tools can reference OpenRouter models and aliases
        without special cases.

    Characteristics
        * Pulls live model definitions from :class:`OpenRouterModelRegistry`
          (aliases, provider-specific metadata, capability hints)
        * Applies alias-aware restriction checks before exposing models to the
          registry or tooling
        * Reuses :class:`OpenAICompatibleProvider` infrastructure for request
          execution so OpenRouter endpoints behave like standard OpenAI-style
          APIs.
    """

    FRIENDLY_NAME = "OpenRouter"

    # Custom headers required by OpenRouter
    DEFAULT_HEADERS = {
        "HTTP-Referer": get_env("OPENROUTER_REFERER", "https://github.com/BeehiveInnovations/pal-mcp-server")
        or "https://github.com/BeehiveInnovations/pal-mcp-server",
        "X-Title": get_env("OPENROUTER_TITLE", "PAL MCP Server") or "PAL MCP Server",
    }

    # Model registry for managing configurations and aliases
    _registry: OpenRouterModelRegistry | None = None

    # Fusion family: canonical model name -> panel preset (None = use env default).
    # All variants share the same wire model; the preset/plugin selects the panel.
    _FUSION_WIRE_MODEL = "openrouter/fusion"
    _FUSION_VARIANT_PRESETS: dict[str, str | None] = {
        "openrouter/fusion": None,
        "openrouter/fusion-high": "general-high",
        "openrouter/fusion-budget": "general-budget",
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            **kwargs: Additional configuration
        """
        base_url = "https://openrouter.ai/api/v1"
        self._alias_cache: dict[str, str] = {}
        super().__init__(api_key, base_url=base_url, **kwargs)

        # Initialize model registry
        if OpenRouterProvider._registry is None:
            OpenRouterProvider._registry = OpenRouterModelRegistry()
            # Log loaded models and aliases only on first load
            models = self._registry.list_models()
            aliases = self._registry.list_aliases()
            logging.info(f"OpenRouter loaded {len(models)} models with {len(aliases)} aliases")

    # ------------------------------------------------------------------
    # Capability surface
    # ------------------------------------------------------------------

    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: str | None = None,
    ) -> ModelCapabilities | None:
        """Fetch OpenRouter capabilities from the registry or build a generic fallback."""

        capabilities = self._registry.get_capabilities(canonical_name)
        if capabilities:
            return capabilities

        base_identifier = canonical_name.split(":", 1)[0]
        if "/" in base_identifier:
            logging.debug(
                "Using generic OpenRouter capabilities for %s (provider/model format detected)", canonical_name
            )
            generic = ModelCapabilities(
                provider=ProviderType.OPENROUTER,
                model_name=canonical_name,
                friendly_name=self.FRIENDLY_NAME,
                intelligence_score=9,
                context_window=32_768,
                max_output_tokens=32_768,
                supports_extended_thinking=False,
                supports_system_prompts=True,
                supports_streaming=True,
                supports_function_calling=False,
                temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            )
            generic._is_generic = True
            return generic

        logging.debug(
            "Rejecting unknown OpenRouter model '%s' (no provider prefix); requires explicit configuration",
            canonical_name,
        )
        return None

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    def get_provider_type(self) -> ProviderType:
        """Identify this provider for restrictions and logging."""
        return ProviderType.OPENROUTER

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def list_models(
        self,
        *,
        respect_restrictions: bool = True,
        include_aliases: bool = True,
        lowercase: bool = False,
        unique: bool = False,
    ) -> list[str]:
        """Return formatted OpenRouter model names, respecting alias-aware restrictions."""

        if not self._registry:
            return []

        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service() if respect_restrictions else None
        allowed_configs: dict[str, ModelCapabilities] = {}

        for model_name in self._registry.list_models():
            config = self._registry.resolve(model_name)
            if not config:
                continue

            # Custom models belong to CustomProvider; skip them here so the two
            # providers don't race over the same registrations (important for tests
            # that stub the registry with minimal objects lacking attrs).
            if config.provider == ProviderType.CUSTOM:
                continue

            if restriction_service:
                allowed = restriction_service.is_allowed(self.get_provider_type(), model_name)

                if not allowed and config.aliases:
                    for alias in config.aliases:
                        if restriction_service.is_allowed(self.get_provider_type(), alias):
                            allowed = True
                            break

                if not allowed:
                    continue

            allowed_configs[model_name] = config

        if not allowed_configs:
            return []

        # When restrictions are in place, don't include aliases to avoid confusion
        # Only return the canonical model names that are actually allowed
        actual_include_aliases = include_aliases and not respect_restrictions

        return ModelCapabilities.collect_model_names(
            allowed_configs,
            include_aliases=actual_include_aliases,
            lowercase=lowercase,
            unique=unique,
        )

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve aliases defined in the OpenRouter registry."""

        cache_key = model_name.lower()
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]

        config = self._registry.resolve(model_name)
        if config:
            if config.model_name != model_name:
                logging.debug("Resolved model alias '%s' to '%s'", model_name, config.model_name)
            resolved = config.model_name
            self._alias_cache[cache_key] = resolved
            self._alias_cache.setdefault(resolved.lower(), resolved)
            return resolved

        logging.debug(f"Model '{model_name}' not found in registry, using as-is")
        self._alias_cache[cache_key] = model_name
        return model_name

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        """Expose registry-backed OpenRouter capabilities."""

        if not self._registry:
            return {}

        capabilities: dict[str, ModelCapabilities] = {}
        for model_name in self._registry.list_models():
            config = self._registry.resolve(model_name)
            if not config:
                continue

            # See note in list_models: respect the CustomProvider boundary.
            if config.provider == ProviderType.CUSTOM:
                continue

            capabilities[model_name] = config
        return capabilities

    # ------------------------------------------------------------------
    # Fusion panel selection
    # ------------------------------------------------------------------

    def _resolve_fusion_request(self, model_name: str) -> tuple[Optional[str], Optional[dict]]:
        """Detect a Fusion-family request and build its wire model + extra_body.

        Fusion variants (``fusion``/``fusion-high``/``fusion-budget``) all map to the
        single wire model ``openrouter/fusion``; the panel is selected via a
        ``plugins`` entry. The preset comes from the variant, falling back to
        ``OPENROUTER_FUSION_PRESET`` (default ``general-high``) for bare ``fusion``.
        Optional env vars override panel/judge/tool-call budget; per OpenRouter,
        explicit ``analysis_models``/``model`` take precedence over the preset.

        Returns ``(None, None)`` for non-Fusion requests.
        """

        config = self._registry.resolve(model_name) if self._registry else None
        canonical = config.model_name if config else model_name
        if canonical not in self._FUSION_VARIANT_PRESETS:
            return None, None

        preset = self._FUSION_VARIANT_PRESETS[canonical]
        if preset is None:
            preset = get_env("OPENROUTER_FUSION_PRESET", "general-high") or "general-high"

        plugin: dict = {"id": "fusion"}
        if preset:
            plugin["preset"] = preset

        analysis_models = get_env("OPENROUTER_FUSION_ANALYSIS_MODELS")
        if analysis_models:
            models = [m.strip() for m in analysis_models.split(",") if m.strip()]
            if models:
                plugin["analysis_models"] = models

        judge = get_env("OPENROUTER_FUSION_JUDGE")
        if judge and judge.strip():
            plugin["model"] = judge.strip()

        max_tool_calls = get_env("OPENROUTER_FUSION_MAX_TOOL_CALLS")
        if max_tool_calls and max_tool_calls.strip():
            try:
                plugin["max_tool_calls"] = int(max_tool_calls.strip())
            except ValueError:
                logging.warning("Invalid OPENROUTER_FUSION_MAX_TOOL_CALLS=%r; ignoring", max_tool_calls)

        return self._FUSION_WIRE_MODEL, {"plugins": [plugin]}

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
        images: list[str] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Route Fusion-family aliases to the wire model with a panel plugin.

        Non-Fusion requests fall straight through to the shared OpenAI-compatible
        implementation unchanged.
        """

        wire_model, extra_body = self._resolve_fusion_request(model_name)
        if wire_model is not None:
            existing = kwargs.get("extra_body")
            kwargs["extra_body"] = {**existing, **extra_body} if isinstance(existing, dict) else extra_body
            logging.debug("Fusion request '%s' -> wire '%s' with extra_body=%s", model_name, wire_model, extra_body)
            model_name = wire_model

        return super().generate_content(
            prompt,
            model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            images=images,
            **kwargs,
        )
