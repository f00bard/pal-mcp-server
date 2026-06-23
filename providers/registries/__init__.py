"""Registry implementations for provider capability manifests."""

from .azure import AzureModelRegistry
from .custom import CustomEndpointModelRegistry
from .deepseek import DeepSeekModelRegistry
from .dial import DialModelRegistry
from .gemini import GeminiModelRegistry
from .openai import OpenAIModelRegistry
from .openrouter import OpenRouterModelRegistry
from .xai import XAIModelRegistry

__all__ = [
    "AzureModelRegistry",
    "CustomEndpointModelRegistry",
    "DeepSeekModelRegistry",
    "DialModelRegistry",
    "GeminiModelRegistry",
    "OpenAIModelRegistry",
    "OpenRouterModelRegistry",
    "XAIModelRegistry",
]
