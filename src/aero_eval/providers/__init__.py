"""Inference provider package."""

from aero_eval.models import ProviderConfig, ProviderType
from aero_eval.providers.base import InferenceResult, ModelProvider


def create_provider(config: ProviderConfig) -> ModelProvider:
    """Factory function to create a provider from config."""
    if config.provider_type == ProviderType.OPENAI:
        from aero_eval.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    elif config.provider_type == ProviderType.ANTHROPIC:
        from aero_eval.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    elif config.provider_type == ProviderType.VLLM:
        from aero_eval.providers.vllm import VLLMProvider
        return VLLMProvider(config)
    elif config.provider_type == ProviderType.OLLAMA:
        from aero_eval.providers.ollama import OllamaProvider
        return OllamaProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {config.provider_type}")
