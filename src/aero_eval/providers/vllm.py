"""vLLM inference provider (OpenAI-compatible API)."""

from __future__ import annotations

from aero_eval.models import ProviderConfig
from aero_eval.providers.openai import OpenAIProvider


class VLLMProvider(OpenAIProvider):
    """vLLM provider that reuses OpenAI's client with a custom base_url.

    vLLM exposes an OpenAI-compatible API, so we can reuse the OpenAI
    provider with the base_url pointing to the vLLM server.
    """

    def __init__(self, config: ProviderConfig):
        if not config.base_url:
            config = config.model_copy(
                update={"base_url": "http://localhost:8000/v1"}
            )
        super().__init__(config)
