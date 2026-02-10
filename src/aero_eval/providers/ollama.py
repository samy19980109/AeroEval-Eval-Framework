"""Ollama inference provider."""

from __future__ import annotations

import asyncio
import time

import httpx

from aero_eval.models import ProviderConfig
from aero_eval.providers.base import InferenceResult, ModelProvider


class OllamaProvider(ModelProvider):
    """Async Ollama provider using REST API."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._base_url = config.base_url or "http://localhost:11434"
        self.client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=config.timeout_seconds,
        )

    async def generate(
        self, prompt: str, system: str | None = None, **kwargs
    ) -> InferenceResult:
        t0 = time.perf_counter()
        ttft = None
        full_text = ""

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with self.client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json
                data = json.loads(line)
                if "response" in data:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    full_text += data["response"]

        total_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            text=full_text,
            logprobs=None,
            tokens=None,
            ttft_ms=ttft,
            total_latency_ms=total_ms,
            tokens_generated=len(full_text.split()),
            finish_reason="stop",
        )

    async def generate_batch(
        self, prompts: list[str], concurrency: int = 5, **kwargs
    ) -> list[InferenceResult]:
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(p: str) -> InferenceResult:
            async with sem:
                return await self.generate(p, **kwargs)

        return await asyncio.gather(*[_bounded(p) for p in prompts])

    async def close(self) -> None:
        await self.client.aclose()
