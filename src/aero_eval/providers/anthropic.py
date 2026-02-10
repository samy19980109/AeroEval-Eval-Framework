"""Anthropic inference provider."""

from __future__ import annotations

import asyncio
import time

from anthropic import AsyncAnthropic

from aero_eval.models import ProviderConfig
from aero_eval.providers.base import InferenceResult, ModelProvider


class AnthropicProvider(ModelProvider):
    """Async Anthropic provider with streaming support."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    async def generate(
        self, prompt: str, system: str | None = None, **kwargs
    ) -> InferenceResult:
        t0 = time.perf_counter()
        ttft = None
        full_text = ""

        async with self.client.messages.stream(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=system or "",
            temperature=self.config.temperature,
        ) as stream:
            async for text in stream.text_stream:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                full_text += text

        total_ms = (time.perf_counter() - t0) * 1000
        message = await stream.get_final_message()
        token_count = message.usage.output_tokens if message.usage else len(full_text.split())

        return InferenceResult(
            text=full_text,
            logprobs=None,  # Anthropic doesn't expose logprobs
            tokens=None,
            ttft_ms=ttft,
            total_latency_ms=total_ms,
            tokens_generated=token_count,
            finish_reason=message.stop_reason or "stop",
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
        await self.client.close()
