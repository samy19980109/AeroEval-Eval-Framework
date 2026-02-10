"""OpenAI inference provider with logprob capture."""

from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

from aero_eval.models import ProviderConfig
from aero_eval.providers.base import InferenceResult, ModelProvider


class OpenAIProvider(ModelProvider):
    """Async OpenAI provider with streaming and logprob support."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def generate(
        self, prompt: str, system: str | None = None, **kwargs
    ) -> InferenceResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        ttft = None
        full_text = ""
        logprobs_list: list[float] = []
        tokens_list: list[str] = []

        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            logprobs=self.config.enable_logprobs,
            top_logprobs=self.config.top_logprobs if self.config.enable_logprobs else None,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                full_text += chunk.choices[0].delta.content

            if (
                self.config.enable_logprobs
                and chunk.choices
                and chunk.choices[0].logprobs
                and chunk.choices[0].logprobs.content
            ):
                for lp in chunk.choices[0].logprobs.content:
                    logprobs_list.append(lp.logprob)
                    tokens_list.append(lp.token)

        total_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            text=full_text,
            logprobs=logprobs_list or None,
            tokens=tokens_list or None,
            ttft_ms=ttft,
            total_latency_ms=total_ms,
            tokens_generated=len(tokens_list) if tokens_list else len(full_text.split()),
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
        await self.client.close()
