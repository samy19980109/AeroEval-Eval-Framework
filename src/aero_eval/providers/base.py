"""Abstract base for inference providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class InferenceResult(BaseModel):
    """Result from a single inference call."""

    text: str
    logprobs: list[float] | None = None
    tokens: list[str] | None = None
    ttft_ms: float | None = None
    total_latency_ms: float
    tokens_generated: int
    finish_reason: str = "stop"
    metadata: dict[str, Any] = {}


class ModelProvider(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    async def generate(
        self, prompt: str, system: str | None = None, **kwargs
    ) -> InferenceResult:
        ...

    @abstractmethod
    async def generate_batch(
        self, prompts: list[str], concurrency: int = 5, **kwargs
    ) -> list[InferenceResult]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
