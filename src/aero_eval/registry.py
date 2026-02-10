"""Plugin registry for scorer classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from aero_eval.models import ScorerTier

if TYPE_CHECKING:
    from aero_eval.scorers.base import BaseScorer


class ScorerRegistry:
    """Central registry for all scorer classes."""

    _scorers: dict[str, Type[BaseScorer]] = {}
    _tier_map: dict[str, ScorerTier] = {}

    @classmethod
    def register(cls, name: str, tier: ScorerTier):
        """Decorator to register a scorer class."""

        def decorator(scorer_cls: Type[BaseScorer]) -> Type[BaseScorer]:
            cls._scorers[name] = scorer_cls
            cls._tier_map[name] = tier
            return scorer_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseScorer]:
        """Look up a scorer class by name."""
        if name not in cls._scorers:
            available = ", ".join(sorted(cls._scorers.keys()))
            raise KeyError(
                f"Scorer '{name}' not found. Available: {available}"
            )
        return cls._scorers[name]

    @classmethod
    def list_all(
        cls, tier: ScorerTier | None = None
    ) -> list[tuple[str, ScorerTier]]:
        """List all registered scorers, optionally filtered by tier."""
        items = [
            (name, cls._tier_map[name])
            for name in sorted(cls._scorers.keys())
        ]
        if tier is not None:
            items = [(n, t) for n, t in items if t == tier]
        return items

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseScorer:
        """Factory method: instantiate a scorer by name."""
        scorer_cls = cls.get(name)
        return scorer_cls(**kwargs)
