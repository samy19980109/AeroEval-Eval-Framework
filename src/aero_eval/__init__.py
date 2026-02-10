"""Aero-Eval: High-performance LLM evaluation framework."""

__version__ = "0.1.0"

# Import scorers to trigger registration with the ScorerRegistry
import aero_eval.scorers  # noqa: F401
