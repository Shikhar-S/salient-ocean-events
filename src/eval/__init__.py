"""Event-based evaluation: matchers and metrics (time-only)."""

from src.eval.matching import Matcher, CollarMatcher, IoUMatcher
from src.eval.metrics import EvalResult, evaluate
from src.eval.benchmark import run_benchmark, BenchmarkRow

__all__ = [
    "Matcher",
    "CollarMatcher",
    "IoUMatcher",
    "EvalResult",
    "evaluate",
    "run_benchmark",
    "BenchmarkRow",
]
