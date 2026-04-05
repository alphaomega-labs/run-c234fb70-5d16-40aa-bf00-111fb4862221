"""Public package API for CCGO-RL release artifacts."""

from .service import CCGOService
from .simulator import (
    BenchmarkConfig,
    CCGORLSimulator,
    MetricSummary,
    simulate_benchmark_rows,
    simulate_counterexamples,
    summarize_by_baseline,
)
from .symbolic import SymbolicCheckResult, SymbolicVerifier, run_checks

__all__ = [
    "BenchmarkConfig",
    "CCGORLSimulator",
    "MetricSummary",
    "simulate_benchmark_rows",
    "simulate_counterexamples",
    "summarize_by_baseline",
    "SymbolicCheckResult",
    "SymbolicVerifier",
    "run_checks",
    "CCGOService",
]

__version__ = "0.1.0"
