from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .simulator import BenchmarkConfig, CCGORLSimulator
from .symbolic import SymbolicVerifier


class CCGOService:
    """High-level package service for simulation and symbolic audit workflows."""

    def __init__(
        self,
        simulator: CCGORLSimulator | None = None,
        symbolic_verifier: SymbolicVerifier | None = None,
    ) -> None:
        self.simulator = simulator or CCGORLSimulator()
        self.symbolic_verifier = symbolic_verifier or SymbolicVerifier()

    def run_simulation(
        self,
        seeds: list[int],
        datasets: list[str],
        baselines: list[str],
        claim: str,
    ) -> list[dict[str, Any]]:
        config = BenchmarkConfig(
            seeds=seeds,
            datasets=datasets,
            baselines=baselines,
            claim=claim,
        )
        return self.simulator.simulate_benchmark_rows(config)

    def summarize(
        self,
        rows: list[dict[str, Any]],
        metrics: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        summary = self.simulator.summarize_by_baseline(rows=rows, metrics=metrics)
        return self.simulator.summary_as_dict(summary)

    def run_symbolic_checks(
        self,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self.symbolic_verifier.run_checks(
            None if output_dir is None else Path(output_dir)
        )
        return [asdict(item) for item in results]

    def run_validation_bundle(
        self,
        seeds: list[int],
        datasets: list[str],
        baselines: list[str],
        claim: str,
        metrics: list[str],
        symbolic_output_dir: str | None = None,
    ) -> dict[str, Any]:
        rows = self.run_simulation(
            seeds=seeds,
            datasets=datasets,
            baselines=baselines,
            claim=claim,
        )
        summary = self.summarize(rows=rows, metrics=metrics)
        symbolic_results = self.run_symbolic_checks(output_dir=symbolic_output_dir)
        return {
            "rows": rows,
            "summary": summary,
            "symbolic_results": symbolic_results,
        }
