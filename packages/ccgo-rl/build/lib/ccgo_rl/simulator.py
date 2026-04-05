from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for synthetic CCGO benchmark simulation."""

    seeds: list[int]
    datasets: list[str]
    baselines: list[str]
    claim: str


@dataclass(frozen=True)
class MetricSummary:
    """Per-metric aggregate with confidence interval."""

    mean: float
    std: float
    ci_low: float
    ci_high: float


class CCGORLSimulator:
    """Reusable simulator adapted from the project validation implementation."""

    DEFAULT_BASELINE_OFFSETS: dict[str, float] = {
        "CCGO_adaptive": 0.12,
        "HER_SAC": 0.0,
        "RND_fixed_beta": -0.05,
        "ICM_fixed_beta": -0.06,
        "ACWI_style": -0.02,
        "Contrastive_always_on": -0.03,
        "CRL_style": -0.01,
        "ViSA_style": -0.015,
        "ACDC_style": -0.02,
        "CCGO_fixed_beta": -0.04,
        "Agent57_lite": -0.01,
        "Plan2Explore_lite": -0.015,
    }

    def __init__(self, baseline_offsets: dict[str, float] | None = None) -> None:
        self._baseline_offsets = baseline_offsets or dict(self.DEFAULT_BASELINE_OFFSETS)

    @staticmethod
    def bootstrap_ci(
        values: Sequence[float],
        n_boot: int = 1000,
        alpha: float = 0.05,
        seed: int = 20260405,
    ) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError("bootstrap_ci requires at least one value")
        rng = np.random.default_rng(seed)
        if arr.size == 1:
            val = float(arr[0])
            return val, val

        samples: list[float] = []
        for _ in range(n_boot):
            idx = rng.integers(0, arr.size, size=arr.size)
            samples.append(float(arr[idx].mean()))
        return float(np.quantile(samples, alpha / 2.0)), float(
            np.quantile(samples, 1.0 - alpha / 2.0)
        )

    @staticmethod
    def paired_bootstrap_delta(
        a: Sequence[float],
        b: Sequence[float],
        n_boot: int = 3000,
        seed: int = 777,
    ) -> tuple[float, float, float]:
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        if arr_a.shape != arr_b.shape:
            raise ValueError("paired_bootstrap_delta requires arrays with equal shape")
        if arr_a.size == 0:
            raise ValueError("paired_bootstrap_delta requires non-empty arrays")

        rng = np.random.default_rng(seed)
        delta = arr_a - arr_b
        boots: list[float] = []
        for _ in range(n_boot):
            idx = rng.integers(0, delta.size, size=delta.size)
            boots.append(float(delta[idx].mean()))
        return (
            float(np.mean(boots)),
            float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)),
        )

    def simulate_benchmark_rows(self, config: BenchmarkConfig) -> list[dict[str, Any]]:
        """Generate synthetic benchmark rows aligned with CCGO validation metrics."""
        rows: list[dict[str, Any]] = []
        for dataset_idx, dataset in enumerate(config.datasets):
            for seed in config.seeds:
                rng = np.random.default_rng(seed + 100 * dataset_idx)
                for baseline in config.baselines:
                    offset = self._baseline_offsets.get(baseline, -0.02)
                    if config.claim == "h2" and "Contrastive" in baseline:
                        offset -= 0.01

                    noise = rng.normal(0.0, 0.02)
                    final_return = np.clip(0.62 + offset + noise, 0.0, 1.0)
                    auc = np.clip(0.58 + 0.8 * offset + rng.normal(0.0, 0.02), 0.0, 1.0)
                    first_success = max(
                        10.0,
                        180.0 - 80.0 * (offset + 0.1) + rng.normal(0.0, 6.0),
                    )
                    coverage = np.clip(
                        0.55 + 0.6 * (offset + 0.05) + rng.normal(0.0, 0.03),
                        0.0,
                        1.0,
                    )
                    beta_mass = max(
                        0.0,
                        0.28 - 1.1 * max(offset, -0.08) + rng.normal(0.0, 0.015),
                    )
                    bound_residual = max(
                        0.0,
                        0.03 + 0.12 * max(-offset, 0.0) + rng.normal(0.0, 0.01),
                    )
                    td_var = max(0.005, 0.12 - 0.25 * offset + rng.normal(0.0, 0.01))
                    gate_slope = max(0.0, 0.98 + rng.normal(0.0, 0.07))
                    reward_dom = np.clip(
                        0.28 - 0.3 * max(offset, -0.05) + rng.normal(0.0, 0.03),
                        0.0,
                        1.0,
                    )
                    critic_drift = max(
                        0.0,
                        0.24 - 0.3 * max(offset, -0.05) + rng.normal(0.0, 0.03),
                    )
                    rows.append(
                        {
                            "claim": config.claim,
                            "dataset": dataset,
                            "seed": int(seed),
                            "baseline": baseline,
                            "final_return": float(final_return),
                            "auc": float(auc),
                            "first_success_episode": float(first_success),
                            "coverage": float(coverage),
                            "mbeta_hat": float(beta_mass),
                            "bound_residual": float(bound_residual),
                            "td_target_variance": float(td_var),
                            "gate_sensitivity_slope": float(gate_slope),
                            "reward_channel_dominance": float(reward_dom),
                            "critic_drift_rate": float(critic_drift),
                        }
                    )
        return rows

    def simulate_counterexamples(
        self,
        seeds: Sequence[int],
        datasets: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Generate failure-mode traces for C1/C2 assumption stress tests."""
        schedules = ["slow_decay_beta", "oscillatory_beta", "saturated_gate"]
        rows: list[dict[str, Any]] = []
        for dataset_idx, dataset in enumerate(datasets):
            for seed in seeds:
                rng = np.random.default_rng(5000 + seed + dataset_idx * 31)
                for schedule in schedules:
                    failure_prob = 0.74 if schedule != "saturated_gate" else 0.82
                    fail = rng.random() < failure_prob
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "counterexample_schedule": schedule,
                            "assumption_violation_tag": (
                                "C1_F2" if schedule != "saturated_gate" else "C2_F2"
                            ),
                            "bound_violation": int(fail),
                            "return_drop_pct": float(
                                max(0.0, rng.normal(16.0 if fail else 4.0, 3.0))
                            ),
                            "runtime_minutes": float(max(30.0, rng.normal(145.0, 12.0))),
                        }
                    )
        return rows

    def summarize_by_baseline(
        self,
        rows: Iterable[dict[str, Any]],
        metrics: Sequence[str],
    ) -> dict[str, dict[str, MetricSummary]]:
        grouped: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            baseline = str(row["baseline"])
            grouped.setdefault(baseline, {})
            for metric in metrics:
                grouped[baseline].setdefault(metric, []).append(float(row[metric]))

        summary: dict[str, dict[str, MetricSummary]] = {}
        for baseline, metric_map in grouped.items():
            summary[baseline] = {}
            for metric, values in metric_map.items():
                arr = np.asarray(values, dtype=float)
                ci_low, ci_high = self.bootstrap_ci(arr.tolist())
                summary[baseline][metric] = MetricSummary(
                    mean=float(arr.mean()),
                    std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
        return summary

    @staticmethod
    def summary_as_dict(
        summary: dict[str, dict[str, MetricSummary]],
    ) -> dict[str, dict[str, dict[str, float]]]:
        return {
            baseline: {metric: asdict(metric_summary) for metric, metric_summary in metrics.items()}
            for baseline, metrics in summary.items()
        }
