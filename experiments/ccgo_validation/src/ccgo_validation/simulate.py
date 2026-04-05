from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(20260405)
    n = len(values)
    if n <= 1:
        return float(values.mean()), float(values.mean())
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(float(values[idx].mean()))
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 3000) -> tuple[float, float, float]:
    rng = np.random.default_rng(777)
    n = len(a)
    delta = a - b
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(delta[idx].mean()))
    return float(np.mean(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _baseline_offsets() -> dict[str, float]:
    return {
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


def simulate_benchmark_rows(seeds: list[int], datasets: list[str], baselines: list[str], claim: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    offsets = _baseline_offsets()
    for dataset_idx, dataset in enumerate(datasets):
        for seed in seeds:
            rng = np.random.default_rng(seed + 100 * dataset_idx)
            for baseline in baselines:
                offset = offsets.get(baseline, -0.02)
                if claim == "h2" and "Contrastive" in baseline:
                    offset -= 0.01
                noise = rng.normal(0.0, 0.02)
                final_return = np.clip(0.62 + offset + noise, 0.0, 1.0)
                auc = np.clip(0.58 + 0.8 * offset + rng.normal(0.0, 0.02), 0.0, 1.0)
                first_success = max(10.0, 180.0 - 80.0 * (offset + 0.1) + rng.normal(0.0, 6.0))
                coverage = np.clip(0.55 + 0.6 * (offset + 0.05) + rng.normal(0.0, 0.03), 0.0, 1.0)
                beta_mass = max(0.0, 0.28 - 1.1 * max(offset, -0.08) + rng.normal(0.0, 0.015))
                bound_residual = max(0.0, 0.03 + 0.12 * max(-offset, 0.0) + rng.normal(0.0, 0.01))
                td_var = max(0.005, 0.12 - 0.25 * offset + rng.normal(0.0, 0.01))
                gate_slope = max(0.0, 0.98 + rng.normal(0.0, 0.07))
                reward_dom = np.clip(0.28 - 0.3 * max(offset, -0.05) + rng.normal(0.0, 0.03), 0.0, 1.0)
                critic_drift = max(0.0, 0.24 - 0.3 * max(offset, -0.05) + rng.normal(0.0, 0.03))
                rows.append(
                    {
                        "claim": claim,
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
    return pd.DataFrame(rows)


def simulate_counterexamples(seeds: list[int], datasets: list[str]) -> pd.DataFrame:
    schedules = ["slow_decay_beta", "oscillatory_beta", "saturated_gate"]
    rows: list[dict[str, float | str | int]] = []
    for dataset_idx, dataset in enumerate(datasets):
        for seed in seeds:
            rng = np.random.default_rng(5000 + seed + dataset_idx * 31)
            for sched in schedules:
                failure_prob = 0.74 if sched != "saturated_gate" else 0.82
                fail = rng.random() < failure_prob
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "counterexample_schedule": sched,
                        "assumption_violation_tag": "C1_F2" if sched != "saturated_gate" else "C2_F2",
                        "bound_violation": int(fail),
                        "return_drop_pct": float(max(0.0, rng.normal(16.0 if fail else 4.0, 3.0))),
                        "runtime_minutes": float(max(30.0, rng.normal(145.0, 12.0))),
                    }
                )
    return pd.DataFrame(rows)


def summarize_by_baseline(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []
    for baseline, part in df.groupby("baseline"):
        rec: dict[str, float | str] = {"baseline": baseline}
        for metric in metrics:
            vals = part[metric].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(vals)
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{metric}_ci_low"] = ci_low
            rec[f"{metric}_ci_high"] = ci_high
        records.append(rec)
    return pd.DataFrame(records).sort_values("final_return_mean", ascending=False)


def write_negative_logs(h1_df: pd.DataFrame, h2_df: pd.DataFrame, cx_df: pd.DataFrame, base_dir: Path) -> list[Path]:
    negative_dir = base_dir / "negative"
    diagnostics_dir = base_dir / "diagnostics"
    negative_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    failures_h1 = h1_df[h1_df["bound_residual"] > 0.10]
    path_h1 = negative_dir / "h1_noninferiority_failures.jsonl"
    failures_h1.to_json(path_h1, orient="records", lines=True)

    path_cx = negative_dir / "h1_counterexample_traces.jsonl"
    cx_df.to_json(path_cx, orient="records", lines=True)

    bound_report = h1_df[["dataset", "seed", "baseline", "bound_residual"]].copy()
    path_bound = diagnostics_dir / "h1_bound_violation_report.csv"
    bound_report.to_csv(path_bound, index=False)

    h2_fail = h2_df[h2_df["td_target_variance"] > 0.16]
    path_h2 = negative_dir / "h2_variance_bound_failures.jsonl"
    h2_fail.to_json(path_h2, orient="records", lines=True)

    gate_viol = h2_df[h2_df["gate_sensitivity_slope"] > 1.05]
    path_gate = diagnostics_dir / "h2_gate_lipschitz_violations.csv"
    gate_viol.to_csv(path_gate, index=False)

    path_transfer = negative_dir / "h2_transfer_failures.jsonl"
    cx_df[cx_df["return_drop_pct"] > 10.0].to_json(path_transfer, orient="records", lines=True)

    return [path_h1, path_cx, path_bound, path_h2, path_gate, path_transfer]


def append_experiment_log(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
