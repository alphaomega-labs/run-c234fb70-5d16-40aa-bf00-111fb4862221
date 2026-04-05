from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk", palette="colorblind")


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_h1_main(df: pd.DataFrame, out_path: Path) -> None:
    part = df[df["baseline"].isin(["CCGO_adaptive", "HER_SAC", "RND_fixed_beta", "ICM_fixed_beta"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.barplot(data=part, x="baseline", y="final_return", hue="dataset", ax=axes[0, 0], errorbar="sd")
    axes[0, 0].set_title("Final Return by Baseline")
    axes[0, 0].set_ylabel("Final Return (normalized)")
    axes[0, 0].set_xlabel("Baseline")
    axes[0, 0].tick_params(axis="x", rotation=20)

    sns.barplot(data=part, x="baseline", y="auc", hue="dataset", ax=axes[0, 1], errorbar="sd")
    axes[0, 1].set_title("AUC by Baseline")
    axes[0, 1].set_ylabel("AUC (normalized)")
    axes[0, 1].set_xlabel("Baseline")
    axes[0, 1].tick_params(axis="x", rotation=20)

    sns.boxplot(data=part, x="baseline", y="first_success_episode", ax=axes[1, 0])
    axes[1, 0].set_title("First-Success Episode")
    axes[1, 0].set_ylabel("Episode")
    axes[1, 0].set_xlabel("Baseline")
    axes[1, 0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=part, x="baseline", y="coverage", ax=axes[1, 1])
    axes[1, 1].set_title("State-Action Coverage")
    axes[1, 1].set_ylabel("Coverage (ratio)")
    axes[1, 1].set_xlabel("Baseline")
    axes[1, 1].tick_params(axis="x", rotation=20)

    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    _finish(fig, out_path)


def fig_h1_bound(df: pd.DataFrame, out_path: Path) -> None:
    part = df[df["baseline"].isin(["CCGO_adaptive", "HER_SAC", "RND_fixed_beta", "ICM_fixed_beta"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=part, x="mbeta_hat", y="bound_residual", hue="baseline", style="dataset", ax=axes[0])
    axes[0].set_title("Empirical M_beta vs Bound Residual")
    axes[0].set_xlabel("M_beta estimate")
    axes[0].set_ylabel("Gap - (2 R_i M_beta + epsilon)")

    sns.boxplot(data=part, x="baseline", y="reward_channel_dominance", ax=axes[1])
    axes[1].set_title("Reward Channel Dominance")
    axes[1].set_xlabel("Baseline")
    axes[1].set_ylabel("Dominance ratio")
    axes[1].tick_params(axis="x", rotation=20)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8)
    _finish(fig, out_path)


def fig_h2_main(df: pd.DataFrame, out_path: Path) -> None:
    part = df[df["baseline"].isin(["CCGO_adaptive", "Contrastive_always_on", "HER_SAC", "CRL_style"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=part, x="td_target_variance", y="final_return", hue="baseline", style="dataset", ax=axes[0])
    axes[0].set_title("Success/Return vs TD Variance")
    axes[0].set_xlabel("TD-target variance")
    axes[0].set_ylabel("Final Return (normalized)")

    sns.boxplot(data=part, x="baseline", y="critic_drift_rate", ax=axes[1])
    axes[1].set_title("Critic Drift Stability")
    axes[1].set_xlabel("Baseline")
    axes[1].set_ylabel("Drift norm (a.u.)")
    axes[1].tick_params(axis="x", rotation=20)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8)
    _finish(fig, out_path)


def fig_h2_gate(df: pd.DataFrame, out_path: Path, theoretical: float) -> None:
    part = df[df["baseline"].isin(["CCGO_adaptive", "Contrastive_always_on", "CRL_style"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(data=part, x="gate_sensitivity_slope", hue="baseline", fill=True, common_norm=False, ax=axes[0])
    axes[0].axvline(theoretical, color="black", linestyle="--", label="kappa/4")
    axes[0].set_title("Gate Sensitivity Calibration")
    axes[0].set_xlabel("Empirical gate slope")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    sns.scatterplot(data=part, x="gate_sensitivity_slope", y="td_target_variance", hue="baseline", style="dataset", ax=axes[1])
    axes[1].axvline(theoretical, color="black", linestyle="--")
    axes[1].set_title("Variance vs Gate Slope")
    axes[1].set_xlabel("Empirical gate slope")
    axes[1].set_ylabel("TD-target variance")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, fontsize=8)
    _finish(fig, out_path)


def fig_transfer_counterexample(cx_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=cx_df, x="counterexample_schedule", y="bound_violation", hue="dataset", estimator="mean", ax=axes[0])
    axes[0].set_title("Bound-Violation Incidence")
    axes[0].set_xlabel("Counterexample schedule")
    axes[0].set_ylabel("Violation rate")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=cx_df, x="counterexample_schedule", y="return_drop_pct", ax=axes[1])
    axes[1].set_title("Transfer Drop Without Retuning")
    axes[1].set_xlabel("Counterexample schedule")
    axes[1].set_ylabel("Return drop (%)")
    axes[1].tick_params(axis="x", rotation=20)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8)
    _finish(fig, out_path)


def fig_symbolic_audit(table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    work = table.copy()
    work["passed"] = work["passed"].astype(float)
    pass_rate = work.groupby("claim")["passed"].mean().reset_index()
    sns.barplot(data=pass_rate, x="claim", y="passed", ax=axes[0])
    axes[0].set_title("Symbolic Pass Rate")
    axes[0].set_xlabel("Claim")
    axes[0].set_ylabel("Pass rate")

    sns.barplot(data=table, x="check_id", y="numeric_error", hue="claim", ax=axes[1])
    axes[1].set_title("Numeric-Symbolic Agreement")
    axes[1].set_xlabel("Check")
    axes[1].set_ylabel("|numeric - symbolic|")
    axes[1].tick_params(axis="x", rotation=30)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, fontsize=8)
    _finish(fig, out_path)
