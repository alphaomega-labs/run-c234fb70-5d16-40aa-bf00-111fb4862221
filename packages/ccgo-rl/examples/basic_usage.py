from __future__ import annotations

from ccgo_rl import CCGOService


def main() -> None:
    service = CCGOService()
    rows = service.run_simulation(
        seeds=[7, 11, 23],
        datasets=["sparse_goal_nav"],
        baselines=["CCGO_adaptive", "HER_SAC", "Contrastive_always_on"],
        claim="h1",
    )
    summary = service.summarize(rows=rows, metrics=["final_return", "auc", "bound_residual"])

    print("rows:", len(rows))
    print("baselines:", ", ".join(sorted(summary.keys())))
    print("ccgo final_return mean:", summary["CCGO_adaptive"]["final_return"]["mean"])


if __name__ == "__main__":
    main()
