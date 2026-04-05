from __future__ import annotations

from pathlib import Path

from ccgo_rl import (
    BenchmarkConfig,
    CCGORLSimulator,
    SymbolicVerifier,
    run_checks,
    simulate_benchmark_rows,
    summarize_by_baseline,
)


def test_simulator_rows_and_summary() -> None:
    simulator = CCGORLSimulator()
    config = BenchmarkConfig(
        seeds=[1, 2],
        datasets=["sparse_goal_nav"],
        baselines=["CCGO_adaptive", "HER_SAC"],
        claim="h1",
    )
    rows = simulator.simulate_benchmark_rows(config)
    assert len(rows) == 4

    summary = simulator.summarize_by_baseline(rows, metrics=["final_return", "auc"])
    assert "CCGO_adaptive" in summary
    assert "HER_SAC" in summary
    assert summary["CCGO_adaptive"]["final_return"].ci_low <= summary["CCGO_adaptive"]["final_return"].ci_high


def test_symbolic_verifier_writes_outputs(tmp_path: Path) -> None:
    verifier = SymbolicVerifier()
    results = verifier.run_checks(output_dir=tmp_path)

    assert len(results) == 5
    ids = {item.check_id for item in results}
    assert "sympy_c1_triangle_bound" in ids
    assert "sympy_c1_limit_corollary" in ids
    assert (tmp_path / "theorem_check_table.csv").exists()
    assert (tmp_path / "sympy_report.md").exists()


def test_module_level_wrappers(tmp_path: Path) -> None:
    rows = simulate_benchmark_rows(
        seeds=[3],
        datasets=["deceptive_maze"],
        baselines=["CCGO_adaptive", "HER_SAC"],
        claim="h1",
    )
    assert len(rows) == 2

    summary = summarize_by_baseline(rows=rows, metrics=["final_return"])
    assert "CCGO_adaptive" in summary
    assert "final_return" in summary["CCGO_adaptive"]

    symbolic = run_checks(output_dir=tmp_path)
    assert len(symbolic) >= 1
