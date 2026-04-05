from pathlib import Path

from ccgo_validation.settings import load_settings
from ccgo_validation.simulate import simulate_benchmark_rows
from ccgo_validation.sympy_checks import run_checks


def test_settings_load(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        """{
  \"experiment_id\": \"x\",
  \"output_dir\": \"experiments/ccgo_validation\",
  \"paper_figures_dir\": \"paper/figures\",
  \"paper_tables_dir\": \"paper/tables\",
  \"paper_data_dir\": \"paper/data\",
  \"paper_appendix_dir\": \"paper/appendix\",
  \"seeds\": [1, 2],
  \"datasets_h1\": [\"d1\"],
  \"datasets_h2\": [\"d2\"],
  \"datasets_transfer\": [\"d3\"]
}
""",
        encoding="utf-8",
    )
    settings = load_settings(cfg, tmp_path)
    assert settings.experiment_id == "x"
    assert settings.seeds == [1, 2]


def test_simulate_and_sympy(tmp_path: Path) -> None:
    df = simulate_benchmark_rows([11, 23], ["d"], ["CCGO_adaptive", "HER_SAC"], claim="h1")
    assert len(df) == 4
    table, report = run_checks(tmp_path)
    assert len(table) >= 4
    assert report.exists()
