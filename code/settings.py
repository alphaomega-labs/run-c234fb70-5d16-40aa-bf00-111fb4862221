from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSettings:
    experiment_id: str
    output_dir: Path
    paper_figures_dir: Path
    paper_tables_dir: Path
    paper_data_dir: Path
    paper_appendix_dir: Path
    seeds: list[int]
    baselines: list[str]
    datasets_h1: list[str]
    datasets_h2: list[str]
    datasets_transfer: list[str]


DEFAULT_BASELINES = [
    "CCGO_adaptive",
    "HER_SAC",
    "RND_fixed_beta",
    "ICM_fixed_beta",
    "ACWI_style",
    "Contrastive_always_on",
]


def load_settings(config_path: Path, workspace_root: Path) -> RunSettings:
    cfg: dict[str, Any] = json.loads(config_path.read_text())
    output_dir = workspace_root / cfg["output_dir"]
    return RunSettings(
        experiment_id=cfg["experiment_id"],
        output_dir=output_dir,
        paper_figures_dir=workspace_root / cfg["paper_figures_dir"],
        paper_tables_dir=workspace_root / cfg["paper_tables_dir"],
        paper_data_dir=workspace_root / cfg["paper_data_dir"],
        paper_appendix_dir=workspace_root / cfg["paper_appendix_dir"],
        seeds=cfg["seeds"],
        baselines=cfg.get("baselines", DEFAULT_BASELINES),
        datasets_h1=cfg["datasets_h1"],
        datasets_h2=cfg["datasets_h2"],
        datasets_transfer=cfg["datasets_transfer"],
    )
