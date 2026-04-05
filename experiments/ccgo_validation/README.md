# CCGO Validation Experiments

## Goal
Minimal reproducible validation package for CCGO-RL hypotheses:
- Bounded-annealing claim family: confidence-bounded scheduling and extrinsic-return envelope diagnostics
- Uncertainty-gating claim family: uncertainty-gated contrastive coupling and critic-stability diagnostics

## Setup
1. Create venv at `experiments/.venv`.
2. Install required packages (see phase `python_packages`).
3. Run the CLI entrypoint from workspace root.

## Commands
- Run experiments:
  `PYTHONPATH=experiments/ccgo_validation/src experiments/.venv/bin/python experiments/ccgo_validation/run_experiments.py --workspace-root .`
- Lint:
  `PYTHONPATH=experiments/ccgo_validation/src experiments/.venv/bin/python -m ruff check experiments/ccgo_validation`
- Tests:
  `PYTHONPATH=experiments/ccgo_validation/src experiments/.venv/bin/python -m pytest experiments/ccgo_validation/tests -q`

## Revision Scope Notes
- Validation evidence in this package is protocol-simulated; full environment-native reruns remain pending.
- Agent57-like and Plan2Explore-like heavyweight comparators were pre-specified but not executed under the current CPU-only envelope.
- The symbolic audit includes one unresolved limit-corollary mismatch; asymptotic interpretation should remain restricted to the audited admissible scheduler regime.

## Outputs
- Synthetic seedwise data: `paper/data/*.csv`
- Manuscript tables: `paper/tables/*.csv`
- Vector figures (PDF): `paper/figures/*.pdf`
- SymPy reports: `experiments/ccgo_validation/results/sympy/`
- Negative result logs: `experiments/ccgo_validation/results/negative/`
- Results summary: `experiments/ccgo_validation/results/results_summary.json`
