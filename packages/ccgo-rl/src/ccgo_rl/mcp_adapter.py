from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .service import CCGOService


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


SERVICE = CCGOService()


def list_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="simulate_benchmark",
            description=(
                "Run the CCGO synthetic benchmark simulator for specified seeds, datasets, and baselines."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "seeds": {"type": "array", "items": {"type": "integer"}},
                    "datasets": {"type": "array", "items": {"type": "string"}},
                    "baselines": {"type": "array", "items": {"type": "string"}},
                    "claim": {"type": "string"},
                },
                "required": ["seeds", "datasets", "baselines", "claim"],
            },
        ),
        ToolSpec(
            name="summarize_baselines",
            description="Summarize benchmark rows by baseline with confidence intervals.",
            input_schema={
                "type": "object",
                "properties": {
                    "rows": {"type": "array"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rows", "metrics"],
            },
        ),
        ToolSpec(
            name="run_symbolic_checks",
            description="Run theorem-aligned C1/C2 symbolic checks and optional report writing.",
            input_schema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": ["string", "null"]},
                },
                "required": [],
            },
        ),
        ToolSpec(
            name="run_validation_bundle",
            description="Run simulation, summarize baselines, and execute symbolic checks in one request.",
            input_schema={
                "type": "object",
                "properties": {
                    "seeds": {"type": "array", "items": {"type": "integer"}},
                    "datasets": {"type": "array", "items": {"type": "string"}},
                    "baselines": {"type": "array", "items": {"type": "string"}},
                    "claim": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "symbolic_output_dir": {"type": ["string", "null"]},
                },
                "required": ["seeds", "datasets", "baselines", "claim", "metrics"],
            },
        ),
    ]


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    args = arguments or {}

    if name == "simulate_benchmark":
        return SERVICE.run_simulation(
            seeds=list(args["seeds"]),
            datasets=list(args["datasets"]),
            baselines=list(args["baselines"]),
            claim=str(args["claim"]),
        )
    if name == "summarize_baselines":
        return SERVICE.summarize(
            rows=list(args["rows"]),
            metrics=list(args["metrics"]),
        )
    if name == "run_symbolic_checks":
        output_dir = args.get("output_dir")
        return SERVICE.run_symbolic_checks(
            output_dir=None if output_dir is None else str(output_dir)
        )
    if name == "run_validation_bundle":
        return SERVICE.run_validation_bundle(
            seeds=list(args["seeds"]),
            datasets=list(args["datasets"]),
            baselines=list(args["baselines"]),
            claim=str(args["claim"]),
            metrics=list(args["metrics"]),
            symbolic_output_dir=args.get("symbolic_output_dir"),
        )

    raise ValueError(f"Unknown tool: {name}")
