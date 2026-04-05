from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .mcp_adapter import call_tool, list_tools


def _tool_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.input_schema,
        }
        for spec in list_tools()
    ]


def _response(req_id: Any, result: Any = None, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        payload["error"] = {"code": -32000, "message": error}
    else:
        payload["result"] = result
    return payload


def _handle_request(request: dict[str, Any]) -> dict[str, Any]:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    try:
        if method == "initialize":
            return _response(
                req_id,
                {
                    "serverInfo": {"name": "ccgo-rl", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            )
        if method == "tools/list":
            return _response(req_id, {"tools": _tool_payload()})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str):
                raise ValueError("tools/call requires string field 'name'")
            result = call_tool(name=name, arguments=arguments)
            return _response(req_id, {"content": [{"type": "json", "json": result}]})

        raise ValueError(f"Unsupported method: {method}")
    except Exception as exc:  # pylint: disable=broad-except
        return _response(req_id, error=str(exc))


def run_stdio_server() -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            error_payload = _response(None, error=f"Invalid JSON request: {exc}")
            sys.stdout.write(json.dumps(error_payload) + "\n")
            sys.stdout.flush()
            continue

        if isinstance(request, dict) and request.get("method") == "exit":
            return 0

        response = _handle_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


def _run_once(tool: str, args_json: str) -> int:
    arguments = json.loads(args_json)
    result = call_tool(tool, arguments)
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CCGO-RL MCP companion server over stdio. "
            "Tools expose benchmark simulation and symbolic verification APIs."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "once"],
        default="stdio",
        help="Run a persistent stdio server or execute one tool call.",
    )
    parser.add_argument(
        "--tool",
        default="simulate_benchmark",
        help="Tool name for --mode once.",
    )
    parser.add_argument(
        "--args-json",
        default="{}",
        help="JSON object string with tool arguments for --mode once.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print server version and exit.",
    )
    args = parser.parse_args()

    if args.version:
        sys.stdout.write("ccgo-rl mcp server 0.1.0\n")
        return 0

    if args.mode == "once":
        return _run_once(tool=args.tool, args_json=args.args_json)

    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
