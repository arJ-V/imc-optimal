"""Live submission evaluation via imc-prospector.

Wraps the imc-prospector package API for use in the optimization loop,
submitting algorithm files to the real IMC Prosperity platform and
extracting PnL metrics from the results.
"""

import json
import time
from pathlib import Path
from typing import Any

from prosperity3opt.metrics import BacktestMetrics


def check_prospector_available() -> None:
    """Verify imc-prospector is installed and importable."""
    try:
        import imcprospector  # noqa: F401
    except ImportError:
        raise ImportError(
            "imc-prospector is required for live submission mode.\n"
            "Install it with: pip install imc-prospector"
        ) from None


def get_round_id() -> int:
    """Get the current open round ID from the Prosperity platform."""
    from imcprospector.submit import get_current_round

    return get_current_round()


def submit_and_evaluate(
    algorithm_file: Path,
    output_dir: Path,
    round_id: int,
    timeout: int = 300,
) -> BacktestMetrics:
    """Submit an algorithm to the real IMC platform and return metrics.

    Uploads the algorithm file, polls until the submission completes,
    downloads the graph JSON, and extracts PnL into BacktestMetrics.
    """
    from imcprospector.submit import download_graph, submit_algorithm

    submit_algorithm(algorithm_file)
    data = _monitor_submission(round_id, algorithm_file, timeout)

    if data["status"] in ("ERROR", "ERROR_FINISHED"):
        raise RuntimeError(f"Live submission failed with status: {data['status']}")

    output_file = output_dir / f"submission_{data['id']}.json"
    download_graph(data, output_file)

    return _parse_submission_results(output_file)


def _monitor_submission(
    round_id: int,
    algorithm_file: Path,
    timeout: int,
) -> dict[str, Any]:
    """Poll submission status until completion or timeout."""
    from imcprospector.submit import list_algorithms

    algorithms = list_algorithms(round_id)
    data = next(
        (a for a in algorithms if a["filename"].endswith(algorithm_file.name)),
        None,
    )

    if data is None:
        raise RuntimeError(f"Could not find submission for {algorithm_file.name}")

    start = time.monotonic()

    while data["status"] not in ("FINISHED", "ERROR", "ERROR_FINISHED"):
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Submission timed out after {timeout}s (last status: {data['status']})"
            )

        time.sleep(0 if data.get("active", False) else 5)
        algorithms = list_algorithms(round_id)
        data = next(a for a in algorithms if a["id"] == data["id"])

    return data


def _parse_submission_results(output_file: Path) -> BacktestMetrics:
    """Parse submission graph JSON into BacktestMetrics.

    The graph data is a JSON array of {timestamp, value} objects representing
    the cumulative PnL curve over time. The final entry's value is total PnL.
    """
    graph_data = json.loads(output_file.read_text(encoding="utf-8"))

    total_pnl = 0
    day_profits: list[int] = []

    if isinstance(graph_data, list) and len(graph_data) > 0:
        # Time series: [{timestamp, value}, ...] — cumulative PnL curve
        # Extract incremental profits between points for Sharpe/drawdown
        values = [point["value"] for point in graph_data if "value" in point]
        if values:
            total_pnl = int(values[-1])
            # Chunk into ~equal segments to approximate per-day profits
            chunk_size = max(1, len(values) // 5)
            for i in range(0, len(values), chunk_size):
                chunk = values[i : i + chunk_size]
                start = values[i - 1] if i > 0 else 0.0
                day_profits.append(int(chunk[-1] - start))

    elif isinstance(graph_data, dict) and "profitLoss" in graph_data:
        pl_data = graph_data["profitLoss"]
        if isinstance(pl_data, dict):
            day_profits = [int(v) for v in pl_data.values()]
            total_pnl = sum(day_profits)
        elif isinstance(pl_data, (int, float)):
            total_pnl = int(pl_data)
            day_profits = [total_pnl]

    if not day_profits:
        raise RuntimeError(f"Could not extract PnL from submission results: {output_file}")

    metrics = BacktestMetrics(day_profits=day_profits, total_pnl=total_pnl)
    metrics.sharpe_ratio = metrics.calculate_sharpe_ratio()
    metrics.max_drawdown = metrics.calculate_max_drawdown()

    return metrics
