"""Metrics extraction from backtester output."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BacktestMetrics:
    """Metrics extracted from a backtest run."""

    day_profits: list[int]
    total_pnl: int
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from day profits."""
        if len(self.day_profits) < 2:
            return 0.0

        profits_array = np.array(self.day_profits, dtype=float)
        mean_return = np.mean(profits_array)
        std_return = np.std(profits_array)

        if std_return == 0:
            return 0.0

        return mean_return / std_return

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from day profits."""
        if len(self.day_profits) == 0:
            return 0.0

        profits_array = np.array(self.day_profits, dtype=float)
        cumulative = np.cumsum(profits_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative

        if len(drawdown) == 0 or running_max[-1] == 0:
            return 0.0

        max_dd = np.max(drawdown)
        peak = running_max[-1] if running_max[-1] != 0 else 1.0
        return (max_dd / peak) * 100.0


def parse_backtester_output(stdout: str) -> BacktestMetrics:
    """Parse backtester output to extract metrics.

    Handles both P3 and P4 output formats:
      P3: "Total profit: N" lines before "Profit summary:"
      P4: same structure plus "Risk metrics:" section with key-value pairs
    """
    day_profits = []
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    in_profit_summary = False
    in_risk_metrics = False

    lines = stdout.splitlines()
    for line in lines:
        stripped = line.strip()

        if stripped == "Profit summary:":
            in_profit_summary = True
            continue

        if stripped.startswith("Risk metrics"):
            in_profit_summary = False
            in_risk_metrics = True
            continue

        if not in_profit_summary and not in_risk_metrics:
            # Before profit summary: collect per-day "Total profit:" lines
            if stripped.startswith("Total profit:"):
                profit_str = stripped.split(":")[1].strip().replace(",", "")
                try:
                    day_profits.append(int(float(profit_str)))
                except ValueError:
                    pass

        if in_risk_metrics:
            # P4 risk metrics: "  key: value" format
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip().lower()
                val = val.strip()

                if key == "sharpe_ratio" and val not in ("n/a", ""):
                    try:
                        sharpe_ratio = float(val)
                    except ValueError:
                        pass
                elif key == "max_drawdown_abs" and val not in ("n/a", ""):
                    try:
                        max_drawdown = float(val)
                    except ValueError:
                        pass
            elif stripped == "":
                in_risk_metrics = False

    total_pnl = sum(day_profits) if day_profits else 0

    metrics = BacktestMetrics(
        day_profits=day_profits,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )

    if metrics.sharpe_ratio is None:
        metrics.sharpe_ratio = metrics.calculate_sharpe_ratio()

    if metrics.max_drawdown is None:
        metrics.max_drawdown = metrics.calculate_max_drawdown()

    return metrics
