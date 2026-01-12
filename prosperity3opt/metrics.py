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

        # Sharpe ratio: mean return / std deviation
        # Using simple Sharpe without risk-free rate (assumed to be 0)
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

        # Return as percentage of peak
        max_dd = np.max(drawdown)
        peak = running_max[-1] if running_max[-1] != 0 else 1.0
        return (max_dd / peak) * 100.0


def parse_backtester_output(stdout: str) -> BacktestMetrics:
    """Parse backtester output to extract metrics."""
    day_profits = []
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    lines = stdout.splitlines()
    for line in lines:
        # Extract day profits
        if line.startswith("Total profit: "):
            profit_str = line.split(": ")[1].replace(",", "")
            try:
                day_profits.append(int(profit_str))
            except ValueError:
                pass

        # Try to extract Sharpe ratio if present
        if "sharpe" in line.lower() and sharpe_ratio is None:
            parts = line.split()
            for i, part in enumerate(parts):
                if "sharpe" in part.lower() and i + 1 < len(parts):
                    try:
                        sharpe_ratio = float(parts[i + 1])
                        break
                    except (ValueError, IndexError):
                        pass

        # Try to extract drawdown if present
        if ("drawdown" in line.lower() or "dd" in line.lower()) and max_drawdown is None:
            parts = line.split()
            for i, part in enumerate(parts):
                if ("drawdown" in part.lower() or "dd" in part.lower()) and i + 1 < len(parts):
                    try:
                        max_drawdown = float(parts[i + 1])
                        break
                    except (ValueError, IndexError):
                        pass

        # Stop parsing after profit summary
        if line == "Profit summary:":
            break

    total_pnl = sum(day_profits) if day_profits else 0

    metrics = BacktestMetrics(
        day_profits=day_profits,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )

    # Calculate metrics if not provided by backtester
    if metrics.sharpe_ratio is None:
        metrics.sharpe_ratio = metrics.calculate_sharpe_ratio()

    if metrics.max_drawdown is None:
        metrics.max_drawdown = metrics.calculate_max_drawdown()

    return metrics
