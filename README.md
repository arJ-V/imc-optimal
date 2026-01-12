# IMC Prosperity 3 Optimizer
[![Publish to PyPI](https://github.com/arJ-V/imc-optimal/actions/workflows/publish.yml/badge.svg)](https://github.com/arJ-V/imc-optimal/actions/workflows/publish.yml)
[![CI](https://github.com/arJ-V/imc-optimal/actions/workflows/ci.yml/badge.svg)](https://github.com/arJ-V/imc-optimal/actions/workflows/ci.yml)

Hyperparameter optimizer for [IMC Prosperity 3](https://prosperity.imc.com/) algorithms with multi-objective optimization support.

> **Note**: This project is based on the original [imc-prosperity-3-optimizer](https://github.com/jmerle/imc-prosperity-3-optimizer) by [Jasper van Merle (jmerle)](https://github.com/jmerle). The original project provided the foundation for this version.

## Features

- **Multi-objective optimization (default)**: Uses NSGA-II to optimize PnL, Sharpe ratio, and drawdown simultaneously
- **Single-objective mode**: Optional PnL-only optimization using TPE
- **Post-optimization analysis**: Sensitivity analysis, knife-edge detection, and visualizations
- **Overfitting protection**: Default trial limits (50-80 for multi-objective, 30-50 for single-objective)

## Installation

```bash
pip install prosperity3opt
```

## Usage

### Basic Usage

Hyperparameters that need to be optimized must be annotated in your code like this:

```python
RAINFOREST_RESIN_VALUE = 10_000 # opt: int(10_000 - 3, 10_000 + 3)
```

You can use any of [Optuna's `trial.suggest_*` methods](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial) after the "# opt: " comment:
- `# opt: categorical(choices)` - suggest a value for a categorical parameter
- `# opt: float(low, high, *[, step, log])` - suggest a value for a floating point parameter
- `# opt: int(low, high, *[, step, log])` - suggest a value for an integer parameter

### Running the Optimizer

**Multi-objective mode (default)** - Optimizes PnL, Sharpe ratio, and drawdown:
```bash
# Optimize on all days from round 1
prosperity3opt algorithm.py 1

# Optimize on specific days
prosperity3opt algorithm.py 1-0 1--1
```

**Single-objective mode (PnL-only)** - Optimizes only PnL:
```bash
prosperity3opt algorithm.py 1 --pnl-only
```

### Options

- `--pnl-only`: Use single-objective optimization (TPE) instead of multi-objective (NSGA-II)
- `--trials N`: Maximum number of trials (default: 65 for multi-objective, 40 for single-objective)
- `--out PATH`: Path to save optimization results (default: `prosperity3opt.log`)
- `--grid`: Use grid search instead of TPE/NSGA-II
- `--jobs N`: Number of parallel backtests (default: -1, uses all CPU cores)

Run `prosperity3opt --help` for all available options.

## Design Philosophy

### Multi-Objective Optimization (Default)

The default mode uses NSGA-II to optimize three objectives simultaneously:
1. **PnL**: Maximize total profit
2. **Sharpe Ratio**: Maximize risk-adjusted returns
3. **Drawdown**: Minimize maximum drawdown

This returns a Pareto front of solutions, allowing you to choose the highest PnL candidate with acceptable Sharpe ratio and drawdown.

### Trial Limits

To prevent overfitting on limited backtest data:
- Multi-objective: 50-80 trials (default: 65)
- Single-objective: 30-50 trials (default: 40)

### Post-Optimization Analysis

After optimization, the tool provides:
- **Pareto front visualization**: Shows tradeoffs between objectives
- **Parameter importance**: Identifies which parameters matter most
- **Sensitivity analysis**: Flags parameters on "knife edges" (unstable performance)
- **Optimization history**: Tracks PnL progression over trials

## How It Works

The optimizer:
1. Scans your algorithm file for `# opt:` annotations
2. Runs backtests using the `prosperity3bt` module with different parameter combinations
3. Optimizes parameters using Optuna (NSGA-II for multi-objective, TPE for single-objective)
4. Returns a Pareto front (multi-objective) or best solution (single-objective)

**Note**: The optimizer uses the `prosperity3bt` module to access historical data and run backtests. You don't need to upload day-specific code - the backtester handles data access automatically.

## Visualization

After optimization, you can use the Optuna Dashboard to visualize results:

```bash
optuna-dashboard prosperity3opt.log
```

## Credits

This project is based on the original [imc-prosperity-3-optimizer](https://github.com/jmerle/imc-prosperity-3-optimizer) by [Jasper van Merle (jmerle)](https://github.com/jmerle). The original project provided the core optimizer functionality.

## License

MIT License
