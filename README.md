# IMC Prosperity 3 Optimizer
[![Publish to PyPI](https://github.com/arJ-V/imc-optimal/actions/workflows/publish.yml/badge.svg)](https://github.com/arJ-V/imc-optimal/actions/workflows/publish.yml)
[![CI](https://github.com/arJ-V/imc-optimal/actions/workflows/ci.yml/badge.svg)](https://github.com/arJ-V/imc-optimal/actions/workflows/ci.yml)

Hyperparameter optimizer for [IMC Prosperity 3](https://prosperity.imc.com/) algorithms with multi-objective optimization support.

> **Note**: This project is based on the original [imc-prosperity-3-optimizer](https://github.com/jmerle/imc-prosperity-3-optimizer) by [Jasper van Merle (jmerle)](https://github.com/jmerle). The original project provided the foundation for this version.

## Features

- **Multi-objective optimization (default)**: Uses NSGA-II to optimize PnL, Sharpe ratio, and drawdown simultaneously
- **Single-objective mode**: Optional PnL-only optimization using TPE
- **Live submission mode**: Submit to the real IMC Prosperity platform via [imc-prospector](https://github.com/arJ-V/imc-prospector) to evaluate against the actual matching engine
- **Post-optimization analysis**: Sensitivity analysis, knife-edge detection, and visualizations
- **Overfitting protection**: Default trial limits (50-80 for multi-objective, 30-50 for single-objective)

## Installation

```bash
pip install prosperity3opt
```

To enable live submission mode (submitting to the real IMC platform):

```bash
pip install prosperity3opt[live]
# or separately:
pip install imc-prospector
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

### Live Submission Mode

Use `--live` to optimize against the real IMC Prosperity matching engine instead of the local backtester. This is useful when the local backtester's matching engine doesn't perfectly replicate the real platform behavior.

```bash
# Optimize using real platform submissions
prosperity3opt algorithm.py --live

# Customize rate limiting and timeout
prosperity3opt algorithm.py --live --live-delay 15 --live-timeout 600

# More trials on the real platform
prosperity3opt algorithm.py --live --trials 20
```

Live mode automatically:
- Forces sequential submissions (`--jobs 1`)
- Defaults to PnL-only optimization (`--pnl-only`)
- Uses 10 trials by default (each submission takes time on the platform)

**Prerequisites**: Your Prosperity ID token must be configured. Run `imc-prospector submit` once to set it up, or the optimizer will prompt you on first use. See the [imc-prospector README](https://github.com/arJ-V/imc-prospector#first-time-setup) for details on getting your token.

### Options

| Option | Description |
|--------|-------------|
| `--pnl-only` | Use single-objective optimization (TPE) instead of multi-objective (NSGA-II) |
| `--trials N` | Maximum number of trials (default: 65 multi-obj, 40 pnl-only, 10 live) |
| `--out PATH` | Path to save optimization results (default: `prosperity3opt.log`) |
| `--no-out` | Skip saving optimization results |
| `--grid` | Use grid search instead of TPE/NSGA-II |
| `--jobs N` | Number of parallel backtests (default: -1, uses all CPU cores) |
| `--min` | Minimize total profit instead of maximizing |
| `--match-trades` | How to match orders against market trades (`all`, `worse`, `none`) |
| `--live` | Submit to the real IMC Prosperity platform instead of local backtesting |
| `--live-delay N` | Minimum seconds between live submissions (default: 10) |
| `--live-timeout N` | Maximum seconds to wait per live submission (default: 300) |

Run `prosperity3opt --help` for all available options.

## Design Philosophy

### Multi-Objective Optimization (Default)

The default mode uses NSGA-II to optimize three objectives simultaneously:
1. **PnL**: Maximize total profit
2. **Sharpe Ratio**: Maximize risk-adjusted returns
3. **Drawdown**: Minimize maximum drawdown

This returns a Pareto front of solutions, allowing you to choose the highest PnL candidate with acceptable Sharpe ratio and drawdown.

### Local vs. Live Evaluation

The optimizer supports two evaluation backends:

- **Local** (default): Runs backtests via `prosperity3bt`. Fast, parallel, supports all objectives. Best for exploring the parameter space.
- **Live** (`--live`): Submits to the real IMC Prosperity platform via `imc-prospector`. Slower but uses the actual matching engine, which can differ from the local backtester. Best for validating top candidates or when local matching accuracy matters.

A recommended workflow is to optimize locally first, then re-run the top parameter sets with `--live` to validate against the real platform.

### Trial Limits

To prevent overfitting on limited backtest data:
- Multi-objective: 50-80 trials (default: 65)
- Single-objective: 30-50 trials (default: 40)
- Live mode: 10 trials (default, since each submission is slower)

### Post-Optimization Analysis

After optimization, the tool provides:
- **Pareto front visualization**: Shows tradeoffs between objectives
- **Parameter importance**: Identifies which parameters matter most
- **Sensitivity analysis**: Flags parameters on "knife edges" (unstable performance)
- **Optimization history**: Tracks PnL progression over trials

## How It Works

The optimizer:
1. Scans your algorithm file for `# opt:` annotations
2. Runs backtests using `prosperity3bt` (local mode) or submits to the real IMC platform (live mode) with different parameter combinations
3. Optimizes parameters using Optuna (NSGA-II for multi-objective, TPE for single-objective)
4. Returns a Pareto front (multi-objective) or best solution (single-objective)

In **live mode**, the optimizer creates a self-contained algorithm file for each trial with the parameter values hardcoded directly into the source, then submits it via `imc-prospector` and extracts PnL from the platform's results.

## Visualization

After optimization, you can use the Optuna Dashboard to visualize results:

```bash
optuna-dashboard prosperity3opt.log
```

## Credits

This project is based on the original [imc-prosperity-3-optimizer](https://github.com/jmerle/imc-prosperity-3-optimizer) by [Jasper van Merle (jmerle)](https://github.com/jmerle). The original project provided the core optimizer functionality.

Live submission support is powered by [imc-prospector](https://github.com/arJ-V/imc-prospector).

## License

MIT License
