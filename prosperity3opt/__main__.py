# import concurrent.futures as f

# f.ThreadPoolExecutor = f.ProcessPoolExecutor

import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated, Generator, Optional

import optuna
from optuna.samplers import BaseSampler, GridSampler, NSGAIISampler, TPESampler
from optuna.storages import BaseStorage, InMemoryStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend
from prosperity3bt.models import TradeMatchingMode
from typer import Argument, Option, Typer

from prosperity3opt.grid import get_grid_search_space
from prosperity3opt.objective import ObjectiveRunner


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())

    try:
        yield path
    finally:
        shutil.rmtree(path)


def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity3opt {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm to optimize.", show_default=False, exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    days: Annotated[list[str], Argument(help="Days to backtest on (e.g., '1-1', '2'). Required for local mode, not used with --live.", show_default=False)] = [],
    out: Annotated[Path, Option(help="Path to save optimization results to.", show_default=False, dir_okay=False, resolve_path=True)] = Path("prosperity3opt.log"),
    no_out: Annotated[bool, Option("--no-out", help="Skip saving optimization results.")] = False,
    jobs: Annotated[int, Option(help="Number of backtests to run in parallel (-1 to use number of CPU cores).")] = -1,
    minimize: Annotated[bool, Option("--min", help="Minimize the total profit rather than maximizing it.")] = False,
    trials: Annotated[Optional[int], Option(help="Maximum number of trials to run. Defaults: 65 multi-obj, 40 pnl-only, 10 live.", show_default=False)] = None,
    seconds: Annotated[Optional[int], Option(help="Maximum number of seconds to run for (defaults to infinity).", show_default=False)] = None,
    grid: Annotated[bool, Option("--grid", help="Perform a grid search. Requires all floating point parameters to have a step size.")] = False,
    pnl_only: Annotated[bool, Option("--pnl-only", help="Optimize PnL only using TPE (single-objective mode). Default is multi-objective with NSGA-II.")] = False,
    match_trades: Annotated[TradeMatchingMode, Option(help="How to match orders against market trades. 'all' matches trades with prices equal to or worse than your quotes, 'worse' matches trades with prices worse than your quotes, 'none' does not match trades against orders at all.")] = TradeMatchingMode.all,
    live: Annotated[bool, Option("--live", help="Submit to the real IMC Prosperity platform instead of local backtesting. Requires imc-prospector (pip install imc-prospector).")] = False,
    live_delay: Annotated[int, Option("--live-delay", help="Minimum seconds between live submissions to avoid rate limiting.")] = 10,
    live_timeout: Annotated[int, Option("--live-timeout", help="Maximum seconds to wait for each live submission to complete.")] = 300,
    version: Annotated[bool, Option("--version", "-v", help="Show the program's version number and exit.", is_eager=True, callback=version_callback)] = False,
) -> None:  # fmt: skip
    """
    Optimize an IMC Prosperity 3 algorithm using Optuna and prosperity3bt.

    By default, uses multi-objective optimization (NSGA-II) optimizing for PnL, Sharpe ratio, and drawdown.
    Use --pnl-only to optimize only PnL using TPE (single-objective mode).
    Use --live to submit algorithms to the real IMC platform instead of local backtesting.
    """
    if out is not None and no_out:
        print("Error: --out and --no-out are mutually exclusive")
        sys.exit(1)

    if live:
        if not days:
            days = []
        if jobs != 1:
            print("Note: --live mode requires sequential submissions, using --jobs 1")
            jobs = 1
        if not pnl_only and not grid:
            print(
                "Note: --live mode defaults to --pnl-only since Sharpe/drawdown from "
                "the platform are derived from per-product PnL, not per-day."
            )
            pnl_only = True
        print(
            "=== LIVE SUBMISSION MODE ===\n"
            "Algorithms will be submitted to the real IMC Prosperity platform.\n"
            "Ensure your Prosperity ID token is configured (imc-prospector will prompt if needed).\n"
            f"Delay between submissions: {live_delay}s | Timeout per submission: {live_timeout}s"
        )
    else:
        if not days:
            print("Error: at least one day argument is required for local backtest mode")
            print("Usage: prosperity3opt <algorithm.py> <days...> [OPTIONS]")
            sys.exit(1)

    backtester_args = ["--match-trades", match_trades.value]
    multi_objective = not pnl_only

    with temporary_directory() as temp_dir:
        runner = ObjectiveRunner(
            temp_dir,
            algorithm,
            days,
            backtester_args,
            multi_objective=multi_objective,
            use_live=live,
            live_delay=live_delay,
            live_timeout=live_timeout,
        )

        if len(runner.params) == 0:
            print("Error: no hyperparameters found")
            sys.exit(1)

        storage: BaseStorage
        if no_out:
            storage = InMemoryStorage()
        else:
            storage = JournalStorage(JournalFileBackend(str(out)))

        sampler: BaseSampler
        if grid:
            search_space = get_grid_search_space(runner)
            sampler = GridSampler(search_space)

            num_combinations = 1
            for v in search_space.values():
                num_combinations *= len(v)

            print(f"Running grid search on {num_combinations:,.0f} possible hyperparameter combinations")
            if live:
                est_time = num_combinations * (live_delay + 60)
                print(f"Estimated minimum time for live grid search: ~{est_time // 60} minutes")
        elif multi_objective:
            sampler = NSGAIISampler()
            if trials is None:
                trials = 65
            print("Using multi-objective optimization (NSGA-II): optimizing PnL, Sharpe ratio, and drawdown")
        else:
            sampler = TPESampler()
            if trials is None:
                trials = 10 if live else 40
            mode_label = "live platform" if live else "local backtester"
            print(f"Using single-objective optimization (TPE) via {mode_label}: optimizing PnL only")

        if multi_objective:
            study = optuna.create_study(
                storage=storage,
                sampler=sampler,
                directions=["maximize", "maximize", "minimize"],
                study_name=datetime.now().strftime("prosperity3opt_%Y-%m-%d_%H-%M-%S"),
            )
        else:
            study = optuna.create_study(
                storage=storage,
                sampler=sampler,
                direction="minimize" if minimize else "maximize",
                study_name=datetime.now().strftime("prosperity3opt_%Y-%m-%d_%H-%M-%S"),
            )

        try:
            study.optimize(runner.objective, n_jobs=jobs, n_trials=trials, timeout=seconds)
        except KeyboardInterrupt:
            print("Stopping optimization...")
        finally:
            try:
                if multi_objective:
                    print("\nOptimization complete!")
                    print(f"Total trials: {len(study.trials)}")
                    print(f"\nPareto front contains {len(study.best_trials)} solutions")
                    print("\nTop Pareto front solutions:")
                    for i, trial in enumerate(study.best_trials[:10], 1):  # Show top 10
                        values = trial.values
                        print(f"\n{i}. PnL: {values[0]:,.0f}, Sharpe: {values[1]:.4f}, Drawdown: {values[2]:.4f}%")
                        print(f"   Parameters: {trial.params}")
                else:
                    print(f"Best profit: {study.best_value:,.0f}")
                    print(f"Best parameters: {study.best_params}")
            except Exception as e:
                print(f"Error displaying results: {e}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
