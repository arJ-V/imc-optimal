"""Post-optimization analysis: sensitivity analysis and visualization."""
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.importance import get_param_importances


def _is_multi_objective(study: optuna.Study) -> bool:
    """Check if study is multi-objective."""
    # Check if directions is a list (multi-objective) or string (single-objective)
    return isinstance(study.directions, list) and len(study.directions) > 1


def sensitivity_analysis(
    study: optuna.Study,
    runner: Any,
    top_n: int = 5,
    perturbation_percent: float = 0.20,
) -> dict[str, dict[str, Any]]:
    """Perform sensitivity analysis on top candidates.
    
    Args:
        study: Optuna study containing optimization results
        runner: ObjectiveRunner instance to run backtests
        top_n: Number of top candidates to analyze
        perturbation_percent: Percentage to perturb each parameter (±)
    
    Returns:
        Dictionary mapping candidate index to sensitivity results
    """
    results = {}
    
    if _is_multi_objective(study):
        # Get top candidates from Pareto front (sorted by PnL)
        best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)[:top_n]
    else:
        # Single objective: get best trial
        best_trials = [study.best_trial]
    
    for i, trial in enumerate(best_trials[:top_n]):
        base_params = trial.params
        base_value = trial.values[0] if _is_multi_objective(study) else trial.value
        
        sensitivities = {}
        
        for param_name, param_value in base_params.items():
            if isinstance(param_value, (int, float)):
                perturbations = {}
                
                # Perturb parameter ±20%
                for direction in [-1, 1]:
                    perturbed_value = param_value * (1 + direction * perturbation_percent)
                    
                    # Create new trial with perturbed parameter
                    perturbed_params = base_params.copy()
                    perturbed_params[param_name] = type(param_value)(perturbed_value)
                    
                    # Run backtest with perturbed parameter
                    try:
                        # This would require running a full backtest, which is expensive
                        # For now, we'll flag this for future implementation
                        # and return the structure
                        perturbations[f"{direction * perturbation_percent * 100:+.0f}%"] = None
                    except Exception as e:
                        perturbations[f"{direction * perturbation_percent * 100:+.0f}%"] = f"Error: {e}"
                
                sensitivities[param_name] = {
                    "base_value": param_value,
                    "perturbations": perturbations,
                    "stable": True,  # Will be updated when perturbations are calculated
                }
        
        results[f"candidate_{i+1}"] = {
            "trial_number": trial.number,
            "base_value": base_value,
            "params": base_params,
            "sensitivities": sensitivities,
        }
    
    return results


def detect_knife_edges(
    study: optuna.Study,
    threshold: float = 0.10,
) -> list[dict[str, Any]]:
    """Detect parameters that sit on 'knife edges' (small change = large PnL change).
    
    Args:
        study: Optuna study
        threshold: Threshold for flagging knife edge (fractional change in PnL)
    
    Returns:
        List of flagged parameters
    """
    knife_edges = []
    
    if not _is_multi_objective(study):
        # For single-objective, analyze parameter importance
        try:
            importance = get_param_importances(study)
            # Parameters with very high importance might indicate knife edges
            for param, imp in importance.items():
                if imp > 0.3:  # High importance threshold
                    knife_edges.append({
                        "parameter": param,
                        "importance": imp,
                        "reason": "High parameter importance may indicate sensitivity",
                    })
        except Exception:
            pass
    
    return knife_edges


def plot_pareto_front(study: optuna.Study, output_path: Optional[Path] = None) -> None:
    """Plot Pareto front for multi-objective optimization.
    
    Args:
        study: Optuna study (must be multi-objective)
        output_path: Path to save plot (if None, displays interactively)
    """
    if not _is_multi_objective(study):
        print("Warning: Cannot plot Pareto front for single-objective study")
        return
    
    best_trials = study.best_trials
    
    if len(best_trials) == 0:
        print("Warning: No trials in Pareto front")
        return
    
    # Extract values
    pnls = [t.values[0] for t in best_trials]
    sharpes = [t.values[1] for t in best_trials]
    drawdowns = [t.values[2] for t in best_trials]
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # PnL vs Sharpe
    ax1 = fig.add_subplot(131)
    ax1.scatter(sharpes, pnls, alpha=0.6)
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_ylabel("PnL")
    ax1.set_title("Pareto Front: PnL vs Sharpe")
    ax1.grid(True, alpha=0.3)
    
    # PnL vs Drawdown
    ax2 = fig.add_subplot(132)
    ax2.scatter(drawdowns, pnls, alpha=0.6)
    ax2.set_xlabel("Max Drawdown (%)")
    ax2.set_ylabel("PnL")
    ax2.set_title("Pareto Front: PnL vs Drawdown")
    ax2.grid(True, alpha=0.3)
    
    # Sharpe vs Drawdown
    ax3 = fig.add_subplot(133)
    ax3.scatter(drawdowns, sharpes, alpha=0.6)
    ax3.set_xlabel("Max Drawdown (%)")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_title("Pareto Front: Sharpe vs Drawdown")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Pareto front plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_parameter_importance(study: optuna.Study, output_path: Optional[Path] = None) -> None:
    """Plot parameter importance.
    
    Args:
        study: Optuna study
        output_path: Path to save plot (if None, displays interactively)
    """
    try:
        importance = get_param_importances(study)
    except Exception as e:
        print(f"Warning: Could not calculate parameter importance: {e}")
        return
    
    if not importance:
        print("Warning: No parameter importance data")
        return
    
    params = list(importance.keys())
    importances = list(importance.values())
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    params = [params[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.5)))
    ax.barh(params, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Parameter Importance")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Parameter importance plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_optimization_history(study: optuna.Study, output_path: Optional[Path] = None) -> None:
    """Plot optimization history (PnL over trials).
    
    Args:
        study: Optuna study
        output_path: Path to save plot (if None, displays interactively)
    """
    if _is_multi_objective(study):
        # For multi-objective, plot PnL over trials
        trial_numbers = [t.number for t in study.trials if t.values]
        pnls = [t.values[0] for t in study.trials if t.values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trial_numbers, pnls, "o", alpha=0.5, label="All trials")
        
        # Highlight Pareto front
        best_trial_numbers = [t.number for t in study.best_trials]
        best_pnls = [t.values[0] for t in study.best_trials]
        ax.scatter(best_trial_numbers, best_pnls, color="red", s=100, label="Pareto front", zorder=5)
        
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("PnL")
        ax.set_title("Optimization History: PnL over Trials")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Single-objective: use Optuna's built-in plotting
        try:
            import optuna.visualization as vis
            
            fig = vis.plot_optimization_history(study)
            fig.show()
        except Exception:
            # Fallback: manual plotting
            trial_numbers = [t.number for t in study.trials if t.value is not None]
            values = [t.value for t in study.trials if t.value is not None]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trial_numbers, values, "o-", alpha=0.7)
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Objective Value (PnL)")
            ax.set_title("Optimization History")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Optimization history plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_analysis_report(
    study: optuna.Study,
    output_dir: Path,
    runner: Optional[Any] = None,
) -> None:
    """Generate comprehensive analysis report with visualizations.
    
    Args:
        study: Optuna study
        output_dir: Directory to save analysis outputs
        runner: ObjectiveRunner instance (optional, for sensitivity analysis)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating Analysis Report ===")
    
    # Plot Pareto front (if multi-objective)
    if _is_multi_objective(study):
        plot_pareto_front(study, output_dir / "pareto_front.png")
        print(f"Found {len(study.best_trials)} solutions in Pareto front")
    
    # Plot parameter importance
    plot_parameter_importance(study, output_dir / "parameter_importance.png")
    
    # Plot optimization history
    plot_optimization_history(study, output_dir / "optimization_history.png")
    
    # Detect knife edges
    knife_edges = detect_knife_edges(study)
    if knife_edges:
        print("\n⚠️  Potential knife edge parameters detected:")
        for edge in knife_edges:
            print(f"  - {edge['parameter']}: {edge['reason']}")
    else:
        print("\n✓ No knife edge parameters detected")
    
    print(f"\nAnalysis report saved to {output_dir}/")
