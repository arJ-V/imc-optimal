import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from multiprocessing import Lock
from pathlib import Path
from typing import Callable, Generator, Tuple, Union

from optuna import Trial, TrialPruned
from optuna.distributions import CategoricalChoiceType

from prosperity3opt.metrics import BacktestMetrics, parse_backtester_output


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())

    try:
        yield path
    finally:
        shutil.rmtree(path)


class ObjectiveRunner:
    def __init__(
        self,
        temp_dir: Path,
        algorithm_file: Path,
        days: list[str],
        backtester_args: list[str],
        multi_objective: bool = True,
        use_live: bool = False,
        live_delay: int = 10,
        live_timeout: int = 300,
    ) -> None:
        self._temp_dir = temp_dir
        self._days = days
        self._backtester_args = backtester_args
        self._multi_objective = multi_objective
        self._use_live = use_live
        self._live_delay = live_delay
        self._live_timeout = live_timeout

        self.params = dict[str, Callable[[Trial], CategoricalChoiceType]]()
        self.param_definitions = dict[str, str]()
        self._original_lines: list[str] = []
        self._original_algorithm_name = algorithm_file.name

        self._algorithm_file = self._process_algorithm_file(algorithm_file)

        if use_live:
            from prosperity3opt.submission import check_prospector_available, get_round_id

            check_prospector_available()
            self._round_id = get_round_id()
            self._submission_dir = temp_dir / "submissions"
            self._submission_dir.mkdir(exist_ok=True)
            self._last_submission_time = 0.0

        self._params_seen = set[str]()
        self._params_seen_lock = Lock()

    def _process_algorithm_file(self, algorithm_file: Path) -> Path:
        datamodel_path = algorithm_file.parent / "datamodel.py"
        if datamodel_path.exists():
            shutil.copyfile(datamodel_path, self._temp_dir / "datamodel.py")

        with algorithm_file.open("r", encoding="utf-8") as f:
            self._original_lines = f.readlines()

        output_file = self._temp_dir / "algorithm.py"

        header = (
            "import json as prosperity3opt_json\n"
            "import os as prosperity3opt_os\n"
            'prosperity3opt_params = prosperity3opt_json.loads(prosperity3opt_os.environ["PROSPERITY3OPT_PARAMS"])\n'
        )

        # Find insertion point: after any __future__ imports and module docstrings
        insert_after = 0
        in_docstring = False
        docstring_char = None
        for i, line in enumerate(self._original_lines):
            stripped = line.strip()
            if in_docstring:
                if docstring_char and docstring_char in stripped:
                    in_docstring = False
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    continue
                in_docstring = True
                continue
            if stripped.startswith("from __future__"):
                insert_after = i + 1
                continue
            break

        pattern = re.compile(r"^\s*([^ ]+)\s*=\s*(.*?)\s*#\s*opt:\s*((categorical|float|int).*)\s*$")

        with output_file.open("w+", encoding="utf-8") as fout:
            for line in self._original_lines[:insert_after]:
                fout.write(line)
            fout.write("\n")
            fout.write(header)
            fout.write("\n")
            for line in self._original_lines[insert_after:]:
                fout.write(pattern.sub(self._process_opt_match, line))

        return output_file

    def _process_opt_match(self, match: re.Match) -> str:
        var_name = match.group(1)
        param_name = self._get_param_name(var_name)

        param_definition = match.group(3)
        param_definition_with_name = param_definition.replace("(", f'("{param_name}", ', 1)

        self.param_definitions[param_name] = param_definition
        self.params[param_name] = f"trial.suggest_{param_definition_with_name}"

        print(f"Hyperparameter: {var_name=}, {param_name=}, {param_definition=}")

        line = match.group(0)
        value_start = match.start(2)
        value_end = match.end(2)
        new_value = f'prosperity3opt_params["{param_name}"]'

        return line[:value_start] + new_value + line[value_end:]

    def _get_param_name(self, var_name: str) -> str:
        if var_name not in self.params:
            return var_name

        suffix = 2
        while f"{var_name}{suffix}" in self.params:
            suffix += 1

        return f"{var_name}{suffix}"

    def _run_backtest(self, params: str) -> BacktestMetrics:
        """Run a single local backtest and return metrics."""
        env = os.environ.copy()
        env["PROSPERITY3OPT_PARAMS"] = params

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "prosperity4bt",
                "cli",
                str(self._algorithm_file),
                *self._days,
                *self._backtester_args,
                "--no-out",
                "--no-progress",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        stdout = proc.stdout.decode("utf-8")

        if proc.returncode != 0:
            raise RuntimeError(f"prosperity4bt exited with status code {proc.returncode}. Output:\n{stdout}")

        return parse_backtester_output(stdout)

    def _create_live_algorithm(self, params_dict: dict) -> Path:
        """Create an algorithm file with hardcoded parameter values for live submission.

        Replaces each `# opt:` annotated line with the trial's suggested value,
        producing a self-contained file that runs without env-var injection.
        """
        output_file = self._temp_dir / self._original_algorithm_name
        pattern = re.compile(r"^(\s*[^ ]+\s*=\s*)(.*?)(\s*#\s*opt:\s*(categorical|float|int).*)\s*$")

        param_names = list(self.params.keys())
        param_idx = 0

        with output_file.open("w", encoding="utf-8") as fout:
            for line in self._original_lines:
                match = pattern.match(line)
                if match and param_idx < len(param_names):
                    prefix = match.group(1)
                    value = params_dict[param_names[param_idx]]
                    fout.write(f"{prefix}{repr(value)}\n")
                    param_idx += 1
                else:
                    fout.write(line)

        return output_file

    def _run_live_submission(self, params_dict: dict) -> BacktestMetrics:
        """Submit algorithm to the real IMC platform and return metrics."""
        from prosperity3opt.submission import submit_and_evaluate

        elapsed = time.time() - self._last_submission_time
        if elapsed < self._live_delay:
            remaining = self._live_delay - elapsed
            print(f"Rate limiting: waiting {remaining:.0f}s before next submission...")
            time.sleep(remaining)

        algorithm_file = self._create_live_algorithm(params_dict)

        try:
            metrics = submit_and_evaluate(
                algorithm_file,
                self._submission_dir,
                self._round_id,
                timeout=self._live_timeout,
            )
        finally:
            self._last_submission_time = time.time()

        return metrics

    def objective(self, trial: Trial) -> Union[float, Tuple[float, float, float]]:
        """Objective function for optimization.

        Returns:
            - For single-objective: float (PnL)
            - For multi-objective: tuple of (PnL, Sharpe, max_drawdown)
        """
        params_dict = {name: eval(value, {"trial": trial}, {}) for name, value in self.params.items()}
        params_json = json.dumps(params_dict, sort_keys=True)

        with self._params_seen_lock:
            if params_json in self._params_seen:
                raise TrialPruned("Same parameters as previous trial.")

            self._params_seen.add(params_json)

        if self._use_live:
            metrics = self._run_live_submission(params_dict)
        else:
            metrics = self._run_backtest(params_json)

        if self._multi_objective:
            return (metrics.total_pnl, metrics.sharpe_ratio, metrics.max_drawdown)
        else:
            return float(metrics.total_pnl)
