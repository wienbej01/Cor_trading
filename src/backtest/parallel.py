"""
Parallel backtesting framework for FX-Commodity correlation arbitrage strategy.
Implements parallel execution of backtests for different models and parameter sets.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from loguru import logger
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from backtest.engine import run_backtest


@dataclass
class BacktestConfig:
    """Configuration for parallel backtesting."""

    # Parallel execution parameters
    max_workers: int = 4  # Number of parallel workers
    timeout: int = 3600  # Timeout in seconds (1 hour)

    # Model selection
    models_to_test: List[str] = field(default_factory=lambda: ["ols", "kalman", "corr"])

    # Parameter ranges for optimization
    entry_z_range: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    exit_z_range: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    stop_z_range: List[float] = field(default_factory=lambda: [2.5, 3.0, 3.5, 4.0])

    # Data parameters
    train_test_split: float = 0.7  # Proportion of data for training


@dataclass
class BacktestResult:
    """Result of a single backtest."""

    model_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    equity_curve: pd.Series
    signals: pd.DataFrame
    execution_time: float


def _run_single_backtest(args: Tuple[str, Dict, pd.DataFrame, Dict]) -> BacktestResult:
    """
    Run a single backtest (used in parallel execution).

    Args:
        args: Tuple of (model_name, params, data, config)

    Returns:
        BacktestResult with results.
    """
    import time

    start_time = time.time()

    model_name, params, data, config = args

    try:
        # Update config with current parameters
        config_copy = config.copy()
        config_copy["thresholds"] = config_copy.get("thresholds", {}).copy()
        config_copy["thresholds"].update(params)

        # Run backtest
        backtest_df, metrics = run_backtest(data, config_copy)

        # Extract equity curve and signals
        equity_curve = (
            backtest_df["equity"] if "equity" in backtest_df.columns else pd.Series()
        )
        signals = (
            backtest_df[["signal"]]
            if "signal" in backtest_df.columns
            else pd.DataFrame()
        )

        execution_time = time.time() - start_time

        return BacktestResult(
            model_name=model_name,
            params=params,
            metrics=metrics,
            equity_curve=equity_curve,
            signals=signals,
            execution_time=execution_time,
        )
    except Exception as e:
        logger.error(f"Backtest failed for {model_name} with params {params}: {e}")
        execution_time = time.time() - start_time

        # Return empty result with error
        return BacktestResult(
            model_name=model_name,
            params=params,
            metrics={"error": str(e)},
            equity_curve=pd.Series(),
            signals=pd.DataFrame(),
            execution_time=execution_time,
        )


def _prepare_data_for_model(model_name: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for a specific model.

    Args:
        model_name: Name of the model.
        data: Input data.

    Returns:
        Prepared data for the model.
    """
    # For now, return data as-is
    # In the future, this could transform data for specific models
    return data


class ParallelBacktester:
    """Parallel backtesting framework for model comparison."""

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize the parallel backtester.

        Args:
            config: Backtest configuration.
        """
        self.config = config or BacktestConfig()
        self.results: List[BacktestResult] = []

    def run_model_comparison(
        self, data: pd.DataFrame, base_config: Dict, models: List[str] = None
    ) -> List[BacktestResult]:
        """
        Run backtests for different models in parallel.

        Args:
            data: Market data for backtesting.
            base_config: Base configuration for backtesting.
            models: List of model names to test. If None, uses config.models_to_test.

        Returns:
            List of backtest results.
        """
        if models is None:
            models = self.config.models_to_test

        # Prepare arguments for parallel execution
        args_list = []
        for model_name in models:
            # Prepare data for this model
            model_data = _prepare_data_for_model(model_name, data)

            # Use default parameters for model comparison
            params = {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 3.5}

            args_list.append((model_name, params, model_data, base_config))

        # Run backtests in parallel
        results = self._run_parallel_backtests(args_list)
        self.results.extend(results)

        return results

    def run_parameter_sweep(
        self, data: pd.DataFrame, base_config: Dict, model_name: str = "default"
    ) -> List[BacktestResult]:
        """
        Run parameter sweep for a specific model in parallel.

        Args:
            data: Market data for backtesting.
            base_config: Base configuration for backtesting.
            model_name: Name of the model to test.

        Returns:
            List of backtest results.
        """
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()

        # Prepare arguments for parallel execution
        args_list = []
        for params in param_combinations:
            # Prepare data for this model
            model_data = _prepare_data_for_model(model_name, data)
            args_list.append((model_name, params, model_data, base_config))

        # Run backtests in parallel
        results = self._run_parallel_backtests(args_list)
        self.results.extend(results)

        return results

    def _generate_param_combinations(self) -> List[Dict[str, float]]:
        """
        Generate valid parameter combinations.

        Returns:
            List of parameter combination dictionaries.
        """
        combinations = []

        for entry_z in self.config.entry_z_range:
            for exit_z in self.config.exit_z_range:
                for stop_z in self.config.stop_z_range:
                    # Ensure valid parameter constraints
                    if entry_z > exit_z and stop_z > entry_z:
                        combinations.append(
                            {"entry_z": entry_z, "exit_z": exit_z, "stop_z": stop_z}
                        )

        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations

    def _run_parallel_backtests(self, args_list: List[Tuple]) -> List[BacktestResult]:
        """
        Run backtests in parallel.

        Args:
            args_list: List of arguments for backtest functions.

        Returns:
            List of backtest results.
        """
        results = []

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_run_single_backtest, args): args for args in args_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_args, timeout=self.config.timeout):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"Completed backtest: {result.model_name} with params {result.params}"
                    )
                except Exception as e:
                    args = future_to_args[future]
                    logger.error(f"Backtest failed for args {args}: {e}")
                    # Add error result
                    results.append(
                        BacktestResult(
                            model_name=args[0] if args else "unknown",
                            params=args[1] if len(args) > 1 else {},
                            metrics={"error": str(e)},
                            equity_curve=pd.Series(),
                            signals=pd.DataFrame(),
                            execution_time=0.0,
                        )
                    )

        return results

    def get_best_results(
        self, metric: str = "sharpe_ratio", top_n: int = 5
    ) -> List[BacktestResult]:
        """
        Get best backtest results based on a metric.

        Args:
            metric: Metric to rank by.
            top_n: Number of top results to return.

        Returns:
            List of top backtest results.
        """
        # Filter results with valid metrics
        valid_results = [
            r
            for r in self.results
            if r.metrics and metric in r.metrics and not np.isnan(r.metrics[metric])
        ]

        # Sort by metric (descending for most metrics)
        reverse = metric not in ["max_drawdown"]  # Lower is better for max_drawdown
        sorted_results = sorted(
            valid_results, key=lambda r: r.metrics[metric], reverse=reverse
        )

        return sorted_results[:top_n]

    def aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate all backtest results into a summary DataFrame.

        Returns:
            DataFrame with aggregated results.
        """
        if not self.results:
            return pd.DataFrame()

        # Collect results data
        data = []
        for result in self.results:
            row = {
                "model_name": result.model_name,
                "execution_time": result.execution_time,
            }

            # Add parameters
            row.update(result.params)

            # Add metrics
            if result.metrics:
                row.update(result.metrics)

            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Sort by Sharpe ratio if available
        if "sharpe_ratio" in df.columns:
            df = df.sort_values("sharpe_ratio", ascending=False)

        return df


def run_parallel_backtest(
    data: pd.DataFrame, config: Dict, backtest_config: BacktestConfig = None
) -> ParallelBacktester:
    """
    Run parallel backtest with default configuration.

    Args:
        data: Market data for backtesting.
        config: Configuration for backtesting.
        backtest_config: Backtest configuration. If None, uses default.

    Returns:
        ParallelBacktester instance with results.
    """
    if backtest_config is None:
        backtest_config = BacktestConfig()

    backtester = ParallelBacktester(backtest_config)

    # Run model comparison
    backtester.run_model_comparison(data, config)

    return backtester


def create_default_backtest_config() -> BacktestConfig:
    """
    Create a default backtest configuration.

    Returns:
        Default backtest configuration.
    """
    return BacktestConfig()
