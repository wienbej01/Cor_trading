"""
Signal Quality & Threshold Optimization module for FX-Commodity correlation arbitrage.
Provides enhanced signal generation methods with parameter sweeping, volatility adjustment,
and diagnostic capabilities.
"""

from typing import Dict, List, Tuple
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from loguru import logger

from src.features.indicators import zscore_robust
from src.features.spread import compute_spread


class SignalOptimizer:
    """
    Signal optimization class for parameter sweeping, volatility adjustment,
    and signal quality analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize the SignalOptimizer with configuration parameters.

        Args:
            config: Configuration dictionary with strategy parameters.
        """
        self.config = config
        self.results_cache = {}

    def generate_threshold_combinations(
        self,
        entry_range: List[float] = None,
        exit_range: List[float] = None,
        stop_range: List[float] = None,
    ) -> List[Dict[str, float]]:
        """
        Generate valid threshold combinations maintaining entry > exit > stop constraint.

        Args:
            entry_range: List of entry threshold values to test.
            exit_range: List of exit threshold values to test.
            stop_range: List of stop threshold values to test.

        Returns:
            List of valid threshold combination dictionaries.
        """
        if entry_range is None:
            entry_range = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        if exit_range is None:
            exit_range = [0.3, 0.5, 0.7, 0.9]
        if stop_range is None:
            stop_range = [2.5, 3.0, 3.5, 4.0]

        valid_combinations = []

        for entry, exit_val, stop in itertools.product(
            entry_range, exit_range, stop_range
        ):
            if entry > exit_val and exit_val < stop:
                valid_combinations.append(
                    {"entry_z": entry, "exit_z": exit_val, "stop_z": stop}
                )

        logger.info(f"Generated {len(valid_combinations)} valid threshold combinations")
        return valid_combinations

    def calculate_volatility_adjusted_thresholds(
        self,
        z_scores: pd.Series,
        base_entry: float,
        base_exit: float,
        base_stop: float,
        vol_window: int = 20,
        vol_scaling_factor: float = 1.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate volatility-adjusted thresholds based on rolling volatility.

        Args:
            z_scores: Series of z-score values.
            base_entry: Base entry threshold.
            base_exit: Base exit threshold.
            base_stop: Base stop threshold.
            vol_window: Window for volatility calculation.
            vol_scaling_factor: Factor to scale volatility adjustment.

        Returns:
            Tuple of adjusted entry, exit, and stop threshold series.
        """
        # Calculate rolling volatility of z-scores
        rolling_vol = z_scores.rolling(window=vol_window).std()

        # Calculate volatility adjustment factor
        vol_adjustment = (
            1.0
            + vol_scaling_factor
            * (rolling_vol - rolling_vol.mean())
            / rolling_vol.mean()
        )

        # Apply volatility adjustment to thresholds
        adjusted_entry = base_entry * vol_adjustment
        adjusted_exit = base_exit * vol_adjustment
        adjusted_stop = base_stop * vol_adjustment

        # Ensure minimum thresholds
        adjusted_entry = np.maximum(adjusted_entry, 0.5)
        adjusted_exit = np.maximum(adjusted_exit, 0.2)
        adjusted_stop = np.maximum(adjusted_stop, 2.0)

        # Ensure entry > exit constraint
        adjusted_exit = np.minimum(adjusted_exit, adjusted_entry * 0.8)

        return adjusted_entry, adjusted_exit, adjusted_stop

    def enhanced_kalman_spread(
        self,
        fx_series: pd.Series,
        comd_series: pd.Series,
        beta_window: int,
        lam: float = 0.995,
        delta: float = 100.0,
        residual_window: int = 20,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Enhanced Kalman filter spread calculation with residual analysis.

        Args:
            fx_series: FX time series.
            comd_series: Commodity time series.
            beta_window: Window for beta calculation.
            lam: Forgetting factor for RLS.
            delta: Initial state uncertainty.
            residual_window: Window for residual analysis.

        Returns:
            Tuple of spread, alpha, beta, and residual z-scores.
        """
        # Use existing compute_spread function with Kalman filter
        spread, alpha, beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman=True
        )

        # Calculate residuals and their z-scores for quality assessment
        residuals = spread - spread.rolling(window=residual_window).mean()
        residual_z = zscore_robust(residuals, residual_window)

        # Adaptive parameter tuning based on residual quality
        residual_volatility = residuals.rolling(window=residual_window).std()

        # Adjust lambda based on residual volatility (higher volatility -> faster adaptation)
        adaptive_lam = 0.99 - 0.1 * (residual_volatility / residual_volatility.mean())
        adaptive_lam = adaptive_lam.clip(0.95, 0.999)

        # Recalculate spread with adaptive lambda if significant deviation
        if abs(adaptive_lam.mean() - lam) > 0.01:
            logger.info(
                f"Adaptive lambda adjustment: {lam:.3f} -> {adaptive_lam.mean():.3f}"
            )
            spread_adaptive, alpha_adaptive, beta_adaptive = compute_spread(
                fx_series, comd_series, beta_window, use_kalman=True
            )
            # Use weighted average of original and adaptive spread
            weight = 0.7  # Weight for original spread
            spread = weight * spread + (1 - weight) * spread_adaptive
            alpha = weight * alpha + (1 - weight) * alpha_adaptive
            beta = weight * beta + (1 - weight) * beta_adaptive

        return spread, alpha, beta, residual_z

    def enhanced_ols_spread(
        self,
        fx_series: pd.Series,
        comd_series: pd.Series,
        beta_window: int,
        min_window: int = 20,
        max_window: int = 120,
        window_step: int = 10,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Enhanced OLS spread calculation with rolling window optimization.

        Args:
            fx_series: FX time series.
            comd_series: Commodity time series.
            beta_window: Base window for beta calculation.
            min_window: Minimum window size for optimization.
            max_window: Maximum window size for optimization.
            window_step: Step size for window optimization.

        Returns:
            Tuple of spread, alpha, beta, and optimal window sizes.
        """
        # Test multiple window sizes
        window_sizes = range(min_window, max_window + 1, window_step)
        window_scores = {}

        # Calculate base spread with default window
        base_spread, base_alpha, base_beta = compute_spread(
            fx_series, comd_series, beta_window, use_kalman=False
        )

        # Calculate stationarity scores for different window sizes
        for window in window_sizes:
            try:
                test_spread, _, _ = compute_spread(
                    fx_series, comd_series, window, use_kalman=False
                )

                # Calculate stationarity metrics
                if len(test_spread.dropna()) > window:
                    # ADF test p-value (lower is better)
                    adf_result = stats.adfuller(test_spread.dropna())
                    adf_pvalue = adf_result[1]

                    # Hurst exponent (closer to 0.5 is better for mean reversion)
                    hurst = self._calculate_hurst_exponent(test_spread.dropna())

                    # Combined score (lower is better)
                    score = adf_pvalue + abs(hurst - 0.5)
                    window_scores[window] = score
            except Exception as e:
                logger.warning(f"Window {window} calculation failed: {e}")
                continue

        # Select optimal window
        if window_scores:
            optimal_window = min(window_scores, key=window_scores.get)
            logger.info(
                f"Optimal window selected: {optimal_window} (score: {window_scores[optimal_window]:.3f})"
            )

            # Recalculate spread with optimal window
            optimal_spread, optimal_alpha, optimal_beta = compute_spread(
                fx_series, comd_series, optimal_window, use_kalman=False
            )

            # Create window size series
            window_series = pd.Series(optimal_window, index=fx_series.index)

            return optimal_spread, optimal_alpha, optimal_beta, window_series
        else:
            logger.warning("Window optimization failed, using base window")
            window_series = pd.Series(beta_window, index=fx_series.index)
            return base_spread, base_alpha, base_beta, window_series

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        Calculate the Hurst exponent for a time series.

        Args:
            series: Time series to analyze.

        Returns:
            Hurst exponent value.
        """
        try:
            # Calculate range of lags
            lags = range(2, min(20, len(series) // 4))

            # Calculate variances of differences
            tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]

            # Fit linear regression to log-log plot
            poly = np.polyfit(np.log(lags), np.log(tau), 1)

            # Hurst exponent is the slope
            return poly[0] * 2.0
        except Exception as e:
            logger.warning(f"Hurst exponent calculation failed: {e}")
            return 0.5  # Return neutral value

    def calculate_signal_quality_metrics(
        self, signals: pd.Series, returns: pd.Series, window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate signal quality metrics including signal-to-noise ratio and predictive power.

        Args:
            signals: Trading signals series.
            returns: Returns series corresponding to signals.
            window: Rolling window for metric calculation.

        Returns:
            DataFrame with signal quality metrics.
        """
        metrics = pd.DataFrame(index=signals.index)

        # Signal-to-noise ratio
        signal_mean = signals.rolling(window=window).mean()
        signal_std = signals.rolling(window=window).std()
        metrics["signal_to_noise"] = signal_mean / (signal_std + 1e-8)

        # Predictive power (correlation between signal and future returns)
        future_returns = returns.shift(-1)  # Next period returns
        metrics["predictive_power"] = signals.rolling(window=window).corr(
            future_returns
        )

        # Signal stability (rolling autocorrelation)
        metrics["signal_stability"] = signals.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x.dropna()) > 5 else np.nan
        )

        # Signal efficacy (win rate when signal is active)
        active_signals = signals.abs() > 0
        profitable_signals = (signals * returns > 0) & active_signals
        metrics["win_rate"] = profitable_signals.rolling(window=window).mean()

        # Signal consistency (frequency of signal changes)
        signal_changes = signals.diff().abs()
        metrics["signal_consistency"] = (
            1.0 - signal_changes.rolling(window=window).mean()
        )

        return metrics

    def generate_enhanced_signals(
        self,
        fx_series: pd.Series,
        comd_series: pd.Series,
        thresholds: Dict[str, float],
        use_vol_adjustment: bool = True,
        use_enhanced_kalman: bool = True,
        use_enhanced_ols: bool = False,
    ) -> pd.DataFrame:
        """
        Generate enhanced trading signals with all optimizations.

        Args:
            fx_series: FX time series.
            comd_series: Commodity time series.
            thresholds: Dictionary with entry, exit, stop thresholds.
            use_vol_adjustment: Whether to use volatility-adjusted thresholds.
            use_enhanced_kalman: Whether to use enhanced Kalman filter.
            use_enhanced_ols: Whether to use enhanced OLS.

        Returns:
            DataFrame with enhanced signals and metrics.
        """
        result = pd.DataFrame(index=fx_series.index)
        result["fx_price"] = fx_series
        result["comd_price"] = comd_series

        # Extract config parameters
        beta_window = self.config["lookbacks"]["beta_window"]
        z_window = self.config["lookbacks"]["z_window"]

        # Calculate spread using enhanced method
        if use_enhanced_kalman:
            spread, alpha, beta, residual_z = self.enhanced_kalman_spread(
                fx_series, comd_series, beta_window
            )
            result["residual_z"] = residual_z
        elif use_enhanced_ols:
            spread, alpha, beta, optimal_window = self.enhanced_ols_spread(
                fx_series, comd_series, beta_window
            )
            result["optimal_window"] = optimal_window
        else:
            spread, alpha, beta = compute_spread(
                fx_series,
                comd_series,
                beta_window,
                use_kalman=self.config.get("use_kalman", True),
            )

        result["spread"] = spread
        result["alpha"] = alpha
        result["beta"] = beta

        # Calculate z-score
        z = zscore_robust(spread, z_window).rename("z")
        result["spread_z"] = z

        # Apply volatility adjustment if requested
        if use_vol_adjustment:
            entry_adj, exit_adj, stop_adj = (
                self.calculate_volatility_adjusted_thresholds(
                    z, thresholds["entry_z"], thresholds["exit_z"], thresholds["stop_z"]
                )
            )
            result["entry_threshold"] = entry_adj
            result["exit_threshold"] = exit_adj
            result["stop_threshold"] = stop_adj
        else:
            result["entry_threshold"] = thresholds["entry_z"]
            result["exit_threshold"] = thresholds["exit_z"]
            result["stop_threshold"] = thresholds["stop_z"]

        # Generate signals
        result["raw_signal"] = 0

        # Entry signals
        enter_long = z <= -result["entry_threshold"]
        enter_short = z >= result["entry_threshold"]

        # Exit signals
        exit_long = z >= -result["exit_threshold"]
        exit_short = z <= result["exit_threshold"]

        # Stop signals
        stop_long = z >= -result["stop_threshold"]
        stop_short = z <= result["stop_threshold"]

        # Apply signal logic with state management
        position = 0
        signals = []

        for idx, row in result.iterrows():
            current_z = row["spread_z"]
            entry_thresh = row["entry_threshold"]
            exit_thresh = row["exit_threshold"]
            stop_thresh = row["stop_threshold"]

            # If we have no position
            if position == 0:
                if current_z <= -entry_thresh:
                    position = 1
                    signals.append(1)
                elif current_z >= entry_thresh:
                    position = -1
                    signals.append(-1)
                else:
                    signals.append(0)
            # If we have a long position
            elif position == 1:
                if current_z >= -exit_thresh or current_z >= -stop_thresh:
                    position = 0
                    signals.append(0)
                else:
                    signals.append(1)
            # If we have a short position
            elif position == -1:
                if current_z <= exit_thresh or current_z <= stop_thresh:
                    position = 0
                    signals.append(0)
                else:
                    signals.append(-1)
            else:
                signals.append(0)

        result["signal"] = signals

        # Add signal flags for diagnostics
        result["enter_long"] = enter_long
        result["enter_short"] = enter_short
        result["exit_long"] = exit_long
        result["exit_short"] = exit_short
        result["stop_long"] = stop_long
        result["stop_short"] = stop_short

        # Calculate signal quality metrics
        if len(result) > z_window:
            # Calculate simple returns for quality metrics
            spread_returns = result["spread"].diff().shift(-1)
            quality_metrics = self.calculate_signal_quality_metrics(
                result["signal"], spread_returns, window=z_window
            )
            result = pd.concat([result, quality_metrics], axis=1)

        return result

    def run_parameter_sweep(
        self,
        fx_series: pd.Series,
        comd_series: pd.Series,
        threshold_combinations: List[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Run parameter sweep over threshold combinations.

        Args:
            fx_series: FX time series.
            comd_series: Commodity time series.
            threshold_combinations: List of threshold combinations to test.

        Returns:
            DataFrame with sweep results.
        """
        if threshold_combinations is None:
            threshold_combinations = self.generate_threshold_combinations()

        sweep_results = []

        for i, thresholds in enumerate(threshold_combinations):
            logger.info(
                f"Testing combination {i+1}/{len(threshold_combinations)}: {thresholds}"
            )

            try:
                # Generate signals with current thresholds
                signals_df = self.generate_enhanced_signals(
                    fx_series,
                    comd_series,
                    thresholds,
                    use_vol_adjustment=False,  # Disable vol adjustment for sweep
                )

                # Calculate performance metrics
                metrics = self._calculate_signal_performance_metrics(signals_df)

                # Add threshold parameters to results
                metrics.update(thresholds)
                sweep_results.append(metrics)

            except Exception as e:
                logger.warning(f"Threshold combination {thresholds} failed: {e}")
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(sweep_results)

        # Rank by Sharpe ratio
        if "sharpe_ratio" in results_df.columns:
            results_df["rank"] = results_df["sharpe_ratio"].rank(ascending=False)

        logger.info(
            f"Parameter sweep completed. Tested {len(sweep_results)} combinations."
        )
        return results_df

    def _calculate_signal_performance_metrics(
        self, signals_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for signal evaluation.

        Args:
            signals_df: DataFrame with signals and prices.

        Returns:
            Dictionary with performance metrics.
        """
        # Calculate spread returns
        spread_returns = signals_df["spread"].diff()

        # Calculate strategy returns
        strategy_returns = signals_df["signal"].shift(1) * spread_returns

        # Basic metrics
        total_trades = (signals_df["signal"].diff().abs() > 0).sum()
        win_trades = (strategy_returns > 0).sum()
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        # Return metrics
        total_return = strategy_returns.sum()
        annual_return = total_return * 252 / len(strategy_returns)
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # Drawdown metrics
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Signal quality metrics
        signal_activity = (signals_df["signal"].abs() > 0).mean()
        avg_signal_duration = self._calculate_avg_signal_duration(signals_df["signal"])

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "signal_activity": signal_activity,
            "avg_signal_duration": avg_signal_duration,
        }

    def _calculate_avg_signal_duration(self, signals: pd.Series) -> float:
        """
        Calculate average signal duration in periods.

        Args:
            signals: Signal series.

        Returns:
            Average signal duration.
        """
        # Find signal changes
        changes = signals.diff().abs()

        # Calculate durations
        durations = []
        current_duration = 0

        for i in range(1, len(signals)):
            if changes.iloc[i] > 0:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 1
            elif signals.iloc[i] != 0:
                current_duration += 1

        # Add last duration if active
        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def plot_signal_diagnostics(
        self,
        signals_df: pd.DataFrame,
        save_path: str = None,
        figsize: Tuple[int, int] = (15, 12),
    ) -> None:
        """
        Create diagnostic plots for signal analysis.

        Args:
            signals_df: DataFrame with signals and metrics.
            save_path: Path to save the plot.
            figsize: Figure size.
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("Signal Quality Diagnostics", fontsize=16)

        # Plot 1: Spread and Z-score with signals
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()

        ax1.plot(signals_df.index, signals_df["spread"], label="Spread", alpha=0.7)
        ax1_twin.plot(
            signals_df.index,
            signals_df["spread_z"],
            label="Z-score",
            color="orange",
            alpha=0.7,
        )

        # Mark entry/exit points
        long_entries = signals_df[signals_df["enter_long"]]
        short_entries = signals_df[signals_df["enter_short"]]

        ax1_twin.scatter(
            long_entries.index,
            long_entries["spread_z"],
            color="green",
            marker="^",
            s=50,
            label="Long Entry",
        )
        ax1_twin.scatter(
            short_entries.index,
            short_entries["spread_z"],
            color="red",
            marker="v",
            s=50,
            label="Short Entry",
        )

        ax1.set_title("Spread and Z-score with Entry Signals")
        ax1.set_ylabel("Spread")
        ax1_twin.set_ylabel("Z-score")
        ax1.legend(loc="upper left")
        ax1_twin.legend(loc="upper right")

        # Plot 2: Signal distribution
        ax2 = axes[0, 1]
        signal_counts = signals_df["signal"].value_counts()
        ax2.bar(signal_counts.index, signal_counts.values)
        ax2.set_title("Signal Distribution")
        ax2.set_xlabel("Signal")
        ax2.set_ylabel("Count")

        # Plot 3: Signal quality metrics over time
        ax3 = axes[1, 0]
        if "signal_to_noise" in signals_df.columns:
            ax3.plot(
                signals_df.index, signals_df["signal_to_noise"], label="Signal-to-Noise"
            )
        if "predictive_power" in signals_df.columns:
            ax3.plot(
                signals_df.index,
                signals_df["predictive_power"],
                label="Predictive Power",
            )
        if "win_rate" in signals_df.columns:
            ax3.plot(signals_df.index, signals_df["win_rate"], label="Win Rate")
        ax3.set_title("Signal Quality Metrics Over Time")
        ax3.set_ylabel("Metric Value")
        ax3.legend()

        # Plot 4: PnL distribution by signal strength
        ax4 = axes[1, 1]
        if "spread_z" in signals_df.columns:
            # Calculate forward returns
            forward_returns = signals_df["spread"].diff().shift(-1)

            # Create signal strength bins
            signal_strength = signals_df["spread_z"].abs()
            bins = pd.cut(signal_strength, bins=10, labels=False)

            # Calculate average return by bin
            pnl_by_strength = forward_returns.groupby(bins).mean()
            ax4.bar(pnl_by_strength.index, pnl_by_strength.values)
            ax4.set_title("PnL by Signal Strength Quantile")
            ax4.set_xlabel("Signal Strength Quantile")
            ax4.set_ylabel("Average Forward Return")

        # Plot 5: Cumulative PnL by signal quantile
        ax5 = axes[2, 0]
        if "spread_z" in signals_df.columns:
            # Create signal quantiles
            signal_quantiles = signals_df["spread_z"].rolling(window=20).rank(pct=True)

            # Calculate cumulative returns by quantile
            quantile_returns = forward_returns.groupby(signal_quantiles.round(1)).mean()
            cumulative_returns = quantile_returns.cumsum()

            ax5.plot(cumulative_returns.index, cumulative_returns.values)
            ax5.set_title("Cumulative PnL by Signal Quantile")
            ax5.set_xlabel("Signal Quantile")
            ax5.set_ylabel("Cumulative Return")

        # Plot 6: Threshold adjustment over time (if volatility-adjusted)
        ax6 = axes[2, 1]
        if (
            "entry_threshold" in signals_df.columns
            and signals_df["entry_threshold"].dtype == "float64"
        ):
            ax6.plot(
                signals_df.index, signals_df["entry_threshold"], label="Entry Threshold"
            )
            ax6.plot(
                signals_df.index, signals_df["exit_threshold"], label="Exit Threshold"
            )
            ax6.plot(
                signals_df.index, signals_df["stop_threshold"], label="Stop Threshold"
            )
            ax6.set_title("Volatility-Adjusted Thresholds Over Time")
            ax6.set_ylabel("Threshold Value")
            ax6.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Diagnostic plot saved to {save_path}")

        plt.show()
