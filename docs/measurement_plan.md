# Measurement Plan - FX-Commodity Mean Reversion

This document defines the KPIs, failure modes, and thresholds for evaluating the FX-commodity correlation mean-reversion strategy. Metrics are computed with strict adherence to no-leakage principles.

Code references
- Performance engine: [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:256)
- Distribution analysis: [backtest.distribution_analysis.analyze_return_distribution()](src/backtest/distribution_analysis.py:1)
- Rolling metrics: [backtest.rolling_metrics.calculate_rolling_metrics()](src/backtest/rolling_metrics.py:1)

1) Key Performance Indicators (KPIs)
- Total Return: Cumulative PnL over the backtest period
- Annual Return: Geometric annualized return (includes costs)
- Sharpe Ratio: Mean/StdDev of daily returns (annualized)
- Maximum Drawdown: Largest peak-to-trough equity loss
- Number of Trades: Total round-trip trades executed
- Win Rate: Percentage of profitable trades
- Average Win: Mean return of winning trades
- Average Loss: Mean return of losing trades
- Profit Factor: Gross wins / Gross losses
- Skewness: Asymmetry of trade return distribution via [backtest.distribution_analysis.analyze_return_distribution()](src/backtest/distribution_analysis.py:1)
- Kurtosis: Tail heaviness of trade return distribution
- VaR (95%): Loss threshold not exceeded 95% of the time
- CVaR (95%): Average loss when losses exceed VaR

2) Failure modes and thresholds
- Zero trades: Indicates over-filtering or data issues
- Sharpe < 0: Strategy not generating risk-adjusted returns
- MaxDD > 30%: Unacceptable drawdown risk
- Win Rate < 40%: Poor trade selection
- Profit Factor < 1.1: Insufficient profitability
- Skewness < -1: Heavy left tail (large losses)
- Kurtosis > 10: Excessive tail risk
- VaR (95%) < -5%: High daily loss risk

3) Regime bucketing for granular analysis
- Time buckets: 2-year periods for long-term consistency
- Volatility regimes: Low (<0.05), Medium (0.05-0.10), High (>0.10)
- Market conditions: Trending Up (>2%), Ranging (-2% to 2%), Trending Down (<-2%)

4) Testability checklist
- [ ] Metrics computed only from executed trades (no leakage)
- [ ] Regime bucketing uses only past data at decision time
- [ ] Thresholds align with risk management guardrails
- [ ] Distribution metrics match [backtest.distribution_analysis.analyze_return_distribution()](src/backtest/distribution_analysis.py:1)

5) Economic rationale for metrics
- Return-based metrics capture profitability after realistic costs
- Risk metrics (Sharpe, MaxDD) ensure strategy viability
- Trade-level metrics validate signal quality and execution
- Distributional metrics identify tail and asymmetry risks

6) Implementation notes
- All metrics in [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:256)
- Rolling metrics via [backtest.rolling_metrics.calculate_rolling_metrics()](src/backtest/rolling_metrics.py:1)
- Distribution analysis via [backtest.distribution_analysis.analyze_return_distribution()](src/backtest/distribution_analysis.py:1)
- NaN handling for sparse trades in [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:275)