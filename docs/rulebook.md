# Strategy Rulebook - FX-Commodity Correlation Mean Reversion

This document specifies the exact trading rules, filters, sessions, risk, and management for the FX-Commodity correlation mean-reversion system. All rules are implemented in the codebase and referenced with clickable links to concrete implementations.

Code references
- Signal core: [strategy.mean_reversion.generate_signals()](src/strategy/mean_reversion.py:48)
- Time stop: [strategy.mean_reversion.apply_time_stop()](src/strategy/mean_reversion.py:339)
- Sizing: [strategy.mean_reversion.calculate_position_sizes()](src/strategy/mean_reversion.py:293)
- Spread: [features.spread.compute_spread()](src/features/spread.py:54)
- Robust z-score: [features.indicators.zscore_robust()](src/features/indicators.py:233)
- Corr gate: [features.regime.correlation_gate()](src/features/regime.py:17)
- Combined regime: [features.regime.combined_regime_filter()](src/features/regime.py:355)
- OU half-life: [features.cointegration.ou_half_life()](src/features/cointegration.py:52)
- Hurst exponent: [features.cointegration.hurst_exponent()](src/features/cointegration.py:124)
- Engine: [backtest.engine.backtest_pair()](src/backtest/engine.py:57), [backtest.engine.calculate_performance_metrics()](src/backtest/engine.py:256), [backtest.engine.create_backtest_report()](src/backtest/engine.py:342)

1) Sessions and bar convention
- Data frequency: Daily close
- Signal evaluation time: bar_close at t; execution at t+1 close (one-bar delay)
- Costs: Parameterized in configs; applied in engine

2) Instruments
- FX legs: e.g., USDCAD, USDNOK, AUDUSD, USDZAR, USDCLP, USDMXN, USDBRL
- Commodity legs: WTI (CL=F), Brent (BZ=F), Gold (GC=F), Platinum (PL=F), Copper (HG=F), Soybeans (ZS=F)
- Symbols per pair defined in [configs/pairs.yaml](configs/pairs.yaml)

3) Spread and z-score
- Compute multiplicative intercept alpha_t and power beta_t in log domain, using RLS (Kalman) when enabled, else rolling OLS
- Spread in levels: S_t = FX_t − alpha_t × COMD_t^{beta_t} from [features.spread.compute_spread()](src/features/spread.py:54)
- Robust z-score: z_t = (S_t − med_{t−w:t}(S)) / (1.4826×MAD_{t−w:t} + 1e−12) in [features.indicators.zscore_robust()](src/features/indicators.py:233)

4) Regime gating (leakage-free; only past info)
- Correlation gate: valid_t = |ρ(FX, COMD)|_{w} ≥ min_abs_corr using [features.regime.correlation_gate()](src/features/regime.py:17)
- External filter: [features.regime.combined_regime_filter()](src/features/regime.py:355) integrates:
  • Volatility regimes (low/normal/high)
  • Trend regimes (range vs trend)
  • HMM-based ranging (state=0), optional VIX overlay
- Structural diagnostics as soft prior (not a hard block):
  • ADF p-value on spread ≤ adf_p
  • OU half-life in [2, 252] trading days via [features.cointegration.ou_half_life()](src/features/cointegration.py:52)
  • Hurst exponent < 0.6 via [features.cointegration.hurst_exponent()](src/features/cointegration.py:124)
- Implementation uses these diagnostics to scale thresholds, see [strategy.mean_reversion.generate_signals()](src/strategy/mean_reversion.py:181)

5) Dynamic thresholds
- Spread volatility normalization: vol_scale_t = stdev_20(S)_t / median_252(stdev_20(S)) clipped to [0.7, 1.5]
- Structural scaling: structural_scale = 0.9 if diagnostics OK else 1.1
- Entry: entry_z_dyn_t = max(0.8, entry_z × structural_scale × vol_scale_t)
- Exit: exit_z_dyn_t = min(1.25, exit_z × (1/structural_scale) × vol_scale_t)

6) Entry/Exit rules
- Let good_regime_t = correlation_gate_t AND external_regime_t
- Long entry condition at decision time t:
  • z_t ≤ −entry_z_dyn_t AND good_regime_t
- Short entry condition at decision time t:
  • z_t ≥ entry_z_dyn_t AND good_regime_t
- Exit to flat at decision time t:
  • |z_t| ≤ exit_z_dyn_t OR z-stop OR time-stop
- Execution at t+1 close (one-bar delay) realized by shifting positions in [backtest.engine.backtest_pair()](src/backtest/engine.py:109)

7) Stops and management
- Z-stop:
  • Long: trigger if z_t ≤ −stop_z
  • Short: trigger if z_t ≥ stop_z
- Time-stop: in [strategy.mean_reversion.apply_time_stop()](src/strategy/mean_reversion.py:339)
  • effective_max_days = min(max_days, round(3×OU_half_life)) bounded to [2, max_days]
  • If days_held ≥ effective_max_days, set signal_t = 0 and mark time_stop_exit_t = True
- Profit targets: disabled by default; placeholders exist for future extension

8) Position sizing and execution
- Inverse-volatility sizing per leg from [strategy.mean_reversion.calculate_position_sizes()](src/strategy/mean_reversion.py:293):
  • fx_size_t = target_vol_per_leg / ATR_proxy(fx_close, atr_window)
  • comd_size_t = target_vol_per_leg / ATR_proxy(comd_close, atr_window)
  • If inverse_fx_for_quote_ccy_strength=True, divide fx_size_t by fx_price_t
- Positions:
  • fx_position_t = fx_size_t × signal_t
  • comd_position_t = −comd_size_t × signal_t (opposite leg)
- Costs:
  • Engine applies bps per trade; see [backtest.engine.backtest_pair()](src/backtest/engine.py:123)

9) PnL and equity
- Per-bar PnL before costs:
  • fx_pnl_t = delayed_fx_position_t × Δfx_price_t
  • comd_pnl_t = delayed_comd_position_t × Δcomd_price_t
  • total_pnl_t = fx_pnl_t + comd_pnl_t − costs_t (see [backtest.engine.backtest_pair()](src/backtest/engine.py:118))
- Equity curve proxy:
  • equity_t = ∏_{i≤t} (1 + total_pnl_i), see [backtest.engine.backtest_pair()](src/backtest/engine.py:139)
- Trade stats stored via [backtest.engine._calculate_trade_stats()](src/backtest/engine.py:173)

10) Risk guardrails
- Max position, daily/weekly loss caps, circuit breakers parameterized in [configs/pairs.yaml](configs/pairs.yaml)
- Structural softening prevents over-filtering but preserves sanity bounds (OU HL cap, Hurst < 0.6)
- All signals/positions respect one-bar delay

11) Parameter template (per pair)
- lookbacks.beta_window: 90
- lookbacks.z_window: 40
- lookbacks.corr_window: 20
- thresholds.entry_z: 2.0
- thresholds.exit_z: 1.0
- thresholds.stop_z: 3.5
- thresholds.adf_p: 0.05
- regime.min_abs_corr: 0.4
- time_stop.max_days: 20
- sizing.atr_window: 20
- sizing.target_vol_per_leg: 0.01

12) Testability checklist
- Signals only reference past data at decision time t (z_t, gates_t)
- Entries/exits materialize at t+1 close (verify delayed_position columns in [backtest.engine.backtest_pair()](src/backtest/engine.py:110))
- Purge/embargo for ML not needed in rule-based backtest but must be enforced for supervised research (see forthcoming label spec)
- Costs, thresholds, and windows are parameterized in [configs/pairs.yaml](configs/pairs.yaml)

13) Economic rationale
- Exporter FX is correlated with its commodity; large, regime-consistent dislocations tend to mean-revert
- Correlation+regime filters remove most trending/high-vol states; structural diagnostics down-weight thresholds when weaker

14) Reproduction
- CLI: python src/run_backtest.py run --pair <pair> --start 2015-01-01 --end 2025-08-15 --save-data
- See latest results in backtest_results and tracker in [docs/backtest_tracker.md](docs/backtest_tracker.md)