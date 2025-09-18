# Handoff Package — Testing & Analytics Hub (Walk-Forward, Ablations, Attribution)

Scope
- This package delivers a reproducible test plan to validate the FX-Commodity mean-reversion system via Walk-Forward Optimization (WFO), ablations, and attribution with leakage controls (purged/embargoed splits).
- All referenced code and docs are in-repo with clickable pointers.

Key references (click-through)
- CLI runner: [src/run_backtest.py](src/run_backtest.py:1)
- Backtest engine: [src/backtest/engine.py](src/backtest/engine.py:57)
- Spread/zscore: [src/features/spread.py](src/features/spread.py:54), [src/features/indicators.py](src/features/indicators.py:233)
- Regime filters/HMM: [src/features/regime.py](src/features/regime.py:355)
- Cointegration diagnostics: [src/features/cointegration.py](src/features/cointegration.py:52)
- Rulebook: [docs/rulebook.md](docs/rulebook.md)
- Label spec: [docs/label_spec.md](docs/label_spec.md)
- Feature spec: [docs/feature_spec.md](docs/feature_spec.md)
- Measurement plan: [docs/measurement_plan.md](docs/measurement_plan.md)
- Latest performance table: [docs/performance_summary.md](docs/performance_summary.md)
- Aggregation utility: [aggregate_backtest_reports.py](aggregate_backtest_reports.py:1)

1) Environment and Repro
- Python: 3.9+ recommended
- Install: pip install -r requirements.txt
- Data: Yahoo Finance + futures symbols; loader handles alignment
- Baseline run (per pair):
  - PYTHONPATH=. python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15 --save-data
- Multi-pair batch:
  - PYTHONPATH=. bash -lc 'for p in usdcad_wti usdnok_brent audusd_gold usdzar_platinum usdclp_copper usdmxn_wti usdbrl_soybeans; do python src/run_backtest.py run --pair $p --start 2015-01-01 --end 2025-08-15 --save-data; done'
- Aggregate reports:
  - PYTHONPATH=. python aggregate_backtest_reports.py  → writes [docs/performance_summary.md](docs/performance_summary.md)

2) WFO (Walk-Forward Optimization) Plan
Signal execution realism (one-bar delay) and costs are enforced in [backtest.engine.backtest_pair()](src/backtest/engine.py:82,109,123).
- Rolling Train/Test Windows (5y train, 1y test, step 1y):
  - W1: Train 2015-01-01 → 2019-12-31 | Test 2020-01-01 → 2020-12-31
  - W2: Train 2016-01-01 → 2020-12-31 | Test 2021-01-01 → 2021-12-31
  - W3: Train 2017-01-01 → 2021-12-31 | Test 2022-01-01 → 2022-12-31
  - W4: Train 2018-01-01 → 2022-12-31 | Test 2023-01-01 → 2023-12-31
  - W5: Train 2019-01-01 → 2023-12-31 | Test 2024-01-01 → 2024-12-31
  - W6: Train 2020-01-01 → 2024-12-31 | Test 2025-01-01 → 2025-08-15
- Purge/Embargo:
  - Effective holding may reach time_stop.max_days and can scale with OU half-life (see [src/strategy/mean_reversion.py](src/strategy/mean_reversion.py:339,360-366)). Use purge=40 trading days and embargo=20 trading days between train/test boundaries as a conservative default.
  - For supervised research use [docs/label_spec.md](docs/label_spec.md) and enforce embargo per window (no overlap of label windows across splits).

3) Hyperparameter Tuning Grid (train fold only)
Tune on train; freeze on test.
- thresholds.entry_z: [1.5, 2.0, 2.5]
- thresholds.exit_z: [0.5, 1.0]
- thresholds.stop_z: [3.0, 3.5, 4.0]
- lookbacks.corr_window: [20, 40]
- regime.min_abs_corr: [0.30, 0.40, 0.50]
- regime.enable_hmm: [true, false]
- regime.hmm.{window: [126, 252], df: [3.0, 5.0], tol: [1e-4, 1e-3], n_iter: fixed 100}

Selection Criterion (train): maximize Sharpe, tie-break by Profit Factor then lower MaxDD (see [docs/measurement_plan.md](docs/measurement_plan.md))

4) Ablations (to quantify signal value)
Run each ablation with the same WFO windows:
- A0 Baseline: all gates active; structural scaling ON (see [src/strategy/mean_reversion.py](src/strategy/mean_reversion.py:181-184,186-201))
- A1 No HMM: regime.enable_hmm=false
- A2 No structural scaling: set structural_scale=1.0 (patch [src/strategy/mean_reversion.py](src/strategy/mean_reversion.py:181))
- A3 No volatility normalization: set vol_scale=1.0 (patch [src/strategy/mean_reversion.py](src/strategy/mean_reversion.py:177-184))
- A4 No correlation gate: force correlation_gate to True (patch [src/features/regime.py](src/features/regime.py:63-73) in a branch)
- A5 Simpler z: replace robust z with standard z (guard in [src/features/indicators.py](src/features/indicators.py:13-41))

5) KPIs, Acceptance, and Reporting
Compute metrics per test fold and aggregate (median preferred; robust to outliers). See [src/backtest/engine.py](src/backtest/engine.py:256-339).
- Required KPIs per test fold:
  - total_return, annual_return, sharpe_ratio, max_drawdown, num_trades, win_rate, profit_factor
  - distribution: skew, kurtosis, VaR/CVaR 95% (see [src/backtest/distribution_analysis.py](src/backtest/distribution_analysis.py:1))
- Acceptance (initial guardrails; adjust post-study):
  - Median OOS Sharpe ≥ 0.20, Profit Factor ≥ 1.20
  - OOS MaxDD ≤ 25%
  - No test fold with 0 trades; no single fold dominates (>60% of cumulative OOS return)
- Required artifacts to save per fold:
  - Test fold report (txt), signals/backtest CSVs, tuned params snapshot
  - Aggregate markdown: per-pair fold table + cross-pair summary

6) Execution Recipes
- Single WFO fold (example: usdnok_brent):
  - Train tune loop: run multiple configs (grid above) on Train range; pick best by Sharpe→PF→MaxDD
  - Test run with frozen params on Test range
- Batch across pairs (baseline config):
  - PYTHONPATH=. bash -lc 'for p in usdcad_wti usdnok_brent audusd_gold usdzar_platinum usdclp_copper usdmxn_wti usdbrl_soybeans; do python src/run_backtest.py run --pair $p --start 2015-01-01 --end 2025-08-15 --save-data; done'
  - Aggregate: PYTHONPATH=. python aggregate_backtest_reports.py
- Post-Processing:
  - Collate fold-level results into docs/wfo_summary.md (use aggregate script as a template)

7) Attribution & Diagnostics
- Regime-bucket attribution (time buckets, vol/trend regimes as in measurement plan)
- Entry/exit threshold histograms; mean |z| at entries (already printed by CLI)
- HMM state mix in traded bars (see [src/features/regime.py](src/features/regime.py:428-451))
- Trade duration distribution relative to time-stop (see [src/strategy/mean_reversion.py](src/strategy/mean_reversion.py:339-407))

8) Risk & Realism Checks
- One-bar delay verified by position shift in [src/backtest/engine.py](src/backtest/engine.py:109-112)
- Costs applied per trade in [src/backtest/engine.py](src/backtest/engine.py:123-132)
- Dynamic sizing via ATR proxy in [src/strategy/mean_reversion.py](src/strategy/mean_reversion.py:313-335)

9) Deliverables in this Handoff
- Rulebook: [docs/rulebook.md](docs/rulebook.md)
- Features: [docs/feature_spec.md](docs/feature_spec.md)
- Labels: [docs/label_spec.md](docs/label_spec.md)
- Metrics plan: [docs/measurement_plan.md](docs/measurement_plan.md)
- Latest performance snapshot: [docs/performance_summary.md](docs/performance_summary.md)
- Aggregation utility: [aggregate_backtest_reports.py](aggregate_backtest_reports.py:1)

10) Open Items for Testing & Analytics
- Implement WFO driver (lightweight): loop over windows, tune-on-train, freeze-on-test, persist tuned params and results
- Add purged/embargoed split helpers for ML label training per [docs/label_spec.md](docs/label_spec.md)
- Produce fold-by-fold and cross-pair summary tables and plots
- Run ablations A1–A5; quantify lift vs baseline; report statistical significance (bootstrap or White’s Reality Check)

Appendix — Example WFO skeleton (pseudo)
```python
# Pseudo only: outlines orchestration
windows = [
  ("2015-01-01","2019-12-31","2020-01-01","2020-12-31"),
  ("2016-01-01","2020-12-31","2021-01-01","2021-12-31"),
  # ...
]
grid = {
  "entry_z":[1.5,2.0,2.5],
  "exit_z":[0.5,1.0],
  "stop_z":[3.0,3.5,4.0],
  "corr_window":[20,40],
  "min_abs_corr":[0.30,0.40,0.50],
  "enable_hmm":[True,False],
  "hmm_window":[126,252],
  "hmm_df":[3.0,5.0],
}
for pair in pairs:
  for (ts,te,vs,ve) in windows:
    best_params = tune_on_train(pair, ts, te, grid)   # maximize Sharpe→PF→MinDD
    test_metrics = run_on_test(pair, vs, ve, best_params)
    persist(pair, (ts,te,vs,ve), best_params, test_metrics)
```

Contact
- Strategy owner: See commit history and [docs/rulebook.md](docs/rulebook.md)
- For execution realism/risk constraints: consult [docs/risk_execution_controls.md](docs/risk_execution_controls.md)