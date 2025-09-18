# System Technical Documentation

## Overview
This document describes the technical architecture and process flow of the Cor Trading System, a quantitative automated trading platform for FX-commodity pairs. The system emphasizes profitability, risk control, and market realism.

**Current Version:** v0.2.0 (ML-enhancement branch, Phase 1.1)
**Date:** 2025-09-18
**Commit:** 32e4166
**Constitutional Fitness Score (CFS):** 7/10 (Enhanced diagnostics; needs ML/risk enhancements)

## Process Flow

### 1. Data Ingestion
- **Sources:** Yahoo Finance (via `src/data/yahoo_loader.py`), EIA API (`src/data/eia_api.py`), Broker API (`src/data/broker_api.py`).
- **Pairs:** Configured in `configs/pairs.yaml` (e.g., USDCAD-WTI, USDNOK-Brent).
- **Temporal Integrity:** Daily/H1 data from 2015-01-01 to present; no lookahead bias enforced.
- **Assumptions:** Data availability; slippage/costs modeled post-ingestion.
- **Flow:** Raw data → cleaning → feature preparation.

### 2. Features
- **Core Modules:** `src/features/` (cointegration, indicators, regime, spread, signal_optimization).
- **Regime Detection:** HMM-based (`notebooks/hmm_regime_prototype.py`, `src/features/regime.py`).
- **Diagnostics:** `src/features/diagnostics.py`, `src/ml/diagnostics.py`.
- **Principles:** Parsimony; explainability; no overfitting.
- **Current State:** Basic mean-reversion signals; ML filter pending (Phase 1).

### 3. Strategy & Signals
- **Strategies:** Mean-reversion (`src/strategy/mean_reversion.py`, D1/H1 variants).
- **Signal Optimization:** `src/features/signal_optimization.py`.
- **ML Integration:** Ensemble (`src/ml/ensemble.py`), Trade Filter (`src/ml/filter.py`).
- **Prohibitions:** No arbitrary patterns; deterministic seeds.

### 4. Risk Management
- **Manager:** `src/risk/manager.py` (position sizing, drawdown limits).
- **Stress Testing:** `src/risk/stress_test.py`.
- **Execution:** Stub broker (`src/exec/broker_stub.py`), Policy (`src/exec/policy.py`).
- **Non-Negotiables:** Costs/slippage inclusion; liquidity focus; circuit breakers.
- **Enhancements:** Phase 2 - Robustness controls.

### 5. Backtesting & Testing
- **Engine:** `src/backtest/engine.py`, Parallel (`src/backtest/parallel.py`), Rolling Metrics (`src/backtest/rolling_metrics.py`).
- **Runner:** `src/run_backtest.py`.
- **Tests:** Pytest suite (`tests/`); integration, ML diagnostics, risk enhanced.
- **Analytics:** Distribution analysis (`src/backtest/distribution_analysis.py`); regime buckets.
- **Metrics:** Comprehensive metrics catalog (`docs/dev/metrics_catalog.md`); equity, trade, daily, and cost metrics; stratified by regimes (time, vol, market condition).
- **Phase 3:** Walk-forward, Monte Carlo, ablations.

### 6. Architecture
- **Core:** `src/core/config.py`.
- **Interfaces:** Feature preparation/validation (`src/interfaces/`).
- **Utils:** Logging (`src/utils/logging.py`).
- **ML:** `src/ml/` (diagnostics, ensemble, filter).
- **Dependencies:** `requirements.txt`; Python 3.x, pandas, numpy, etc.
- **Configs:** `configs/pairs.yaml`.
- **Diagram:**
```
Data Ingestion → Features → Strategy → Risk → Backtest → Reports
          ↓
       ML Filter (Phase 1)
```

### 7. Operations
- **Deployment:** Paper/live via `src/exec/`; monitoring pending (Phase 4).
- **CI/CD:** Smoke tests (`scripts/qa/smoke.sh` - Phase 0); GitHub workflows pending.
- **Logging:** Centralized via `src/utils/logging.py` with run_id, seed, and git SHA.
- **Alerts/Kill-Switch:** To be implemented in Operations & SRE mode.
- **Reports:** Immutable artifacts in `reports/<pair>/<run_id>/` with summary.json, trades.parquet, config.json.

## Changes in Phase 0
- Verified ML-enhancement branch.
- Initialized CHANGELOG.md and this SYSTEM_TECH_DOC.md.
- Created smoke test setup (pending execution).
- No code changes implemented; CFS baseline established for Phase 0 documentation.

## Changes in Phase 1.1 (Unified Metrics & Run Artifacts)
- Implemented comprehensive metrics module (`src/backtest/metrics.py`).
- Enhanced logging with run_id, seed, and git SHA (`src/utils/logging.py`).
- Updated backtest engine to emit artifacts (`src/backtest/engine.py`).
- Documented metrics catalog (`docs/dev/metrics_catalog.md`).
- Added run artifacts generation (summary.json, trades.parquet, config.json).

## References
- [Orchestrator Plan](docs/Orchestrator_Plan_ML_Risk_Diagnostics.md)
- [Rulebook](docs/rulebook.md)
- [Risk Controls](docs/risk_execution_controls.md)
- [Metrics Catalog](docs/dev/metrics_catalog.md)

**Next:** Phase 1.2 - Regime/Participation Gates (Quant & Research Studio mode).