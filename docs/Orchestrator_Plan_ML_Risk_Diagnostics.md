

# Docs/Orchestrator\_Plan\_ML\_Risk\_Diagnostics.md

# Orchestrator Plan — Diagnostics, ML Trade Filter, Risk & Robustness

**Branch:** `ML-enhancement` (already created)
**Entry Mode:** `orchestrator-router-trading`
**Target Modes:** `quant-research-studio`, `engineering-platform`, `risk-execution-controls`, `testing-analytics-hub`, `operations-sre`

## Non-Negotiables (inherit from Router)

* No lookahead; purged/embargoed splits for ML.
* Always include costs & slippage; no same-bar fills.
* Economic rationale first; parsimony in features.
* Deterministic with seed; explicit configs; audit trail.
* Update `SYSTEM_TECH_DOC.md` on every change (version, date, commit).

---

## Phase 0 — Project Setup & Safety Rails (All Tracks)

**P0.1 – Ensure branch & housekeeping**
**Mode:** engineering-platform
**Files:** `docs/CHANGELOG.md`, `SYSTEM_TECH_DOC.md`
**Steps:**

1. Confirm current branch is `ML-enhancement`; if not: `git checkout ML-enhancement`.
2. Append CHANGELOG: “Start diagnostics + ML filter + risk program.”
3. Open `SYSTEM_TECH_DOC.md` section “Program Overview – ML/Risk/Diagnostics”.

**Accept:** Branch is `ML-enhancement`; docs updated; repo clean.

**P0.2 – Minimal CI/QA smoke**
**Mode:** engineering-platform → testing-analytics-hub
**Files:** `pyproject.toml`, `requirements-dev.txt`, `scripts/qa/smoke.sh`, `.github/workflows/ci.yml`
**Steps:**

1. Add/complete `pyproject.toml` entries for `pytest`, `ruff`, `mypy`.
2. Create `requirements-dev.txt` (sklearn, joblib, matplotlib if missing).
3. Create `scripts/qa/smoke.sh` to run: install, `pytest -q`, `ruff check .`, `mypy src`.
4. Add GH Actions workflow to run smoke on push to `ML-enhancement`.

**Accept:** Smoke passes or failures documented in `Docs/reports/ci_baseline.md`.
**Artifacts:** CI log link; `Docs/reports/ci_baseline.md`.
**Handoff:** testing-analytics-hub back to orchestrator.

---

## Phase 1 — Diagnostic Improvements (Track 1)

**Goal:** Make every run emit uniform, comparable metrics; add regime/participation gates.

**P1.1 – Unified Metrics & Run Artifacts**
**Mode:** engineering-platform
**Files:**

* `src/backtest/metrics.py` (new)
* `src/utils/logging.py` (extend with run\_id, seed, git SHA)
* `src/backtest/engine.py` (emit artifacts)
* `docs/dev/metrics_catalog.md` (new)
  **Steps:**

1. Implement metrics: equity stats, trade stats, per-day stats, cost/slippage tracking.
2. At end of run, write to `reports/<pair>/<run_id>/{summary.json,trades.parquet,config.json}`.
3. Document fields in `docs/dev/metrics_catalog.md`.

**Commands:**

```bash
python src/run_backtest.py run --pair usdcad_wti --start 2018-01-01 --end 2025-08-15 --seed 42
```

**Accept:** Files above produced with correct schema; log includes run\_id/seed/git SHA.
**Artifacts:** `reports/.../summary.json`, `trades.parquet`.

**P1.2 – Regime/Participation Gates** ✅ Completed
**Mode:** quant-research-studio → engineering-platform
**Files:**

* `src/features/regime.py` (enhanced: ROC/HP-slope for trend; realized-vol quantiles for vol)
* `src/strategy/filters.py` (new: `AllowTradeContext`)
* `src/strategy/mean_reversion.py` (wire pre-entry gate)
* `configs/pairs.yaml` (add regime params & z-score by regime)
  **Steps:**

1. Design regimes (ROC/HP-slope for trend; realized-vol quantiles for vol).
2. Implement `AllowTradeContext(regime, vol, time, liquidity) -> bool`.
3. Gate entries inside mean-reversion strategy; all thresholds config-driven.

**Commands:**

```bash
python src/run_backtest.py run --pair usdnok_brent --start 2015-01-01 --end 2025-08-15 --seed 42
```

**Accept:** Trades suppressed in disallowed regimes; config toggles reflected in logs. Verified: 0 trades due to strict regime gates; logs show rejections and config application.

**P1.3 – Bucketed & Regime Reports**
**Mode:** testing-analytics-hub
**Files:** `scripts/reports/bucketed.py` (new), `Docs/bucketed_performance.md`
**Steps:**

1. Aggregate `reports/*/*/summary.json` by vol/trend buckets & 2-year windows.
2. Emit markdown with key distributional metrics & plots (saved to `reports/plots/...`).

**Commands:**

```bash
python scripts/reports/bucketed.py --input reports --out Docs/bucketed_performance.md
```

**Accept:** Updated markdown & plots; shows participation changes and PF/Sharpe by bucket.
**Handoff:** testing-analytics-hub → orchestrator for review checkpoint.

---

## Phase 2 — Supervised ML Trade Filter (Track 2)

**Goal:** Learn a *gate* that predicts spread reversion quality; integrate as a pre-trade filter.

**P2.1 – Label & Dataset Spec (No Leakage)**
**Mode:** quant-research-studio
**Files:** `src/ml/schema.py` (label/feature spec), `docs/dev/ml_labeling.md`
**Steps:**

1. Define label: “Revert to mean within H bars with RR≥X & max adverse excursion ≤ Y.”
2. Specify purge/embargo; define features (parsimony, economic rationale).
3. Document label timing, leakage defenses, and class balance choices.

**Accept:** Spec complete; approved by Router; proceed to build.

**P2.2 – Dataset Builder**
**Mode:** engineering-platform
**Files:** `src/ml/dataset.py` (new)
**Steps:**

1. Build time-ordered samples per opportunity; write `data/ml/<pair>/{train,val}.parquet`.
2. Support date ranges, class balancing, and feature/label integrity checks.

**Commands:**

```bash
python -m src.ml.dataset --pair usdcad_wti --start 2015-01-01 --end 2024-12-31 --h 20 --rr 1.5
```

**Accept:** Parquet datasets created; integrity checks pass.

**P2.3 – Baseline Classifier & Training Loop**
**Mode:** engineering-platform
**Files:** `src/ml/models.py`, `src/ml/train.py` (CLI), `artifacts/models/...`
**Steps:**

1. Implement `HistGradientBoostingClassifier` baseline with calibrated probs.
2. Time-series CV (k-fold), metrics: ROC-AUC, PR-AUC, Brier, calibration plot.
3. Persist `model.joblib`, `metrics.json`, and plots under `artifacts/models/<pair>/<ts>/`.

**Commands:**

```bash
python src/ml/train.py --pair usdcad_wti --cv 5 --seed 42 --out artifacts/models/usdcad_wti
```

**Accept:** Artifacts present; CV metrics logged and saved.

**P2.4 – Inference Path in Strategy**
**Mode:** engineering-platform
**Files:** `src/strategy/ml_filter.py` (new), `src/strategy/mean_reversion.py` (wire), `configs/pairs.yaml` (ml section)
**Steps:**

1. Implement `TradeFilter(model_path, threshold).accept(context)->bool`.
2. Call filter pre-order; bypass if `ml_filter.enabled=false`.
3. Config: `ml_filter.enabled`, `threshold`, `model_path`, `strict`.

**Accept:** Backtest runs with/without ML filter via config toggle.

**P2.5 – Impact Study**
**Mode:** testing-analytics-hub
**Files:** `scripts/reports/compare_runs.py`, `Docs/reports/ml_filter_impact.md`
**Steps:**

1. Run baseline vs ML-gated backtests with same seeds/horizons.
2. Compare PF, Sharpe, DD, win-rate, turnover, slippage sensitivity.
3. Emit markdown with results & links to run artifacts.

**Commands:**

```bash
python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15 --seed 42 --use-ml-filter false
python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15 --seed 42 --use-ml-filter true --ml-threshold 0.62 --ml-model artifacts/models/usdcad_wti/model.joblib
python scripts/reports/compare_runs.py --runs reports/.../baseline/summary.json reports/.../ml/summary.json --out Docs/reports/ml_filter_impact.md
```

**Accept:** Report shows reduced false positives without excessive turnover loss.
**Handoff:** testing-analytics-hub → orchestrator for review checkpoint.

---

## Phase 3 — Risk Management & Robustness (Track 4)

**Goal:** Centralize risk policy; enforce caps/cooldowns/halts; validate via stress & walk-forward.

**P3.1 – Risk Policy & Sizing**
**Mode:** risk-execution-controls
**Files:** `src/risk/policy.py` (new), `src/risk/sizing.py` (new), `configs/risk.yaml` (new)
**Steps:**

1. Define: max per-pair risk (ATR-scaled), portfolio net exposure caps, trade cooldown, daily loss stop, MDD cap, slippage model.
2. Implement inverse-vol sizing with floors/ceilings; optional Kelly cap.
3. Configurable via `configs/risk.yaml` + per-pair overrides.

**Accept:** Policy/sizing loadable; unit tests for each rule.

**P3.2 – Integration into Execution**
**Mode:** engineering-platform
**Files:** `src/strategy/mean_reversion.py`, `src/backtest/engine.py`, `src/risk/exits.py` (new)
**Steps:**

1. Enforce risk checks pre-entry & pre-order; intraday halts on daily-loss breach.
2. Implement ATR-based stop & optional trailing; simulate fills with slippage.

**Accept:** Risk events logged; orders blocked/downsized appropriately.

**P3.3 – Robustness Suite**
**Mode:** testing-analytics-hub
**Files:** `tests/test_risk_policy.py`, `Docs/reports/risk_stress_results.md`
**Steps:**

1. Property tests: size caps, cooldowns, daily loss stops.
2. Stress tests: +25% costs, 2× slippage, widened spread std.
3. Walk-forward yearly retrain for ML; OOS evaluation with embargo.

**Commands:**

```bash
pytest -q
python scripts/reports/run_stress_pack.py --out Docs/reports/risk_stress_results.md
```

**Accept:** Tests pass; stress results documented with pass/fail notes.
**Handoff:** testing-analytics-hub → orchestrator for review checkpoint.

---

## Phase 4 — Integration, Docs & Golden Runs

**P4.1 – Docs & Config Reference**
**Mode:** engineering-platform → operations-sre
**Files:** `README.md`, `docs/user/config_reference.md`, `docs/runbooks/backtest_runbook.md`
**Steps:** Update Quick Start (ML filter + risk), reference configs, add runbook.

**P4.2 – Golden Backtests**
**Mode:** testing-analytics-hub
**Steps:** Run canonical set on 4 pairs (2015–2025) for: baseline, +ML, +Risk, both.
**Artifacts:** `Docs/reports/golden_runs_<date>.md`, links to `reports/…`.

**P4.3 – Merge & Tag**
**Mode:** engineering-platform
**Steps:** PRs from `ML-enhancement` to `main` per phase; tag `v0.4.0-ml-risk-diagnostics`; release notes.

---

## Handoff Protocol (per task)

* **Entry check:** branch, deps, configs present.
* **Run self-tests:** `bash scripts/qa/smoke.sh`.
* **Artifacts:** write under `reports/` or `artifacts/` with `run_id`.
* **Commit style:** conventional commits (`feat(ml)`, `feat(risk)`, `chore(ci)`).
* **PR checklist:** smoke green; configs documented; artifacts linked; leakage tests included; backward-compatible defaults.

---

## Quick Command Reference

```bash
# Ensure branch
git checkout ML-enhancement

# Smoke
bash scripts/qa/smoke.sh

# Build ML dataset + train
python -m src.ml.dataset --pair usdcad_wti --start 2015-01-01 --end 2024-12-31 --h 20 --rr 1.5
python src/ml/train.py --pair usdcad_wti --cv 5 --seed 42 --out artifacts/models/usdcad_wti

# Backtests
python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15 --seed 42
python src/run_backtest.py run --pair usdcad_wti --start 2015-01-01 --end 2025-08-15 --seed 42 \
  --use-ml-filter true --ml-threshold 0.62 --ml-model artifacts/models/usdcad_wti/model.joblib \
  --risk-config configs/risk.yaml

# Reports
python scripts/reports/bucketed.py --input reports --out Docs/bucketed_performance.md
python scripts/reports/compare_runs.py --runs <baseline_summary.json> <ml_summary.json> --out Docs/reports/ml_filter_impact.md
python scripts/reports/run_stress_pack.py --out Docs/reports/risk_stress_results.md
```

---

