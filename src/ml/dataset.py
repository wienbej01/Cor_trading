#!/usr/bin/env python3
"""
ML Dataset Builder for Trade Filter.

Generates time-ordered samples with features and labels for supervised learning.
Ensures no data leakage through proper temporal separation and embargo periods.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.yahoo_loader import download_and_align_pair
from features.spread import compute_spread
from features.regime import volatility_regime, trend_regime
from features.indicators import rolling_corr
from .schema import MLSchema, LabelSpec, FeatureSpec
from utils.logging import setup_logging
# Configuration manager (pairs -> symbols mapping)
from core.config import config as config

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds ML datasets for trade filter training."""

    def __init__(
        self,
        pair: str,
        schema: Optional[MLSchema] = None,
        min_history: int = 60,
        fx_path: Optional[str] = None,
        comd_path: Optional[str] = None,
    ):
        """
        Args:
            pair: logical pair key (used to resolve symbols from config)
            schema: MLSchema with LabelSpec and FeatureSpec
            min_history: minimum bars required per-signal for features
            fx_path: optional local CSV/Parquet path for FX prices (fallback before Yahoo)
            comd_path: optional local CSV/Parquet path for Commodity prices (fallback before Yahoo)
        """
        self.pair = pair
        self.schema = schema or MLSchema()
        self.data = None
        self.spread_data = None
        # Minimum bars of history required for feature computation per signal
        self.min_history = int(min_history)
        # Optional local file fallbacks
        self.fx_path = fx_path
        self.comd_path = comd_path

    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and prepare raw data for the pair."""
        logger.info(f"Loading data for {self.pair} from {start_date} to {end_date}")

        try:
            # Resolve symbol mapping for the logical pair name from configs
            pair_cfg = config.get_pair_config(self.pair)
            fx_symbol = pair_cfg.get('fx_symbol') or pair_cfg.get('fx') or pair_cfg.get('fx_ticker')
            comd_symbol = pair_cfg.get('comd_symbol') or pair_cfg.get('comd') or pair_cfg.get('comd_ticker')
 
            if not fx_symbol or not comd_symbol:
                raise ValueError(f"Pair configuration for '{self.pair}' missing 'fx_symbol' or 'comd_symbol'")
 
            # Download and align the two series (returns a DataFrame)
            aligned = download_and_align_pair(
                fx_symbol,
                comd_symbol,
                start_date,
                end_date,
                fx_local_path=self.fx_path,
                comd_local_path=self.comd_path,
            )
 
            # Aligned is a DataFrame with two columns: FX and commodity
            fx_series = aligned.iloc[:, 0].copy()
            comd_series = aligned.iloc[:, 1].copy()
            fx_series.name = fx_series.name or fx_symbol
            comd_series.name = comd_series.name or comd_symbol
 
            data = pd.DataFrame({
                'fx_price': fx_series,
                'comd_price': comd_series,
                'date': fx_series.index
            })
 
            # Safety checks: ensure we have enough aligned rows for feature engineering
            n_rows = len(aligned)
            min_history = 60  # minimum history required by feature engineering
            if n_rows < min_history:
                raise ValueError(
                    f"Insufficient aligned data for pair '{self.pair}': {n_rows} rows; "
                    f"at least {min_history} required. Expand the date range and retry."
                )
 
            # Compute spread - use pair-config lookback and kalman flag.
            # Ensure beta_window is not larger than available data and has a sensible floor.
            beta_window_cfg = int(pair_cfg.get('lookbacks', {}).get('beta_window', 90))
            use_kalman = bool(pair_cfg.get('use_kalman', True))
            effective_beta_window = max(10, min(beta_window_cfg, n_rows))
 
            try:
                spread, alpha, beta = compute_spread(
                    fx_series, comd_series, beta_window=effective_beta_window, use_kalman=use_kalman
                )
            except Exception as e:
                logger.error(
                    "compute_spread failed for pair '%s' (n_rows=%d, beta_window=%d, use_kalman=%s): %s",
                    self.pair, n_rows, effective_beta_window, use_kalman, e
                )
                raise RuntimeError(f"Failed to compute spread for pair '{self.pair}': {e}")
 
            data['spread'] = spread
            data['alpha'] = alpha
            data['beta'] = beta
 
            # Compute returns
            data['fx_returns'] = fx_series.pct_change()
            data['comd_returns'] = comd_series.pct_change()
            data['spread_returns'] = spread.pct_change()
 
            self.data = data
            # Diagnostic counters
            logger.info(f"total_rows_loaded={len(data)}")
            logger.info(f"Loaded {len(data)} data points")
            return data
 
        except Exception as e:
            logger.error(f"Failed to load data for pair '{self.pair}': {e}")
            raise

    def generate_candidate_signals(self, z_threshold: float = 2.0) -> pd.DataFrame:
        """Generate candidate trade signals based on z-score."""
        if self.data is None:
            raise ValueError("Data not loaded")

        logger.info("Generating candidate signals")

        # Compute z-score
        spread_mean = self.data['spread'].expanding().mean()
        spread_std = self.data['spread'].expanding().std()
        self.data['spread_z'] = (self.data['spread'] - spread_mean) / spread_std

        # Generate signals (long when z < -threshold, short when z > threshold)
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        signals.loc[self.data['spread_z'] <= -z_threshold, 'signal'] = 1  # Long spread
        signals.loc[self.data['spread_z'] >= z_threshold, 'signal'] = -1  # Short spread

        # Filter to actual signal changes
        signals['entry'] = (signals['signal'] != signals['signal'].shift(1)) & (signals['signal'] != 0)

        candidate_signals = signals[signals['entry']].copy()
        candidate_signals['entry_price'] = self.data.loc[candidate_signals.index, 'spread']

        logger.info(f"Generated {len(candidate_signals)} candidate signals")
        logger.info(f"candidate_signals_count={len(candidate_signals)}")
        if len(candidate_signals) == 0:
            logger.error("Zero-sample outcome: candidate_signals_count dropped to 0")
        return candidate_signals

    def compute_features(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for each signal."""
        logger.info("Computing features")
        logger.info(f"features_candidates={len(signals_df)}")

        if self.data is None:
            raise ValueError("Data not loaded")

        features_records = {}

        for idx in signals_df.index:
            try:
                # Get data up to this point (no lookahead)
                historical_data = self.data.loc[:idx].copy()

                if len(historical_data) < self.min_history:  # Minimum history required (configurable)
                    # Not enough history for this signal; skip
                    continue

                features = {}

                # Spread features
                spread = historical_data['spread']
                features['spread_z_20'] = (spread.iloc[-1] - spread.iloc[-20:].mean()) / spread.iloc[-20:].std()
                features['spread_z_60'] = (spread.iloc[-1] - spread.iloc[-60:].mean()) / spread.iloc[-60:].std()
                features['spread_momentum_5'] = spread.iloc[-1] - spread.iloc[-6]
                features['spread_momentum_20'] = spread.iloc[-1] - spread.iloc[-21]

                # Volatility features
                spread_returns = historical_data['spread_returns'].dropna()
                fx_returns = historical_data['fx_returns'].dropna()
                comd_returns = historical_data['comd_returns'].dropna()

                features['spread_vol_20'] = spread_returns.iloc[-20:].std() * np.sqrt(252)
                features['spread_vol_60'] = spread_returns.iloc[-60:].std() * np.sqrt(252)
                features['fx_vol_20'] = fx_returns.iloc[-20:].std() * np.sqrt(252)
                features['comd_vol_20'] = comd_returns.iloc[-20:].std() * np.sqrt(252)

                # Correlation features
                fx_prices = historical_data['fx_price']
                comd_prices = historical_data['comd_price']
                rc20 = rolling_corr(fx_prices, comd_prices, 20)
                rc60 = rolling_corr(fx_prices, comd_prices, 60)
                rc252 = rolling_corr(fx_prices, comd_prices, 252)

                features['rolling_corr_20'] = rc20.iloc[-1] if len(rc20) > 0 else float('nan')
                features['rolling_corr_60'] = rc60.iloc[-1] if len(rc60) > 0 else float('nan')

                corr_mean = rc252.expanding().mean().iloc[-1] if len(rc252) > 0 else 0.0
                corr_std = rc252.expanding().std().iloc[-1] if len(rc252) > 0 else 1.0
                features['corr_z_score'] = (features['rolling_corr_20'] - corr_mean) / (corr_std if corr_std != 0 else 1.0)

                # Regime features
                tr = trend_regime(fx_prices, use_roc_hp=True)
                vr = volatility_regime(fx_prices, use_quantiles=True)

                features['trend_regime'] = int(tr.iloc[-1]) if len(tr) > 0 else 0
                features['vol_regime'] = int(vr.iloc[-1]) if len(vr) > 0 else 1
                features['combined_regime'] = features['trend_regime'] + features['vol_regime']

                # Temporal features
                date = idx
                features['day_of_week'] = date.weekday()
                features['month_of_year'] = date.month
                features['quarter'] = (date.month - 1) // 3 + 1

                features_records[idx] = features

            except Exception as e:
                logger.warning(f"Failed to compute features for {idx}: {e}")
                continue

        if not features_records:
            logger.info("No features computed (insufficient history for all candidate signals)")
            logger.info("features_passing_min_history=0")
            logger.error("Zero-sample outcome: features_passing_min_history dropped to 0")
            return pd.DataFrame()

        features_df = pd.DataFrame.from_dict(features_records, orient='index')
        # Ensure proper ordering by timestamp/index
        features_df.index = pd.to_datetime(features_df.index)
        features_df = features_df.sort_index()
        logger.info(f"Computed features for {len(features_df)} signals")
        logger.info(f"features_passing_min_history={len(features_df)}")
        return features_df

    def create_labels(self, signals_df: pd.DataFrame) -> pd.Series:
        """Create labels by simulating trade outcomes."""
        logger.info("Creating labels")

        labels = []

        for idx in signals_df.index:
            try:
                entry_price = signals_df.loc[idx, 'entry_price']
                signal = signals_df.loc[idx, 'signal']

                # Get future spread data
                future_idx = self.data.index.get_loc(idx) + 1
                if future_idx + self.schema.label_spec.horizon_bars >= len(self.data):
                    labels.append(0)  # Insufficient future data
                    continue

                future_spread = self.data.iloc[future_idx:future_idx + self.schema.label_spec.horizon_bars]['spread']

                # Set stop loss and take profit based on volatility
                spread_vol = self.data.loc[:idx, 'spread_returns'].iloc[-20:].std()
                stop_distance = 2.0 * spread_vol * np.sqrt(self.schema.label_spec.horizon_bars)
                profit_distance = 1.5 * stop_distance

                if signal == 1:  # Long spread
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + profit_distance
                else:  # Short spread
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - profit_distance

                # Simulate outcome
                label = self.schema.create_label(
                    future_spread, entry_price, stop_loss, take_profit
                )
                labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to create label for {idx}: {e}")
                labels.append(0)

        labels_series = pd.Series(labels, index=signals_df.index)
        logger.info(f"Created {labels_series.sum()} positive labels out of {len(labels_series)}")
        logger.info(f"labels_created={len(labels_series)}")
        return labels_series

    def apply_temporal_filters(self, features_df: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply embargo and purge filters to prevent data leakage."""
        logger.info("Applying temporal filters")

        # Embargo: Remove samples within embargo period of each other
        valid_indices = []
        last_label_idx = None

        for idx in features_df.index:
            if last_label_idx is None or (idx - last_label_idx).days >= self.schema.label_spec.embargo_bars:
                valid_indices.append(idx)
                if labels.loc[idx] == 1:  # Only embargo after positive labels
                    last_label_idx = idx
        logger.info(f"after_embargo={len(valid_indices)}")
        if len(valid_indices) == 0:
            logger.error("Zero-sample outcome: after_embargo dropped to 0")

        # Purge: Remove overlapping samples within purge window
        final_indices = []
        for i, idx in enumerate(valid_indices):
            overlap = False
            for j in range(max(0, i - self.schema.label_spec.purge_window // 5), i):
                if (idx - valid_indices[j]).days < self.schema.label_spec.purge_window:
                    overlap = True
                    break
            if not overlap:
                final_indices.append(idx)

        filtered_features = features_df.loc[final_indices]
        filtered_labels = labels.loc[final_indices]

        logger.info(f"Temporal filtering: {len(filtered_features)} samples remaining")
        logger.info(f"after_purge={len(filtered_features)}")
        if len(filtered_features) == 0:
            logger.error("Zero-sample outcome: after_purge dropped to 0")
        return filtered_features, filtered_labels

    def balance_classes(self, features_df: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes to target distribution with safety for edge cases."""
        logger.info("Balancing classes")

        pos_indices = labels[labels == 1].index
        neg_indices = labels[labels == 0].index

        pos_count = int(len(pos_indices))
        neg_count = int(len(neg_indices))
        logger.info(f"Class counts before balancing: pos={pos_count}, neg={neg_count}")

        # If a class is absent, skip balancing to avoid empty dataset
        if pos_count == 0 or neg_count == 0:
            logger.warning("One class absent (pos=%d, neg=%d). Skipping class balancing.", pos_count, neg_count)
            # Ensure chronological order
            features_df = features_df.sort_index()
            labels = labels.sort_index()
            return features_df, labels

        # Target negative count to reach desired positive ratio
        target_pos_ratio = 0.35  # 35% positives desired
        computed_neg = int(round(pos_count / target_pos_ratio - pos_count))
        target_neg_count = max(1, min(neg_count, computed_neg))

        # Undersample negatives
        if neg_count > target_neg_count:
            neg_sample = np.random.choice(neg_indices, size=target_neg_count, replace=False)
        else:
            neg_sample = neg_indices

        balanced_indices = np.concatenate([pos_indices, neg_sample])
        # Maintain chronological order
        balanced_indices = np.sort(balanced_indices)

        balanced_features = features_df.loc[balanced_indices]
        balanced_labels = labels.loc[balanced_indices]

        logger.info(
            "Balanced dataset: pos=%d, neg=%d, total=%d",
            int(balanced_labels.sum()),
            int(len(balanced_labels) - balanced_labels.sum()),
            int(len(balanced_labels)),
        )
        logger.info(f"after_balance={len(balanced_labels)}")
        if len(balanced_labels) == 0:
            logger.error("Zero-sample outcome: after_balance dropped to 0")
        return balanced_features, balanced_labels

    def build_dataset(self, start_date: str, end_date: str, z_threshold: float = 2.0) -> Tuple[pd.DataFrame, pd.Series]:
        """Build complete dataset."""
        # Load data
        self.load_data(start_date, end_date)

        if self.data is None or len(self.data) == 0:
            logger.error("Zero-sample outcome: total_rows_loaded dropped to 0")
            raise ValueError(f"No data available for pair '{self.pair}' in the provided date range")

        # Generate signals
        signals = self.generate_candidate_signals(z_threshold)
        if signals.empty:
            logger.error("Zero-sample outcome: candidate_signals_count dropped to 0")
            raise ValueError(
                f"No candidate signals generated for pair '{self.pair}'. "
                f"Available rows: {len(self.data)}. Try expanding the date range or reducing the z_threshold."
            )

        # Compute features
        features = self.compute_features(signals)
        if features.empty:
            logger.error("Zero-sample outcome: features_passing_min_history dropped to 0")
            raise ValueError(
                f"No features computed for pair '{self.pair}'. "
                f"Candidate signals: {len(signals)}. Feature engineering requires >=60 bars history per signal."
            )

        # Create labels (create for all signals, then align to features we computed)
        labels_all = self.create_labels(signals)
        labels = labels_all.reindex(features.index).fillna(0).astype(int)

        # Apply temporal filters
        features, labels = self.apply_temporal_filters(features, labels)
        if len(features) == 0:
            logger.error("Zero-sample outcome: after_purge/after_embargo dropped to 0")
            raise ValueError("No samples remaining after temporal filtering")

        # Balance classes
        features, labels = self.balance_classes(features, labels)

        # Validate
        self.schema.validate_features(features)

        logger.info(f"Final dataset: {len(features)} samples, {labels.mean():.2%} positive class")
        return features, labels

    def save_dataset(self, features: pd.DataFrame, labels: pd.Series, output_dir: str):
        """Save dataset to train/val splits with guards for small datasets."""
        os.makedirs(output_dir, exist_ok=True)

        n = int(len(features))
        if n == 0:
            raise ValueError("No samples to save in dataset")

        # Keep existing behavior when n == 1 (train=1, val=0)
        if n == 1:
            split_idx = 1
        else:
            # Compute initial split (80/20 time-ordered)
            split_idx = int(n * 0.8)
            # Protective branch: guarantee at least 1 row in both splits when n >= 2
            if split_idx == 0 or split_idx == n:
                adjusted = max(1, min(n - 1, split_idx))
                logger.warning(
                    "Adjusted split index from %d to %d to ensure non-empty train/val (n=%d)",
                    split_idx, adjusted, n
                )
                split_idx = adjusted

        train_features = features.iloc[:split_idx]
        train_labels = labels.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_labels = labels.iloc[split_idx:]

        # Save to parquet
        train_df = train_features.copy()
        train_df['label'] = train_labels
        train_df.to_parquet(os.path.join(output_dir, 'train.parquet'))

        val_df = val_features.copy()
        # Ensure label column exists even if empty
        val_df['label'] = val_labels
        val_df.to_parquet(os.path.join(output_dir, 'val.parquet'))

        # Save scaler only if we have at least 1 training sample
        if len(train_features) > 0:
            scaler = StandardScaler()
            _ = scaler.fit_transform(train_features)
            joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        else:
            logger.warning("No training samples available; skipping scaler export")

        logger.info(f"Saved dataset to {output_dir}")
        logger.info(f"Train: {len(train_features)} samples")
        logger.info(f"Val: {len(val_features)} samples")


def synth_bootstrap(n: int, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a synthetic bootstrap dataset with feature columns consistent with compute_features output
    and a binary label with ~35% positives.

    This is used only as a last-resort fallback when real data yields zero samples and --use-synth is passed.
    Deterministic via numpy RandomState(random_state). Index aligned between X and y.
    """
    rng = np.random.RandomState(seed=random_state)

    # Reasonable distributions and correlations
    df = pd.DataFrame({
        "spread_z_20": rng.normal(0, 1, n),
        "spread_z_60": rng.normal(0, 1, n),
        "spread_momentum_5": rng.normal(0, 0.5, n),
        "spread_momentum_20": rng.normal(0, 1.0, n),
        "spread_vol_20": np.abs(rng.normal(0.02, 0.01, n)) * np.sqrt(252),
        "spread_vol_60": np.abs(rng.normal(0.02, 0.01, n)) * np.sqrt(252),
        "fx_vol_20": np.abs(rng.normal(0.01, 0.005, n)) * np.sqrt(252),
        "comd_vol_20": np.abs(rng.normal(0.015, 0.007, n)) * np.sqrt(252),
        "rolling_corr_20": np.clip(rng.normal(0.2, 0.3, n), -1, 1),
        "rolling_corr_60": np.clip(rng.normal(0.2, 0.3, n), -1, 1),
    })
    # corr_z_score derived roughly from rolling_corr_20 vs noisy long window stats
    corr_mean = 0.2
    corr_std = 0.25
    df["corr_z_score"] = (df["rolling_corr_20"] - corr_mean) / (corr_std if corr_std != 0 else 1.0)

    # Regimes as categorical integers
    df["trend_regime"] = rng.randint(-1, 2, n)  # -1,0,1 (high exclusive)
    df["vol_regime"] = rng.randint(0, 3, n)     # 0,1,2
    df["combined_regime"] = df["trend_regime"] + df["vol_regime"]

    # Temporal features with deterministic business day index
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    df["day_of_week"] = idx.weekday
    df["month_of_year"] = idx.month
    df["quarter"] = ((idx.month - 1) // 3) + 1
    df.index = idx

    # Labels ~35% positives, aligned to index
    positives = rng.rand(n) < 0.35
    y = pd.Series(positives.astype(int), index=df.index)

    return df, y


def main():
    parser = argparse.ArgumentParser(description="Build ML dataset for trade filter")
    parser.add_argument('--pair', required=True, help='Trading pair')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--h', type=int, default=20, help='Label horizon in bars')
    parser.add_argument('--rr', type=float, default=1.5, help='Risk-reward threshold')
    parser.add_argument('--z-threshold', type=float, default=2.0, help='Entry z-score threshold for candidate signals')
    parser.add_argument('--min-history', type=int, default=60, help='Minimum bars of history required per signal for features')
    parser.add_argument('--embargo-bars', type=int, default=5, help='Embargo window (bars) applied after positive labels')
    parser.add_argument('--purge-window', type=int, default=20, help='Purge window (bars) to avoid overlapping label horizons')
    parser.add_argument('--output', default='data/ml', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Build dataset but do not save parquet files (debug mode)')
    # New optional CLI args for local data and synthetic fallback
    parser.add_argument('--fx-path', default=None, help='Optional local CSV/Parquet path for FX prices (fallback before Yahoo)')
    parser.add_argument('--comd-path', default=None, help='Optional local CSV/Parquet path for Commodity prices (fallback before Yahoo)')
    parser.add_argument('--use-synth', action='store_true', help='Allow synthetic bootstrap if real-data pipeline yields 0 samples')
    parser.add_argument('--synth-n', type=int, default=500, help='Number of synthetic samples to generate when --use-synth is set')

    args = parser.parse_args()

    setup_logging()

    # Update schema with args (leakage controls configurable from CLI)
    label_spec = LabelSpec(
        horizon_bars=args.h,
        rr_threshold=args.rr,
        embargo_bars=args.embargo_bars,
        purge_window=args.purge_window,
    )
    schema = MLSchema(label_spec=label_spec)

    # Build dataset (real-data pipeline)
    features: Optional[pd.DataFrame] = None
    labels: Optional[pd.Series] = None

    builder = DatasetBuilder(
        args.pair,
        schema,
        min_history=args.min_history,
        fx_path=args.fx_path,
        comd_path=args.comd_path,
    )

    real_pipeline_error: Optional[Exception] = None
    try:
        features, labels = builder.build_dataset(args.start, args.end, z_threshold=args.z_threshold)
    except Exception as e:
        real_pipeline_error = e
        logger.error(f"Real-data pipeline failed: {e}")

    f_n = 0 if features is None else int(len(features))
    l_n = 0 if labels is None else int(len(labels))

    # Synthetic bootstrap fallback when enabled and real-data pipeline yields empty features or labels
    if f_n == 0 or l_n == 0:
        logger.info("Final dataset from real-data pipeline is empty "
                    f"(features={f_n}, labels={l_n})")
        if args.use_synth:
            seed = schema.seed  # deterministic seed for reproducibility
            logger.warning(f"Using synthetic bootstrap fallback (--use-synth). seed={seed}, n={args.synth_n}")
            features, labels = synth_bootstrap(n=args.synth_n, random_state=seed)

            # Verify alignment and non-empty
            if features is None or labels is None:
                logger.error("Synthetic fallback returned None for features or labels")
                sys.exit(1)
            if not isinstance(labels, pd.Series):
                labels = pd.Series(labels, index=features.index)
            if features.shape[0] != labels.shape[0] or features.shape[0] == 0:
                logger.error("Synthetic fallback failed to produce aligned, non-empty dataset "
                             f"(features={features.shape[0]}, labels={labels.shape[0]})")
                sys.exit(1)

            logger.info(f"Synthetic dataset generated: total={len(labels)}, positives={int(labels.sum())}")
        else:
            # Preserve previous exception if any for context
            msg = "Final sample count is 0. Re-run with --use-synth to enable synthetic bootstrap."
            if real_pipeline_error is not None:
                raise RuntimeError(f"{msg} Root cause: {real_pipeline_error}") from real_pipeline_error
            raise RuntimeError(msg)

    # Dry-run mode: summarize and exit without writing files
    if args.dry_run:
        logger.info(f"Dry-run complete for pair='{args.pair}'")
        logger.info(f"Final features: {0 if features is None else len(features)} samples")
        if features is not None and len(features) > 0:
            logger.info(f"Feature columns: {features.columns.tolist()}")
            logger.info(f"Sample feature head:\n{features.head().to_string()}")
        logger.info(f"Final labels: {len(labels)} samples, positives={int(labels.sum()) if len(labels)>0 else 0}")
        return

    # Final guard before saving: ensure non-empty and aligned
    if features is None or labels is None:
        logger.error("features or labels is None before saving - aborting")
        sys.exit(1)
    if len(features) == 0 or len(labels) == 0:
        logger.error("features or labels has 0 rows before saving - aborting")
        sys.exit(1)
    if features.shape[0] != labels.shape[0]:
        logger.error(f"features/labels size mismatch before saving: X={features.shape[0]}, y={labels.shape[0]}")
        sys.exit(1)

    # Save
    output_dir = os.path.join(args.output, args.pair)
    builder.save_dataset(features, labels, output_dir)
