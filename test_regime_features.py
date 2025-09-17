#!/usr/bin/env python3
"""
Test script for regime and feature expansion module.
Demonstrates usage and generates diagnostic plots.
"""

import sys
import os
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data.yahoo_loader import download_and_align_pair
from src.features.regime_expansion import compute_regime_and_features
from src.features.diagnostics import (
    plot_regime_timeline,
    plot_feature_correlation_heatmap,
    plot_feature_with_price,
    plot_signals_with_features,
)


def load_config(config_path: str = "configs/pairs.yaml") -> dict:
    """Load pair configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main test function."""
    logger.info("Starting regime and feature expansion test")

    # Load configuration
    config = load_config()
    pair_config = config["usdcad_wti"]

    # Download data
    logger.info("Downloading data for USDCAD-WTI pair")
    df = download_and_align_pair(
        fx_symbol=pair_config["fx_symbol"],
        comd_symbol=pair_config["comd_symbol"],
        start="2020-01-01",
        end="2023-12-31",
        fx_name="fx_price",
        comd_name="comd_price",
    )

    # Compute regimes and features
    logger.info("Computing regimes and features")
    df_with_features = compute_regime_and_features(df, pair_config, lookahead_shift=1)

    # Save example data
    output_path = "test_output.csv"
    df_with_features.to_csv(output_path)
    logger.info(f"Saved data with features to {output_path}")

    # Generate diagnostic plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Regime timeline
    plot_regime_timeline(
        df_with_features,
        price_col="comd_price",
        regime_col="feat_volatility_regime",
        title="WTI Volatility Regime Timeline",
        save_path=os.path.join(plots_dir, "volatility_regime_timeline.png"),
    )

    # 2. Feature correlation heatmap
    plot_feature_correlation_heatmap(
        df_with_features,
        title="Feature Correlation Heatmap",
        save_path=os.path.join(plots_dir, "feature_correlation_heatmap.png"),
    )

    # 3. Feature with price overlay
    plot_feature_with_price(
        df_with_features,
        price_col="comd_price",
        feature_col="feat_corr_rolling",
        title="Rolling Correlation with WTI Price",
        save_path=os.path.join(plots_dir, "corr_with_price.png"),
    )

    # 4. Signals with features (using raw signal for demo)
    df_with_features["signal"] = df_with_features["feat_corr_rolling"].apply(
        lambda x: 1 if x > 0.7 else (-1 if x < 0.3 else 0)
    )
    plot_signals_with_features(
        df_with_features,
        price_col="comd_price",
        signal_col="signal",
        feature_cols=["feat_corr_rolling", "feat_volatility_value"],
        title="Sample Signals with Features",
        save_path=os.path.join(plots_dir, "signals_with_features.png"),
    )

    logger.success("Test completed successfully")


if __name__ == "__main__":
    main()
