"""
Diagnostic plotting functions for regime and feature analysis.
"""

from typing import Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def plot_regime_timeline(
    df: pd.DataFrame,
    price_col: str,
    regime_col: str = "feat_volatility_regime",
    title: str = "Regime Timeline",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot price series with regime overlay.

    Args:
        df: DataFrame with price and regime columns.
        price_col: Column name for price series.
        regime_col: Column name for regime series.
        title: Plot title.
        save_path: Optional path to save plot as PNG.
    """
    if price_col not in df.columns or regime_col not in df.columns:
        logger.warning(
            f"Missing columns for regime timeline plot: {price_col}, {regime_col}"
        )
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot price
    ax1.plot(df.index, df[price_col], color="black", linewidth=1, label=price_col)
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Plot regime as background color
    ax2 = ax1.twinx()
    regime_numeric = pd.Categorical(df[regime_col]).codes
    ax2.fill_between(df.index, regime_numeric, alpha=0.2, color="orange")
    ax2.set_yticks([])
    ax2.set_ylabel(regime_col, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    plt.title(title)
    ax1.legend(loc="upper left")
    ax2.legend([regime_col], loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved regime timeline plot to {save_path}")
    else:
        plt.show()


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    title: str = "Feature Correlation Heatmap",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot correlation heatmap of feature columns.

    Args:
        df: DataFrame with feature columns.
        feature_cols: List of feature column names. If None, uses all columns starting with 'feat_'.
        title: Plot title.
        save_path: Optional path to save plot as PNG.
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith("feat_")]

    if not feature_cols:
        logger.warning("No feature columns found for correlation heatmap")
        return

    # Select numeric columns only
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric feature columns found for correlation heatmap")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=True)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature correlation heatmap to {save_path}")
    else:
        plt.show()


def plot_feature_with_price(
    df: pd.DataFrame,
    price_col: str,
    feature_col: str,
    title: str = "Feature vs Price",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a feature series overlaid with price.

    Args:
        df: DataFrame with price and feature columns.
        price_col: Column name for price series.
        feature_col: Column name for feature series.
        title: Plot title.
        save_path: Optional path to save plot as PNG.
    """
    if price_col not in df.columns or feature_col not in df.columns:
        logger.warning(
            f"Missing columns for feature overlay plot: {price_col}, {feature_col}"
        )
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot price
    ax1.plot(df.index, df[price_col], color="black", linewidth=1, label=price_col)
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Plot feature on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        df.index,
        df[feature_col],
        color="blue",
        linewidth=1,
        alpha=0.7,
        label=feature_col,
    )
    ax2.set_ylabel(feature_col, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature overlay plot to {save_path}")
    else:
        plt.show()


def plot_signals_with_features(
    df: pd.DataFrame,
    price_col: str,
    signal_col: str = "signal",
    feature_cols: Optional[List[str]] = None,
    title: str = "Signals with Features",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot price, signals, and selected features.

    Args:
        df: DataFrame with price, signal, and feature columns.
        price_col: Column name for price series.
        signal_col: Column name for signal series.
        feature_cols: List of feature column names to plot.
        title: Plot title.
        save_path: Optional path to save plot as PNG.
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith("feat_")][
            :3
        ]  # Limit to first 3

    if price_col not in df.columns or signal_col not in df.columns:
        logger.warning(f"Missing columns for signals plot: {price_col}, {signal_col}")
        return

    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Plot price
    ax1.plot(df.index, df[price_col], color="black", linewidth=1, label=price_col)
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Plot buy/sell signals
    buy_signals = df[df[signal_col] == 1]
    sell_signals = df[df[signal_col] == -1]
    ax1.scatter(
        buy_signals.index,
        buy_signals[price_col],
        color="green",
        marker="^",
        alpha=0.7,
        label="Buy",
    )
    ax1.scatter(
        sell_signals.index,
        sell_signals[price_col],
        color="red",
        marker="v",
        alpha=0.7,
        label="Sell",
    )

    # Plot features on secondary axis
    ax2 = ax1.twinx()
    for col in feature_cols:
        if col in df.columns:
            ax2.plot(df.index, df[col], linewidth=1, alpha=0.7, label=col)

    ax2.set_ylabel("Features", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved signals with features plot to {save_path}")
    else:
        plt.show()
