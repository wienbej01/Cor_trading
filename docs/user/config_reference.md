# Configuration Reference

This document provides a reference for all configuration options available in the project. The configuration is split across two main files: `configs/pairs.yaml` for pair-specific settings and `configs/risk.yaml` for global risk management settings.

## `configs/pairs.yaml`

This file defines the trading pairs and their specific strategy parameters.

### Pair-Level Configuration

Each top-level key in this file represents a trading pair (e.g., `usdcad_wti`).

-   `fx_symbol` (str): The symbol for the FX instrument (e.g., "USDCAD=X").
-   `comd_symbol` (str): The symbol for the commodity instrument (e.g., "CL=F").
-   `inverse_fx_for_quote_ccy_strength` (bool): Whether to inverse the FX rate.

#### `lookbacks`

-   `beta_window` (int): Window for hedge ratio calculation.
-   `z_window` (int): Window for z-score calculation.
-   `corr_window` (int): Window for correlation calculation.

#### `thresholds`

-   `entry_z` (float): Base z-score for trade entry.
-   `exit_z` (float): Base z-score for trade exit.
-   `stop_z` (float): Z-score for stop loss.
-   `adf_p` (float): ADF test p-value threshold for cointegration.

#### `time_stop`

-   `max_days` (int): Maximum number of days to hold a position.

#### `regime`

-   `min_abs_corr` (float): Minimum absolute correlation for trading.
-   ... (other regime filter parameters)

#### `ml_filter`

-   `enabled` (bool): If `true`, the ML trade filter is active.
-   `model_path` (str): Path to the trained `.joblib` model file.
-   `threshold` (float): The probability threshold (0.0 to 1.0) for accepting a trade.
-   `strict` (bool): If `true`, trades below the threshold are blocked. If `false`, they are only logged.

#### `risk`

This section can override the global risk settings from `configs/risk.yaml` for a specific pair.

---

## `configs/risk.yaml`

This file defines the global risk management policies.

### `portfolio`

-   `max_net_exposure` (float): Maximum net exposure across all pairs.
-   `max_gross_exposure` (float): Maximum gross exposure across all pairs.

### `default_pair_policy`

-   `max_risk_per_trade` (float): Max % of portfolio equity to risk per trade.
-   `trade_cooldown` (int): Number of bars to wait after a trade.
-   `daily_loss_stop` (dict):
    -   `enabled` (bool): Enable/disable the daily loss stop.
    -   `threshold` (float): The loss threshold (e.g., 0.02 for 2%).
-   `max_drawdown_stop` (dict):
    -   `enabled` (bool): Enable/disable the max drawdown stop.
    -   `threshold` (float): The drawdown threshold.
    -   `lookback` (int): The lookback window for calculating the peak equity.

### `sizing`

-   `method` (str): The position sizing method (`inverse_volatility`, `kelly_criterion`, `fixed_fractional`).
-   `target_volatility` (float): Target volatility contribution for `inverse_volatility` sizing.
-   `max_position_size` (float): Maximum size of a single position as a fraction of equity.
-   `min_position_size` (float): Minimum size of a single position.
