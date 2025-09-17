"""
Unit tests for regime and feature expansion module.
"""

import unittest
import pandas as pd
import numpy as np
from src.features.regime_expansion import (
    _volatility_regime,
    _commodity_cycle_phase,
    _compute_vpa,
    compute_regime_and_features,
)


class TestRegimeFeatures(unittest.TestCase):
    """Test suite for regime and feature functions."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic price series
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        self.price_series = pd.Series(prices, index=dates, name="price")

        # Create synthetic inventory series
        inventory = 1000 + np.cumsum(np.random.normal(0, 10, 100))
        self.inventory_series = pd.Series(inventory, index=dates, name="inventory")

    def test_volatility_regime(self):
        """Test volatility regime classification."""
        regime, vol_value = _volatility_regime(self.price_series, vol_window=10)

        # Check that outputs have correct shape
        self.assertEqual(len(regime), len(self.price_series))
        self.assertEqual(len(vol_value), len(self.price_series))

        # Check that regime values are valid categories
        valid_regimes = {"low", "normal", "high"}
        self.assertTrue(all(r in valid_regimes for r in regime.dropna().unique()))

        # Check that volatility values are numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(vol_value))

    def test_commodity_cycle_phase(self):
        """Test commodity cycle phase detection."""
        cycle, inv_delta = _commodity_cycle_phase(
            self.price_series, self.inventory_series, trend_window=10
        )

        # Check that outputs have correct shape
        self.assertEqual(len(cycle), len(self.price_series))
        self.assertEqual(len(inv_delta), len(self.price_series))

        # Check that cycle values are valid categories
        valid_cycles = {"expansion", "peak", "contraction", "trough"}
        self.assertTrue(all(c in valid_cycles for c in cycle.dropna().unique()))

        # Check that inventory delta is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(inv_delta))

    def test_commodity_cycle_phase_no_inventory(self):
        """Test commodity cycle phase detection with no inventory data."""
        cycle, inv_delta = _commodity_cycle_phase(
            self.price_series, None, trend_window=10
        )

        # Check that outputs have correct shape
        self.assertEqual(len(cycle), len(self.price_series))
        self.assertEqual(len(inv_delta), len(self.price_series))

        # Check that cycle values are "unknown" when no inventory
        self.assertTrue(all(c == "unknown" for c in cycle.dropna().unique()))

        # Check that inventory delta is NaN when no inventory
        self.assertTrue(inv_delta.isna().all())

    def test_compute_vpa(self):
        """Test volume-price analysis features."""
        # With volume data
        volume = pd.Series(
            np.random.uniform(1000, 2000, 100), index=self.price_series.index
        )
        vwret, vol_spike, vol_adj_ret = _compute_vpa(
            self.price_series, volume, window=10
        )

        # Check that outputs have correct shape
        self.assertEqual(len(vwret), len(self.price_series))
        self.assertEqual(len(vol_spike), len(self.price_series))
        self.assertEqual(len(vol_adj_ret), len(self.price_series))

        # Check that volume spike is boolean
        self.assertTrue(pd.api.types.is_bool_dtype(vol_spike))

        # Check that outputs are numeric (except vol_spike)
        self.assertTrue(pd.api.types.is_numeric_dtype(vwret))
        self.assertTrue(pd.api.types.is_numeric_dtype(vol_adj_ret))

        # Without volume data (should use proxy)
        vwret_proxy, vol_spike_proxy, vol_adj_ret_proxy = _compute_vpa(
            self.price_series, None, window=10
        )

        # Check that outputs have correct shape
        self.assertEqual(len(vwret_proxy), len(self.price_series))
        self.assertEqual(len(vol_spike_proxy), len(self.price_series))
        self.assertEqual(len(vol_adj_ret_proxy), len(self.price_series))

    def test_compute_regime_and_features(self):
        """Test main API function."""
        # Create simple DataFrame with price columns
        df = pd.DataFrame(
            {
                "fx_price": self.price_series,
                "comd_price": self.price_series * 1.5,  # Different price level
            }
        )

        # Simple config
        config = {
            "lookbacks": {
                "vol_window": 10,
                "corr_window": 10,
                "beta_window": 10,
            },
            "regime_features": {
                "volatility_regime": True,
                "commodity_cycle": True,
                "vix_overlay": False,  # Disable to avoid external API calls
                "yield_curve_overlay": False,
                "vpa": True,
                "liquidity_sweep": False,  # Disable to avoid needing OHLC
                "trend_filters": True,
                "correlation_features": True,
            },
        }

        # Compute features
        result = compute_regime_and_features(df, config, lookahead_shift=1)

        # Check that result has original columns
        self.assertIn("fx_price", result.columns)
        self.assertIn("comd_price", result.columns)

        # Check that feature columns are present
        expected_features = [
            "feat_volatility_regime",
            "feat_volatility_value",
            "feat_commodity_cycle",
            "feat_inventory_delta",
            "feat_vpa_vwret",
            "feat_vpa_volume_spike",
            "feat_vpa_vol_adj_ret",
            "feat_corr_rolling",
            "feat_beta_std",
        ]

        for feature in expected_features:
            self.assertIn(feature, result.columns)

        # Check that categorical columns are object type
        self.assertEqual(result["feat_volatility_regime"].dtype, "object")
        self.assertEqual(result["feat_commodity_cycle"].dtype, "object")

        # Check that features are shifted by lookahead_shift
        # (This is a simplified check - in practice, would need more sophisticated test)
        self.assertTrue(
            result["feat_volatility_regime"].iloc[0] is None
            or pd.isna(result["feat_volatility_regime"].iloc[0])
        )


if __name__ == "__main__":
    unittest.main()
