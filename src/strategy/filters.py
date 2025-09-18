"""
Trade filters module for mean reversion strategy.
Implements AllowTradeContext gate for regime/participation filtering.
"""

from typing import Dict, Optional, Tuple
import pandas as pd
from loguru import logger

from features.regime import (
    trend_regime,
    volatility_regime,
    combined_regime_filter,
)


class AllowTradeContext:
    """
    Pre-entry trade gate based on regime, volatility, time, and liquidity conditions.
    Returns True if all conditions allow trading, False otherwise.
    
    Args:
        config: Dict with regime/vol/time/liquidity thresholds.
        current_regime: Current market regime (e.g., from combined_regime_filter).
        fx_vol_regime: FX volatility regime (0=low, 1=normal, 2=high).
        comd_vol_regime: Commodity volatility regime (0=low, 1=normal, 2=high).
        current_time: Current datetime for time-based filters.
        liquidity_proxy: Proxy for liquidity (e.g., volume or spread).
    
    Raises:
        ValueError: If config missing required keys.
    """
    
    def __init__(
        self,
        config: Dict,
        current_regime: Optional[pd.Series] = None,
        fx_vol_regime: Optional[pd.Series] = None,
        comd_vol_regime: Optional[pd.Series] = None,
        current_time: Optional[pd.Timestamp] = None,
        liquidity_proxy: Optional[float] = None,
    ):
        self.config = config
        self.current_regime = current_regime
        self.fx_vol_regime = fx_vol_regime
        self.comd_vol_regime = comd_vol_regime
        self.current_time = current_time
        self.liquidity_proxy = liquidity_proxy
        
        # Extract thresholds from config
        self._validate_config()
        self.regime_filter = self.config["regime"].get("filter_strong_trend", False)
        self.vol_filter = self.config["regime"].get("filter_extreme_vol", False)
        self.time_filter = self.config.get("time_filter", {})
        self.liquidity_filter = self.config.get("liquidity_filter", {})
        
        logger.debug("AllowTradeContext initialized with config filters")
    
    def _validate_config(self) -> None:
        """Validate required config keys."""
        required = ["regime", "thresholds"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Config missing required key: {key}")
        logger.debug("Config validation passed")
    
    def _check_regime(self) -> bool:
        """Check if current regime allows trading (e.g., ranging only)."""
        if self.current_regime is None:
            return True  # No regime filter if not provided
        
        # Example: Allow only ranging (0); config-driven
        if self.regime_filter:
            is_ranging = self.current_regime.iloc[-1] == 0 if isinstance(self.current_regime, pd.Series) else self.current_regime == 0
            logger.debug(f"Regime check: ranging={is_ranging}")
            return is_ranging
        return True
    
    def _check_volatility(self) -> bool:
        """Check volatility regimes (e.g., avoid high vol)."""
        if self.fx_vol_regime is None or self.comd_vol_regime is None:
            return True
        
        if self.vol_filter:
            # Avoid high vol (2) in either series
            fx_low_normal = self.fx_vol_regime.iloc[-1] in [0, 1]
            comd_low_normal = self.comd_vol_regime.iloc[-1] in [0, 1]
            vol_ok = fx_low_normal and comd_low_normal
            logger.debug(f"Vol check: fx={self.fx_vol_regime.iloc[-1]}, comd={self.comd_vol_regime.iloc[-1]}, ok={vol_ok}")
            return vol_ok
        return True
    
    def _check_time(self) -> bool:
        """Check time-based filters (e.g., avoid news hours)."""
        if self.current_time is None or not self.time_filter:
            return True
        
        # Example: Avoid trading during high-impact news windows (config-driven)
        hour = self.current_time.hour
        allowed_hours = self.time_filter.get("allowed_hours", [9, 10, 11, 12, 13, 14, 15, 16])  # e.g., London/NY overlap
        time_ok = hour in allowed_hours
        logger.debug(f"Time check: hour={hour}, allowed={time_ok}")
        return time_ok
    
    def _check_liquidity(self) -> bool:
        """Check liquidity proxy (e.g., volume above threshold)."""
        if self.liquidity_proxy is None or not self.liquidity_filter:
            return True
        
        min_liquidity = self.liquidity_filter.get("min_liquidity", 1000)  # e.g., min volume
        liq_ok = self.liquidity_proxy >= min_liquidity
        logger.debug(f"Liquidity check: proxy={self.liquidity_proxy}, min={min_liquidity}, ok={liq_ok}")
        return liq_ok
    
    def accept(self, timestamp: pd.Timestamp) -> bool:
        """
        Main gate method: Check all conditions at given timestamp.
        
        Args:
            timestamp: Current timestamp for time-sensitive checks.
        
        Returns:
            True if trade allowed, False if any condition fails.
        """
        self.current_time = timestamp
        
        # Run all checks
        regime_ok = self._check_regime()
        vol_ok = self._check_volatility()
        time_ok = self._check_time()
        liq_ok = self._check_liquidity()
        
        allowed = all([regime_ok, vol_ok, time_ok, liq_ok])
        
        if not allowed:
            reason = []
            if not regime_ok: reason.append("regime")
            if not vol_ok: reason.append("volatility")
            if not time_ok: reason.append("time")
            if not liq_ok: reason.append("liquidity")
            logger.info(f"Trade rejected at {timestamp}: {', '.join(reason)}")
        else:
            logger.debug(f"Trade allowed at {timestamp}")
        
        return allowed


# Example usage (for testing)
if __name__ == "__main__":
    # Mock data/config
    mock_config = {
        "regime": {"filter_strong_trend": True, "filter_extreme_vol": True},
        "time_filter": {"allowed_hours": [9, 10, 11, 12, 13, 14, 15, 16]},
        "liquidity_filter": {"min_liquidity": 1000},
        "thresholds": {}  # Add as needed
    }
    mock_regime = pd.Series([0, 1, 0], index=pd.date_range("2023-01-01", periods=3))
    mock_fx_vol = pd.Series([1, 2, 1], index=mock_regime.index)
    mock_comd_vol = pd.Series([0, 1, 2], index=mock_regime.index)
    mock_time = pd.Timestamp("2023-01-01 10:00:00")
    mock_liq = 1500.0
    
    filter_ctx = AllowTradeContext(
        mock_config,
        current_regime=mock_regime,
        fx_vol_regime=mock_fx_vol,
        comd_vol_regime=mock_comd_vol,
        current_time=mock_time,
        liquidity_proxy=mock_liq,
    )
    
    print(f"Trade allowed: {filter_ctx.accept(mock_time)}")