import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, ".")  # Add root to path for pytest
from src.features.regime import (
    correlation_gate,
    volatility_regime,
    trend_regime,
    hmm_regime_filter,
    vix_overlay,
    combined_regime_filter,
    TDistrHMM,
)


@pytest.fixture
def sample_fx():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.Series(np.cumsum(np.random.normal(0, 0.01, 200)), index=dates)


@pytest.fixture
def sample_comd():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.Series(np.cumsum(np.random.normal(0, 0.02, 200)), index=dates)


@pytest.fixture
def sample_vix():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.Series(np.random.normal(15, 5, 200), index=dates)


@pytest.fixture
def sample_config():
    return {
        "lookbacks": {"corr_window": 20},
        "regime": {
            "min_abs_corr": 0.15,
            "enable_hmm": True,
            "hmm": {"n_states": 3, "window": 50, "df": 3.0, "tol": 1e-4},
            "vix": {"thresh": 20, "boost_factor": 0.2, "max_shift": 1},
        },
    }


def test_correlation_gate(sample_fx, sample_comd, sample_config):
    gate = correlation_gate(
        sample_fx,
        sample_comd,
        sample_config["lookbacks"]["corr_window"],
        sample_config["regime"]["min_abs_corr"],
    )
    assert isinstance(gate, pd.Series)
    assert len(gate) == len(sample_fx)
    assert gate.dtype == bool


def test_volatility_regime(sample_fx):
    regime = volatility_regime(
        sample_fx, window=20, high_vol_threshold=0.03, low_vol_threshold=0.003
    )
    assert regime.isin([0, 1, 2]).all()
    assert len(regime) == len(sample_fx)


def test_trend_regime(sample_fx):
    regime = trend_regime(sample_fx, window=20, trend_threshold=0.015)
    assert regime.isin([-1, 0, 1]).all()
    assert len(regime) == len(sample_fx)


def test_tdistr_hmm():
    X = np.random.normal(0, 1, (100, 1))
    model = TDistrHMM(n_components=2, df=3.0, n_iter=10)
    model.fit(X)
    assert model.score(X) > -np.inf  # Basic fit check
    states = model.predict(X)
    assert states.shape == (100,)
    assert states.max() < model.n_components


def test_hmm_regime_filter(sample_fx, sample_config):
    spread_ret = sample_fx.pct_change().fillna(0)
    states = hmm_regime_filter(spread_ret, sample_config)
    states = states.fillna(0).astype(int)
    assert states.isin([0, 1, 2]).all()
    assert len(states) == len(sample_fx)
    assert states.dtype == "int"
    assert states.dtype == "int"


def test_vix_overlay(sample_vix, sample_fx, sample_config):
    spread_ret = sample_fx.pct_change().fillna(0)
    states = hmm_regime_filter(spread_ret, sample_config)
    adjusted = vix_overlay(sample_vix, states, sample_config)
    assert adjusted.isin([0, 1, 2]).all()
    assert len(adjusted) == len(sample_fx)


def test_combined_regime_filter(sample_fx, sample_comd, sample_config, sample_vix):
    gate = combined_regime_filter(sample_fx, sample_comd, sample_config, sample_vix)
    assert isinstance(gate, pd.Series)
    assert len(gate) == len(sample_fx)
    assert gate.dtype == bool

    # Test HMM disabled
    config_no_hmm = sample_config.copy()
    config_no_hmm["regime"]["enable_hmm"] = False
    gate_no_hmm = combined_regime_filter(sample_fx, sample_comd, config_no_hmm)
    assert gate_no_hmm.mean() >= gate.mean()  # Fewer filters, more signals
