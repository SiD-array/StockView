"""Tests for feature engineering."""

import pandas as pd
import numpy as np

from features import prepare_features, FEATURE_COLUMNS


def _sample_ohlcv(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="B")
    close = np.linspace(100, 150, rows) + np.random.default_rng(42).normal(0, 1, rows)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.full(rows, 1_000_000),
    }, index=idx)


def test_prepare_features_returns_expected_columns():
    df = _sample_ohlcv()
    X, y, feature_cols = prepare_features(df)

    assert X is not None
    assert len(X) >= 30
    assert feature_cols == FEATURE_COLUMNS
    assert len(y) == len(X)
