"""Tests for metrics and chronological splitting."""

import numpy as np
import pandas as pd

from metrics import chronological_split, compute_metrics, calculate_mape


def test_chronological_split_preserves_order():
    X = pd.DataFrame({"a": range(10)}, index=pd.date_range("2024-01-01", periods=10))
    y = pd.Series(range(10), index=X.index)

    X_train, X_test, y_train, y_test = chronological_split(X, y, train_ratio=0.8)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert X_train.index.max() < X_test.index.min()
    assert list(y_train) == list(range(8))
    assert list(y_test) == [8, 9]


def test_compute_metrics_perfect_prediction():
    y_true = np.array([100.0, 110.0, 105.0])
    y_pred = np.array([100.0, 110.0, 105.0])
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["r2"] == 1.0


def test_calculate_mape_ignores_zero_actuals():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([50.0, 90.0])
    mape = calculate_mape(y_true, y_pred)
    assert mape == 10.0
