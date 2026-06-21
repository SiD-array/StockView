"""Multi-step forecasting with proper feature regeneration."""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from config import CNN_SEQUENCE_LENGTH
from features import prepare_features
from models.training import create_cnn_sequences


def _append_predicted_row(extended: pd.DataFrame, predicted_close: float) -> pd.DataFrame:
    """Append a synthetic OHLCV row for the next trading day."""
    last_idx = extended.index[-1]
    next_idx = last_idx + pd.Timedelta(days=1)
    last_row = extended.iloc[-1].copy()

    last_row["Open"] = last_row["Close"]
    last_row["Close"] = predicted_close
    last_row["High"] = max(float(last_row["High"]), predicted_close)
    last_row["Low"] = min(float(last_row["Low"]), predicted_close)
    # Carry forward volume — unknown for future days
    last_row["Volume"] = float(last_row["Volume"])

    extended.loc[next_idx] = last_row
    return extended


def predict_multi_step_tree(
    data: pd.DataFrame,
    model,
    steps: int,
) -> list[float]:
    """
    Generate multi-step forecasts by rebuilding technical features after each step.
    """
    extended = data.copy()
    predictions: list[float] = []

    for _ in range(steps):
        X, _, _ = prepare_features(extended.copy())
        if X is None or len(X) == 0:
            break

        pred = float(model.predict(X.iloc[-1:].values)[0])
        predictions.append(pred)
        extended = _append_predicted_row(extended, pred)

    return predictions


def predict_multi_step_linear(model, data_length: int, steps: int) -> list[float]:
    """Extrapolate linear trend for future time indices."""
    future_idx = np.array(range(data_length, data_length + steps)).reshape(-1, 1)
    return [float(v) for v in model.predict(future_idx)]


def predict_multi_step_cnn(model, price_data: np.ndarray, steps: int) -> list[float]:
    """Roll CNN input window forward using each new prediction."""
    last_sequence = price_data[-CNN_SEQUENCE_LENGTH:].reshape(1, CNN_SEQUENCE_LENGTH, 1)
    predictions: list[float] = []

    for _ in range(steps):
        pred = float(model.predict(last_sequence, verbose=0)[0][0])
        predictions.append(pred)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = pred

    return predictions
