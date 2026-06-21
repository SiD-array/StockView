"""Model training with chronological validation."""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
import xgboost as xgb  # type: ignore
import lightgbm as lgb  # type: ignore

from config import (
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_SEQUENCE_LENGTH,
    TRAIN_RATIO,
)
from metrics import chronological_split, compute_metrics


def create_cnn_sequences(data, sequence_length: int = CNN_SEQUENCE_LENGTH):
    """Create sequences for CNN time series prediction."""
    sequences = []
    targets = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i - sequence_length : i])
        targets.append(data[i])
    return np.array(sequences), np.array(targets)


def build_cnn_model(input_shape):
    """Build CNN model for time series prediction (lazy TensorFlow import)."""
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation="relu"),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_linear_regression(data: pd.DataFrame):
    """Train linear regression with chronological holdout metrics."""
    data_reset = data.reset_index()
    data_reset["Index"] = range(len(data_reset))

    split_idx = int(len(data_reset) * TRAIN_RATIO)
    train_df = data_reset.iloc[:split_idx]
    test_df = data_reset.iloc[split_idx:]

    model = LinearRegression()
    model.fit(train_df[["Index"]], train_df["Close"])

    y_pred = model.predict(test_df[["Index"]])
    metrics = compute_metrics(test_df["Close"].values, y_pred)

    # Retrain on full history for forecasting
    full_model = LinearRegression()
    full_model.fit(data_reset[["Index"]], data_reset["Close"])

    return full_model, metrics, len(data_reset)


def train_random_forest(X: pd.DataFrame, y: pd.Series):
    """Train Random Forest with chronological holdout metrics."""
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)

    model.fit(X, y)
    return model, metrics


def train_xgboost(X: pd.DataFrame, y: pd.Series):
    """Train XGBoost with chronological holdout metrics."""
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)

    model.fit(X, y)
    return model, metrics


def train_lightgbm(X: pd.DataFrame, y: pd.Series):
    """Train LightGBM with chronological holdout metrics."""
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)

    model.fit(X, y)
    return model, metrics


def train_cnn(price_data, sequence_length: int = CNN_SEQUENCE_LENGTH):
    """Train CNN with chronological sequence split."""
    X_seq, y_seq = create_cnn_sequences(
        price_data.values if hasattr(price_data, "values") else price_data,
        sequence_length,
    )

    if len(X_seq) < 20:
        return None, None

    split_idx = int(TRAIN_RATIO * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_cnn_model((sequence_length, 1))
    model.fit(
        X_train,
        y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test, y_pred)

    # Retrain on all sequences for forecasting
    X_full = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    full_model = build_cnn_model((sequence_length, 1))
    full_model.fit(X_full, y_seq, epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE, verbose=0)

    return full_model, metrics


def get_feature_importance(model, feature_cols: list[str], algorithm: str):
    """Return top-10 feature importances for tree-based models."""
    if algorithm not in ("random_forest", "xgboost", "lightgbm"):
        return None
    if not hasattr(model, "feature_importances_"):
        return None

    importance_dict = dict(zip(feature_cols, model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    return {k: float(v) for k, v in sorted_importance}
