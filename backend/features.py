"""Technical indicator and feature engineering for ML models."""

import pandas as pd  # type: ignore
import ta  # type: ignore

FEATURE_COLUMNS = [
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "RSI",
    "MACD",
    "BB_upper",
    "BB_lower",
    "BB_middle",
    "Volume_SMA",
    "Price_Change",
    "High_Low_Pct",
    "Open_Close_Pct",
    "Close_lag_1",
    "Close_lag_2",
    "Close_lag_3",
    "Volume_lag_1",
    "Volume_lag_2",
    "Day_of_week",
    "Month",
    "Day_of_month",
]


def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive technical indicators for stock prediction."""
    df["SMA_5"] = ta.trend.sma_indicator(df["Close"], window=5)
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)

    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    df["MACD_signal"] = ta.trend.macd_signal(df["Close"])
    df["MACD_histogram"] = ta.trend.macd(df["Close"])

    df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])
    df["BB_middle"] = ta.volatility.bollinger_mavg(df["Close"])
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

    df["Volume_SMA"] = df["Volume"].rolling(window=10).mean()
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    df["Price_Change"] = df["Close"].pct_change()
    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Pct"] = (df["Open"] - df["Close"]) / df["Close"]

    for lag in [1, 2, 3, 5]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

    return df


def prepare_features(df: pd.DataFrame, target_col: str = "Close"):
    """
    Prepare feature matrix and target from OHLCV data.

    Returns:
        (X, y, feature_cols) or (None, None, None) if insufficient data.
    """
    df = create_technical_indicators(df)

    feature_cols = [
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "RSI",
        "MACD",
        "BB_upper",
        "BB_lower",
        "BB_middle",
        "Volume_SMA",
        "Price_Change",
        "High_Low_Pct",
        "Open_Close_Pct",
        "Close_lag_1",
        "Close_lag_2",
        "Close_lag_3",
        "Volume_lag_1",
        "Volume_lag_2",
    ]

    df["Day_of_week"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Day_of_month"] = df.index.day
    feature_cols.extend(["Day_of_week", "Month", "Day_of_month"])

    df_clean = df.dropna()
    if len(df_clean) < 30:
        return None, None, None

    return df_clean[feature_cols], df_clean[target_col], feature_cols
