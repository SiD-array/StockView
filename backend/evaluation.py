"""
StockView Model Evaluation Framework

Provides a reproducible, chronological backtesting pipeline for all forecasting
models implemented in StockView. Reuses the existing data pipeline (yfinance),
feature engineering, and model architectures from main.py.

Metrics computed per model:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R²   (Coefficient of Determination)

Outputs:
    - evaluation_results.csv
    - evaluation_report.txt
    - Charts in evaluation_outputs/
"""

import os

# Disable curl_cffi before yfinance import (same as main.py)
os.environ["YFINANCE_DISABLE_CURL_CFFI"] = "1"

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
import xgboost as xgb  # type: ignore
import lightgbm as lgb  # type: ignore

# Reuse StockView preprocessing and feature engineering (no TensorFlow dependency)
from main import (
    create_cnn_sequences,
    prepare_features,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation_outputs"
RESULTS_CSV = PROJECT_ROOT / "evaluation_results.csv"
REPORT_TXT = PROJECT_ROOT / "evaluation_report.txt"

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
TRAIN_RATIO = 0.8
CNN_SEQUENCE_LENGTH = 10
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 16

# Human-readable names for console output and charts
MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "cnn": "CNN",
}

ALL_MODELS = list(MODEL_DISPLAY_NAMES.keys())


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def download_stock_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download historical OHLCV data using the same yfinance pipeline as StockView.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').
        period: Lookback window passed to yfinance (default: '1y').
        interval: Bar interval (default: '1d').

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns.

    Raises:
        ValueError: If no data is returned or insufficient rows are available.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)

    if data.empty:
        raise ValueError(f"No data found for symbol '{symbol}'.")

    if len(data) < 50:
        raise ValueError(
            f"Insufficient data for '{symbol}': need at least 50 rows, got {len(data)}."
        )

    return data


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and target chronologically (no shuffling).

    The first `train_ratio` fraction is used for training; the remainder is
    reserved for out-of-sample testing to avoid look-ahead bias.

    Args:
        X: Feature DataFrame indexed by date.
        y: Target Series aligned with X.
        train_ratio: Fraction of samples assigned to training (default 0.8).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    split_idx = int(len(X) * train_ratio)
    if split_idx < 1 or split_idx >= len(X):
        raise ValueError(
            f"Cannot split {len(X)} samples at ratio {train_ratio}. "
            "Try a longer data period."
        )

    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE) in percent.

    Rows where the actual value is zero are excluded to avoid division errors.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0

    if not mask.any():
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate all evaluation metrics for a set of predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions aligned with y_true.

    Returns:
        Dict with keys: mae, rmse, mape, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Per-model evaluators (chronological 80/20 split)
# ---------------------------------------------------------------------------

def evaluate_linear_regression_model(data: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate simple linear regression using a time index feature.

    Mirrors the approach in main.py predict() but applies a chronological
    train/test split instead of in-sample evaluation.
    """
    data_reset = data.reset_index()
    data_reset["Index"] = range(len(data_reset))

    split_idx = int(len(data_reset) * TRAIN_RATIO)
    train_df = data_reset.iloc[:split_idx]
    test_df = data_reset.iloc[split_idx:]

    model = LinearRegression()
    model.fit(train_df[["Index"]], train_df["Close"])

    y_pred = model.predict(test_df[["Index"]])
    y_true = test_df["Close"].values
    metrics = calculate_metrics(y_true, y_pred)

    return {
        "model_key": "linear_regression",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_index": test_df["Date"] if "Date" in test_df.columns else test_df.index,
    }


def evaluate_random_forest_model(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Train Random Forest on 80% chronological data and evaluate on the remaining 20%."""
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
    y_true = y_test.values
    metrics = calculate_metrics(y_true, y_pred)

    return {
        "model_key": "random_forest",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_index": y_test.index,
    }


def evaluate_xgboost_model(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Train XGBoost on 80% chronological data and evaluate on the remaining 20%."""
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
    y_true = y_test.values
    metrics = calculate_metrics(y_true, y_pred)

    return {
        "model_key": "xgboost",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_index": y_test.index,
    }


def evaluate_lightgbm_model(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Train LightGBM on 80% chronological data and evaluate on the remaining 20%."""
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
    y_true = y_test.values
    metrics = calculate_metrics(y_true, y_pred)

    return {
        "model_key": "lightgbm",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_index": y_test.index,
    }


def evaluate_cnn_model(data: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate the 1D CNN on raw Close prices with chronological sequence split.

    Reuses create_cnn_sequences() and build_cnn_model() from main.py.
    Requires TensorFlow; raises ImportError if unavailable.
    """
    try:
        from main import build_cnn_model  # Lazy import triggers TensorFlow load
    except ImportError as exc:
        raise ImportError(
            "CNN evaluation requires TensorFlow. Install with: pip install tensorflow-cpu"
        ) from exc

    price_data = data["Close"].values
    X_seq, y_seq = create_cnn_sequences(price_data, CNN_SEQUENCE_LENGTH)

    if len(X_seq) < 20:
        raise ValueError(
            f"Insufficient CNN sequences: need at least 20, got {len(X_seq)}."
        )

    split_idx = int(TRAIN_RATIO * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_cnn_model((CNN_SEQUENCE_LENGTH, 1))
    model.fit(
        X_train,
        y_train,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    y_true = y_test
    metrics = calculate_metrics(y_true, y_pred)

    # Align test dates with the Close series (offset by sequence length)
    test_dates = data.index[CNN_SEQUENCE_LENGTH + split_idx :]

    return {
        "model_key": "cnn",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_index": test_dates,
    }


def evaluate_all_models_on_data(data: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Run every forecasting model on the provided OHLCV DataFrame.

    Returns:
        List of result dicts, one per successfully evaluated model.
    """
    results: list[dict[str, Any]] = []

    # Tabular models share engineered features from prepare_features()
    X, y, _feature_cols = prepare_features(data.copy())
    if X is None:
        raise ValueError(
            "Insufficient data after feature engineering. "
            "Try a longer period (e.g. '1y' or '2y')."
        )

    evaluators: list[tuple[str, Any]] = [
        ("linear_regression", lambda: evaluate_linear_regression_model(data)),
        ("random_forest", lambda: evaluate_random_forest_model(X, y)),
        ("xgboost", lambda: evaluate_xgboost_model(X, y)),
        ("lightgbm", lambda: evaluate_lightgbm_model(X, y)),
        ("cnn", lambda: evaluate_cnn_model(data)),
    ]

    for model_key, evaluate_fn in evaluators:
        try:
            results.append(evaluate_fn())
        except Exception as exc:
            display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            print(f"  Warning: {display_name} evaluation failed ({exc})")

    return results


# ---------------------------------------------------------------------------
# Reporting and visualization
# ---------------------------------------------------------------------------

def results_to_dataframe(
    model_results: list[dict[str, Any]],
    symbol: str | None = None,
) -> pd.DataFrame:
    """Convert a list of model result dicts into a metrics DataFrame."""
    rows = []
    for result in model_results:
        metrics = result["metrics"]
        row = {
            "model": MODEL_DISPLAY_NAMES.get(result["model_key"], result["model_key"]),
            "model_key": result["model_key"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "r2": metrics["r2"],
        }
        if symbol:
            row["symbol"] = symbol.upper()
        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted ASCII comparison table to stdout."""
    header = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>9} {'R²':>8}"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for _, row in df.iterrows():
        mape_str = f"{row['mape']:.2f}%" if pd.notna(row["mape"]) else "N/A"
        print(
            f"{row['model']:<20} "
            f"{row['mae']:>8.2f} "
            f"{row['rmse']:>8.2f} "
            f"{mape_str:>9} "
            f"{row['r2']:>8.2f}"
        )

    print(separator)


def _ensure_output_dir() -> Path:
    """Create the evaluation_outputs directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def plot_actual_vs_predicted(
    model_results: list[dict[str, Any]],
    symbol: str,
    output_dir: Path,
) -> None:
    """
    Save an Actual vs Predicted line chart for every evaluated model.

    Files are saved as: {symbol}_{model_key}_actual_vs_predicted.png
    """
    for result in model_results:
        model_key = result["model_key"]
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        test_index = result.get("test_index")

        fig, ax = plt.subplots(figsize=(10, 5))
        x_axis = test_index if test_index is not None else range(len(y_true))

        ax.plot(x_axis, y_true, label="Actual", color="#2563eb", linewidth=2)
        ax.plot(x_axis, y_pred, label="Predicted", color="#dc2626", linewidth=2, linestyle="--")
        ax.set_title(f"{symbol.upper()} — {display_name}: Actual vs Predicted (Test Set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        filepath = output_dir / f"{symbol.upper()}_{model_key}_actual_vs_predicted.png"
        fig.savefig(filepath, dpi=150)
        plt.close(fig)


def plot_error_comparison(df: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    """
    Save a grouped bar chart comparing MAE, RMSE, and MAPE across models.
    """
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, df["mae"], width, label="MAE", color="#3b82f6")
    ax.bar(x, df["rmse"], width, label="RMSE", color="#8b5cf6")
    ax.bar(x + width, df["mape"], width, label="MAPE (%)", color="#f59e0b")

    ax.set_title("Error Metrics Comparison Across Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error Value")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    filename = f"error_comparison{suffix}.png"
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def plot_performance_comparison(df: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    """
    Save a bar chart comparing R² scores across all models.
    """
    models = df["model"].tolist()
    r2_scores = df["r2"].tolist()
    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in r2_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, r2_scores, color=colors, edgecolor="#374151")
    ax.axhline(y=0, color="#6b7280", linewidth=0.8, linestyle="--")
    ax.set_title("Model Performance Comparison (R² Score on Test Set)")
    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")
    ax.set_ylim(min(min(r2_scores) - 0.1, -0.5), 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, score in zip(bars, r2_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.autofmt_xdate()
    fig.tight_layout()

    filename = f"model_performance_comparison{suffix}.png"
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def generate_summary_report(
    df: pd.DataFrame,
    symbols: list[str] | None = None,
    output_path: Path = REPORT_TXT,
) -> str:
    """
    Write evaluation_report.txt with best/worst models, averages, and recommendations.

    Returns:
        The report text (also written to disk).
    """
    # Rank by R² descending (higher is better)
    ranked = df.sort_values("r2", ascending=False)
    best_row = ranked.iloc[0]
    worst_row = ranked.iloc[-1]

    avg_mape = df["mape"].mean()
    avg_rmse = df["rmse"].mean()
    avg_r2 = df["r2"].mean()

    symbol_line = (
        f"Symbols evaluated: {', '.join(s.upper() for s in symbols)}\n"
        if symbols and len(symbols) > 1
        else f"Symbol evaluated: {symbols[0].upper()}\n"
        if symbols
        else ""
    )

    recommendations = _build_recommendations(df, best_row, worst_row)

    report_lines = [
        "=" * 60,
        "StockView Model Evaluation Report",
        "=" * 60,
        "",
        symbol_line.rstrip(),
        f"Train/Test split: {int(TRAIN_RATIO * 100)}% / {int(round((1 - TRAIN_RATIO) * 100))}% (chronological)",
        f"Data period: {DEFAULT_PERIOD} | Interval: {DEFAULT_INTERVAL}",
        "",
        "-" * 60,
        "SUMMARY METRICS (averaged across models" + (
            " and symbols" if symbols and len(symbols) > 1 else ""
        ) + ")",
        "-" * 60,
        f"Average MAPE: {avg_mape:.2f}%",
        f"Average RMSE: ${avg_rmse:.2f}",
        f"Average R²:   {avg_r2:.4f}",
        "",
        "-" * 60,
        "MODEL RANKINGS",
        "-" * 60,
        f"Best performing model:  {best_row['model']} (R² = {best_row['r2']:.4f}, MAPE = {best_row['mape']:.2f}%)",
        f"Worst performing model: {worst_row['model']} (R² = {worst_row['r2']:.4f}, MAPE = {worst_row['mape']:.2f}%)",
        "",
        "-" * 60,
        "DETAILED RESULTS",
        "-" * 60,
    ]

    for _, row in ranked.iterrows():
        symbol_suffix = f" [{row['symbol']}]" if "symbol" in row and pd.notna(row.get("symbol")) else ""
        report_lines.append(
            f"{row['model']}{symbol_suffix}: "
            f"MAE=${row['mae']:.2f}, RMSE=${row['rmse']:.2f}, "
            f"MAPE={row['mape']:.2f}%, R²={row['r2']:.4f}"
        )

    report_lines.extend([
        "",
        "-" * 60,
        "RECOMMENDATIONS",
        "-" * 60,
    ])
    report_lines.extend(recommendations)
    report_lines.extend(["", "=" * 60])

    report_text = "\n".join(report_lines)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def _build_recommendations(
    df: pd.DataFrame,
    best_row: pd.Series,
    worst_row: pd.Series,
) -> list[str]:
    """Generate actionable recommendations based on evaluation results."""
    recs: list[str] = []

    recs.append(
        f"1. Default algorithm: Use '{best_row['model']}' as the primary forecasting "
        f"model — it achieved the highest R² ({best_row['r2']:.4f}) on the held-out test set."
    )

    if worst_row["r2"] < 0:
        recs.append(
            f"2. Avoid '{worst_row['model']}' for production predictions — "
            f"negative R² ({worst_row['r2']:.4f}) indicates it underperforms a simple mean baseline."
        )
    else:
        recs.append(
            f"2. '{worst_row['model']}' showed the weakest generalization "
            f"(R² = {worst_row['r2']:.4f}); consider ensemble methods or hyperparameter tuning."
        )

    tree_models = df[df["model_key"].isin(["random_forest", "xgboost", "lightgbm"])]
    linear_rows = df[df["model_key"] == "linear_regression"]
    if not tree_models.empty:
        best_tree = tree_models.sort_values("r2", ascending=False).iloc[0]
        if not linear_rows.empty:
            linear_r2 = linear_rows.iloc[0]["r2"]
            if linear_r2 > best_tree["r2"]:
                recs.append(
                    f"3. Tree-based models ({best_tree['model']} best, R² = {best_tree['r2']:.4f}) "
                    f"underperformed linear regression (R² = {linear_r2:.4f}) on this test window. "
                    "This can happen with short test periods — consider walk-forward validation."
                )
            else:
                recs.append(
                    f"3. Among tree-based models, {best_tree['model']} performed best "
                    f"(MAPE = {best_tree['mape']:.2f}%). Engineered features help tree models "
                    "capture non-linear price dynamics."
                )
        else:
            recs.append(
                f"3. Among tree-based models, {best_tree['model']} performed best "
                f"(MAPE = {best_tree['mape']:.2f}%)."
            )

    cnn_rows = df[df["model_key"] == "cnn"]
    if not cnn_rows.empty:
        cnn_row = cnn_rows.iloc[0]
        tabular_best_r2 = df[df["model_key"] != "cnn"]["r2"].max()
        if cnn_row["r2"] < tabular_best_r2:
            recs.append(
                "4. CNN underperformed the best tabular model on daily data. "
                "Consider CNN for intraday intervals or longer training histories."
            )
        else:
            recs.append(
                "4. CNN is competitive with tabular models — suitable for sequence-heavy forecasting tasks."
            )
    elif "cnn" in ALL_MODELS:
        recs.append(
            "4. CNN was not evaluated (TensorFlow unavailable). "
            "Install tensorflow-cpu on Python 3.11 to include CNN in comparisons."
        )

    avg_mape = df["mape"].mean()
    if avg_mape > 5:
        recs.append(
            f"5. Average MAPE ({avg_mape:.2f}%) is relatively high. "
            "Extend the training window (2y+), add macro features, or implement walk-forward validation."
        )
    else:
        recs.append(
            f"5. Average MAPE ({avg_mape:.2f}%) is acceptable for daily stock forecasting. "
            "Document these metrics in the README for portfolio/resume use."
        )

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_symbol(
    symbol: str = "AAPL",
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all forecasting models on a single stock symbol.

    Downloads data via yfinance, trains each model on the first 80% of samples
    chronologically, generates test-set predictions, computes metrics, and
    optionally saves CSV results, charts, and a summary report.

    Args:
        symbol: Ticker symbol to evaluate (default: 'AAPL').
        period: yfinance history period (default: '1y').
        interval: Bar interval (default: '1d').
        save_outputs: Write CSV, report, and charts when True (default).

    Returns:
        DataFrame with one row per model and columns: model, mae, rmse, mape, r2.
    """
    print(f"\nEvaluating models on {symbol.upper()}...")
    print(f"Downloading {period} of {interval} data via yfinance...")

    data = download_stock_data(symbol, period=period, interval=interval)
    model_results = evaluate_all_models_on_data(data)
    results_df = results_to_dataframe(model_results, symbol=symbol)

    print(f"\nResults for {symbol.upper()}:")
    print_comparison_table(results_df)

    if save_outputs:
        output_dir = _ensure_output_dir()
        plot_actual_vs_predicted(model_results, symbol, output_dir)
        plot_error_comparison(results_df, output_dir, suffix=f"_{symbol.upper()}")
        plot_performance_comparison(results_df, output_dir, suffix=f"_{symbol.upper()}")

        results_df.to_csv(RESULTS_CSV, index=False)
        generate_summary_report(results_df, symbols=[symbol])
        print(f"\nSaved: {RESULTS_CSV}")
        print(f"Saved: {REPORT_TXT}")
        print(f"Charts: {output_dir}/")

    return results_df


def evaluate_multiple_symbols(
    symbols: list[str] | None = None,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all models across multiple symbols and compute average metrics.

    Args:
        symbols: List of ticker symbols (default: AAPL, MSFT, TSLA, GOOGL, AMZN).
        period: yfinance history period.
        interval: Bar interval.
        save_outputs: Write aggregated CSV, report, and charts when True.

    Returns:
        DataFrame with per-symbol, per-model rows plus aggregate averages.
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]

    all_results: list[pd.DataFrame] = []

    print(f"\n{'=' * 60}")
    print(f"Multi-Symbol Evaluation: {', '.join(s.upper() for s in symbols)}")
    print(f"{'=' * 60}")

    for symbol in symbols:
        try:
            data = download_stock_data(symbol, period=period, interval=interval)
            model_results = evaluate_all_models_on_data(data)
            symbol_df = results_to_dataframe(model_results, symbol=symbol)
            all_results.append(symbol_df)

            print(f"\n{symbol.upper()}:")
            print_comparison_table(symbol_df)

            if save_outputs:
                output_dir = _ensure_output_dir()
                plot_actual_vs_predicted(model_results, symbol, output_dir)

        except Exception as exc:
            print(f"\n  Skipping {symbol.upper()}: {exc}")

    if not all_results:
        raise ValueError("No symbols were successfully evaluated.")

    combined_df = pd.concat(all_results, ignore_index=True)

    # Average metrics across symbols for each model
    avg_df = (
        combined_df.groupby("model_key", as_index=False)
        .agg(
            model=("model", "first"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            r2=("r2", "mean"),
        )
    )
    avg_df["symbol"] = "AVERAGE"

    print(f"\n{'=' * 60}")
    print("AVERAGE METRICS ACROSS ALL SYMBOLS")
    print(f"{'=' * 60}")
    print_comparison_table(avg_df)

    if save_outputs:
        output_dir = _ensure_output_dir()
        plot_error_comparison(avg_df, output_dir, suffix="_average")
        plot_performance_comparison(avg_df, output_dir, suffix="_average")

        # Save full per-symbol results and append average rows
        export_df = pd.concat([combined_df, avg_df], ignore_index=True)
        export_df.to_csv(RESULTS_CSV, index=False)
        generate_summary_report(avg_df, symbols=symbols)
        print(f"\nSaved: {RESULTS_CSV}")
        print(f"Saved: {REPORT_TXT}")
        print(f"Charts: {output_dir}/")

    return combined_df


if __name__ == "__main__":
    """
    Run a full multi-symbol evaluation when executed directly:

        python evaluation.py
    """
    evaluate_multiple_symbols()
