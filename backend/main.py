"""
StockView FastAPI backend.

REST API for stock data, ML predictions, news sentiment, and model evaluation.
"""

import os
import traceback

import pandas as pd  # type: ignore
import requests  # type: ignore
import warnings
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from yfinance.exceptions import YFRateLimitError  # type: ignore

from cache import TTLCache, is_redis_active
from config import (
    CORS_ORIGINS,
    MODEL_CACHE_TTL_SECONDS,
    NEWS_CACHE_TTL_SECONDS,
    NEWS_URL,
    VALID_ALGORITHMS,
    get_news_api_key,
)
from data import download_stock_data, get_company_name, get_quote
from errors import ProviderError, UpstreamRateLimitError
from providers import finnhub
from features import prepare_features
from models.prediction import (
    predict_multi_step_cnn,
    predict_multi_step_linear,
    predict_multi_step_tree,
)
from models.training import (
    get_feature_importance,
    train_cnn,
    train_lightgbm,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)
from recommendations import get_all_recommendations, get_recommended_algorithm

warnings.filterwarnings("ignore")

app = FastAPI(title="StockView API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_model_cache = TTLCache(ttl_seconds=MODEL_CACHE_TTL_SECONDS, name="model")
_news_cache = TTLCache(ttl_seconds=NEWS_CACHE_TTL_SECONDS, max_size=256, name="news")
_sentiment = SentimentIntensityAnalyzer()


def period_key(data: pd.DataFrame) -> str:
    return f"{len(data)}:{data.index[-1]}"


def _log_and_raise(endpoint: str, exc: Exception) -> None:
    if isinstance(exc, (UpstreamRateLimitError, YFRateLimitError)):
        print(f"Rate limited in {endpoint}: {exc}")
        raise HTTPException(
            status_code=429,
            detail="Market data provider is rate limiting requests. Please try again in a minute.",
            headers={"Retry-After": "60"},
        )
    if isinstance(exc, ProviderError):
        print(f"Provider error in {endpoint}: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))
    print(f"Error in {endpoint}: {exc}")
    print(traceback.format_exc())
    raise HTTPException(status_code=500, detail=str(exc))


def _get_trained_model(symbol: str, algorithm: str, data: pd.DataFrame, X, y, feature_cols):
    """Return a cached trained model and its holdout metrics."""
    cache_key = f"{symbol.upper()}:{algorithm}:{period_key(data)}"
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached

    if algorithm == "linear_regression":
        model, metrics, data_length = train_linear_regression(data)
        result = (model, metrics, data_length, feature_cols)
    elif algorithm == "random_forest":
        model, metrics = train_random_forest(X, y)
        result = (model, metrics, None, feature_cols)
    elif algorithm == "xgboost":
        model, metrics = train_xgboost(X, y)
        result = (model, metrics, None, feature_cols)
    elif algorithm == "lightgbm":
        model, metrics = train_lightgbm(X, y)
        result = (model, metrics, None, feature_cols)
    elif algorithm == "cnn":
        model, metrics = train_cnn(data["Close"].values)
        if model is None:
            raise HTTPException(status_code=400, detail="Insufficient data for CNN model.")
        result = (model, metrics, None, feature_cols)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm. Choose from: {', '.join(VALID_ALGORITHMS)}",
        )

    _model_cache.set(cache_key, result)
    return result


def _compare_all_algorithms(data: pd.DataFrame, X, y) -> dict:
    """Evaluate all algorithms with chronological holdout metrics."""
    results = {}

    for algo in VALID_ALGORITHMS:
        try:
            if algo == "linear_regression":
                _, metrics, _ = train_linear_regression(data)
            elif algo == "random_forest":
                _, metrics = train_random_forest(X, y)
            elif algo == "xgboost":
                _, metrics = train_xgboost(X, y)
            elif algo == "lightgbm":
                _, metrics = train_lightgbm(X, y)
            elif algo == "cnn":
                _, metrics = train_cnn(data["Close"].values)
                if metrics is None:
                    results[algo] = {"error": "Insufficient data for CNN model."}
                    continue
            results[algo] = metrics
        except Exception as exc:
            results[algo] = {"error": str(exc)}

    return results


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "StockView API is running",
        "providers": {
            "quotes": "finnhub" if finnhub.is_configured() else "yfinance",
            "history": "yfinance",
            "cache": "redis" if is_redis_active() else "memory",
            "news": "finnhub"
            if finnhub.is_configured()
            else ("newsapi" if get_news_api_key() else None),
        },
    }


@app.get("/price")
def get_price(symbol: str):
    try:
        quote = get_quote(symbol)

        def r(value):
            return round(float(value), 2) if value is not None else None

        return {
            "company": get_company_name(symbol),
            "symbol": symbol.upper(),
            "price": r(quote["price"]),
            "open": r(quote["open"]),
            "high": r(quote["high"]),
            "low": r(quote["low"]),
            "volume": quote.get("volume"),
            "previous_close": r(quote.get("previous_close")),
            "change": r(quote.get("change")),
            "change_percent": r(quote.get("change_percent")),
            "source": quote.get("source"),
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="Stock symbol not found")
    except HTTPException:
        raise
    except Exception as exc:
        _log_and_raise("get_price", exc)


@app.get("/history")
def get_history(symbol: str, range: str = "1d", interval: str = "5m"):
    try:
        data = download_stock_data(symbol, period=range, interval=interval, min_rows=1)
        data["SMA_10"] = data["Close"].rolling(window=10).mean()

        window = 20 if interval.endswith("m") else 50 if interval.endswith("h") else 100
        mean_price = data["Close"].rolling(window=window).mean()
        std_price = data["Close"].rolling(window=window).std()
        data["Zscore"] = (data["Close"] - mean_price) / std_price
        data["Anomaly"] = data["Zscore"].apply(lambda z: abs(z) > 2 if not pd.isna(z) else False)

        chart_data = []
        for index, row in data.iterrows():
            if hasattr(index, "tz_convert") and index.tz is not None:
                local_time = index.tz_convert("America/New_York")
            else:
                local_time = index

            if interval.endswith("m"):
                time_str = (
                    local_time.strftime("%H:%M")
                    if range in ["1d"]
                    else local_time.strftime("%m/%d %H:%M")
                )
            elif interval.endswith("h"):
                time_str = local_time.strftime("%d %b %H:%M")
            else:
                time_str = local_time.strftime("%m/%d")

            chart_data.append({
                "time": time_str,
                "timestamp": int(local_time.timestamp()),
                "price": round(row["Close"], 2),
                "sma_10": round(row["SMA_10"], 2) if not pd.isna(row["SMA_10"]) else None,
                "volume": int(row["Volume"]),
                "anomaly": bool(row["Anomaly"]),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
            })

        return chart_data
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        _log_and_raise("get_history", exc)


def _score_sentiment(text: str) -> tuple[str, float]:
    """VADER compound score in [-1, 1] mapped to a label."""
    score = _sentiment.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive", score
    if score < -0.2:
        return "Negative", score
    return "Neutral", score


def _news_from_finnhub(symbol: str, limit: int) -> list[dict]:
    articles = finnhub.get_company_news(symbol, days=7, limit=limit)
    return [
        {
            "headline": a["headline"],
            "summary": a["summary"],
            "url": a["url"],
            "source": a["source"],
            "published_at": pd.Timestamp(a["published_at_unix"], unit="s", tz="UTC").isoformat(),
        }
        for a in articles
    ]


def _news_from_newsapi(symbol: str, limit: int) -> list[dict]:
    params = {
        "q": symbol,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": limit,
        "apiKey": get_news_api_key(),
    }
    response = requests.get(NEWS_URL, params=params, timeout=10)
    if response.status_code != 200:
        detail = "News API error"
        try:
            body = response.json()
            detail = body.get("message", body.get("code", detail))
        except Exception:
            detail = response.text or detail

        if response.status_code == 426:
            raise HTTPException(
                status_code=503,
                detail=(
                    "NewsAPI free Developer plan cannot be used on production servers "
                    f"(NewsAPI: {detail}). Set FINNHUB_API_KEY to use Finnhub company news instead."
                ),
            )
        raise HTTPException(status_code=response.status_code, detail=detail)

    return [
        {
            "headline": a["title"],
            "summary": a.get("description") or "",
            "url": a["url"],
            "source": (a.get("source") or {}).get("name", ""),
            "published_at": a["publishedAt"],
        }
        for a in response.json().get("articles", [])
        if a.get("title")
    ]


@app.get("/news")
def get_news(symbol: str, limit: int = 5):
    """Ticker news with sentiment. Finnhub if configured, else NewsAPI."""
    symbol = symbol.upper().strip()
    limit = max(1, min(limit, 20))

    cache_key = f"{symbol}:{limit}"
    cached = _news_cache.get(cache_key)
    if cached is not None:
        return cached

    use_finnhub = finnhub.is_configured()
    if not use_finnhub and not get_news_api_key():
        raise HTTPException(
            status_code=503,
            detail="News is not configured. Set FINNHUB_API_KEY (recommended) or NEWS_API_KEY.",
        )

    try:
        if use_finnhub:
            articles = _news_from_finnhub(symbol, limit)
        else:
            articles = _news_from_newsapi(symbol, limit)

        for article in articles:
            label, score = _score_sentiment(article["headline"])
            article["sentiment"] = label
            article["sentiment_score"] = score

        result = {
            "symbol": symbol,
            "source": "finnhub" if use_finnhub else "newsapi",
            "news": articles,
        }
        _news_cache.set(cache_key, result)
        return result
    except HTTPException:
        raise
    except (ProviderError, UpstreamRateLimitError) as exc:
        stale = _news_cache.get_stale(cache_key)
        if stale is not None:
            return stale
        _log_and_raise("get_news", exc)
    except Exception as exc:
        _log_and_raise("get_news", exc)


@app.get("/evaluation/recommendation")
def evaluation_recommendation(symbol: str):
    """Return the recommended algorithm for a symbol based on offline evaluation."""
    algorithm = get_recommended_algorithm(symbol)
    return {
        "symbol": symbol.upper(),
        "recommended_algorithm": algorithm,
        "note": "Based on chronological 80/20 backtest on 1y daily data.",
    }


@app.get("/evaluation/recommendations")
def evaluation_recommendations():
    """Return all symbol-specific algorithm recommendations."""
    return {"recommendations": get_all_recommendations()}


@app.get("/predict")
def predict(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    steps: int = 5,
    algorithm: str | None = None,
):
    """
    Stock price prediction with out-of-sample metrics and multi-step forecasting.

    Uses chronological 80/20 holdout for metrics. If algorithm is omitted,
    the best-performing model for the symbol is selected automatically.
    """
    try:
        algorithm = algorithm or get_recommended_algorithm(symbol)
        if algorithm not in VALID_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm. Choose from: {', '.join(VALID_ALGORITHMS)}",
            )

        data = download_stock_data(symbol, period=period, interval=interval)
        X, y, feature_cols = prepare_features(data.copy())

        if X is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data after feature engineering. Try a longer period (1y or 2y).",
            )

        model, model_metrics, data_length, feature_cols = _get_trained_model(
            symbol, algorithm, data, X, y, feature_cols
        )

        if algorithm == "linear_regression":
            predictions = predict_multi_step_linear(model, data_length, steps)
        elif algorithm == "cnn":
            predictions = predict_multi_step_cnn(model, data["Close"].values, steps)
        else:
            predictions = predict_multi_step_tree(data, model, steps)

        history = [
            {"time": idx.strftime("%b %d"), "price": round(row["Close"], 2)}
            for idx, row in data.iterrows()
        ]

        last_date = data.index[-1]
        predicted = [
            {
                "time": (last_date + pd.Timedelta(days=i + 1)).strftime("%b %d"),
                "predicted": round(float(pred), 2),
            }
            for i, pred in enumerate(predictions)
        ]

        return {
            "history": history,
            "predictions": predicted,
            "algorithm": algorithm,
            "recommended_algorithm": get_recommended_algorithm(symbol),
            "model_metrics": model_metrics,
            "metrics_note": "Out-of-sample metrics from chronological 80/20 holdout on test set.",
            "feature_importance": get_feature_importance(model, feature_cols, algorithm)
            if algorithm != "cnn"
            else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        _log_and_raise("predict", exc)


@app.get("/predict/compare")
def compare_algorithms(symbol: str, period: str = "1y", interval: str = "1d"):
    """Compare all algorithms using chronological holdout metrics (includes CNN)."""
    try:
        data = download_stock_data(symbol, period=period, interval=interval)
        X, y, _ = prepare_features(data.copy())

        if X is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data after feature engineering. Try a longer period (1y or 2y).",
            )

        results = _compare_all_algorithms(data, X, y)

        best_algo = None
        best_r2 = -float("inf")
        for algo, metrics in results.items():
            if "error" not in metrics and metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_algo = algo

        return {
            "comparison": results,
            "best_algorithm": best_algo,
            "best_r2_score": best_r2,
            "recommended_algorithm": get_recommended_algorithm(symbol),
            "metrics_note": "Out-of-sample metrics from chronological 80/20 holdout on test set.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        _log_and_raise("compare_algorithms", exc)


# Backward-compatible re-exports for evaluation.py
from models.training import create_cnn_sequences, build_cnn_model  # noqa: E402
from metrics import chronological_split, compute_metrics, calculate_mape  # noqa: E402
