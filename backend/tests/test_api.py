"""
Integration tests for all FastAPI REST endpoints.

Mocks external APIs (Yahoo Finance and Finnhub) so that tests run
instantaneously, offline, and reproducibly in CI without hitting rate limits.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from errors import ProviderError, SymbolNotFoundError, UpstreamRateLimitError
from main import app

client = TestClient(app)


def _mock_df(n_rows: int = 120) -> pd.DataFrame:
    """Generate deterministic OHLCV test data with a DatetimeIndex."""
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="B", tz="UTC")
    data = {
        "Open": [100.0 + i * 0.5 for i in range(n_rows)],
        "High": [102.0 + i * 0.5 for i in range(n_rows)],
        "Low": [99.0 + i * 0.5 for i in range(n_rows)],
        "Close": [101.0 + i * 0.5 for i in range(n_rows)],
        "Volume": [1000000 + i * 1000 for i in range(n_rows)],
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Health Check / Root
# ---------------------------------------------------------------------------

def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data
    assert "quotes" in data["providers"]
    assert "cache" in data["providers"]


# ---------------------------------------------------------------------------
# /price Endpoint
# ---------------------------------------------------------------------------

def test_get_price_success():
    mock_quote = {
        "price": 180.5,
        "open": 179.0,
        "high": 182.0,
        "low": 178.5,
        "previous_close": 178.0,
        "change": 2.5,
        "change_percent": 1.4,
        "volume": 50000000,
        "source": "finnhub",
    }
    with patch("main.get_quote", return_value=mock_quote), \
         patch("main.get_company_name", return_value="Apple Inc"):
        response = client.get("/price?symbol=AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["company"] == "Apple Inc"
    assert body["price"] == 180.5
    assert body["change_percent"] == 1.4
    assert body["source"] == "finnhub"


def test_get_price_not_found():
    with patch("main.get_quote", side_effect=SymbolNotFoundError("Symbol not found")):
        response = client.get("/price?symbol=INVALID_TICKER")

    assert response.status_code == 404
    assert "Stock symbol not found" in response.json()["detail"]


def test_get_price_rate_limited():
    with patch("main.get_quote", side_effect=UpstreamRateLimitError("Rate limit")):
        response = client.get("/price?symbol=AAPL")

    assert response.status_code == 429
    assert response.headers.get("retry-after") == "60"
    assert "rate limiting" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /history Endpoint
# ---------------------------------------------------------------------------

def test_get_history_success():
    df = _mock_df(30)
    with patch("main.download_stock_data", return_value=df):
        response = client.get("/history?symbol=AAPL&range=1d&interval=5m")

    assert response.status_code == 200
    points = response.json()
    assert len(points) == 30
    assert "time" in points[0]
    assert "price" in points[0]
    assert "anomaly" in points[0]
    assert "volume" in points[0]


def test_get_history_not_found():
    with patch("main.download_stock_data", side_effect=ValueError("No data")):
        response = client.get("/history?symbol=NONEXISTENT")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /news Endpoint
# ---------------------------------------------------------------------------

def test_get_news_with_sentiment():
    mock_articles = [
        {
            "headline": "Apple reports record quarterly earnings and revenue growth",
            "summary": "Great quarterly report.",
            "url": "https://example.com/1",
            "source": "Bloomberg",
            "published_at": "2026-10-01T12:00:00Z",
        },
        {
            "headline": "Tech shares plunge amid severe inflation and market crash fears",
            "summary": "Market drops.",
            "url": "https://example.com/2",
            "source": "Reuters",
            "published_at": "2026-10-01T10:00:00Z",
        },
    ]
    with patch("main.finnhub.is_configured", return_value=True), \
         patch("main._news_from_finnhub", return_value=mock_articles):
        # Clear main's news cache so it runs fresh
        from main import _news_cache
        _news_cache.clear()

        response = client.get("/news?symbol=AAPL&limit=2")

    assert response.status_code == 200
    news_items = response.json()["news"]
    assert len(news_items) == 2
    # First positive headline should be classified Positive
    assert news_items[0]["sentiment"] == "Positive"
    assert news_items[0]["sentiment_score"] > 0
    # Second negative headline should be classified Negative
    assert news_items[1]["sentiment"] == "Negative"
    assert news_items[1]["sentiment_score"] < 0


def test_get_news_unconfigured():
    with patch("main.finnhub.is_configured", return_value=False), \
         patch("main.get_news_api_key", return_value=""):
        from main import _news_cache
        _news_cache.clear()

        response = client.get("/news?symbol=AAPL")

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /evaluation/recommendation Endpoints
# ---------------------------------------------------------------------------

def test_recommendation_endpoint():
    response = client.get("/evaluation/recommendation?symbol=AAPL")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert "recommended_algorithm" in body


def test_all_recommendations_endpoint():
    response = client.get("/evaluation/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert "recommendations" in body
    assert isinstance(body["recommendations"], dict)


# ---------------------------------------------------------------------------
# /predict & /predict/compare Endpoints
# ---------------------------------------------------------------------------

def test_predict_invalid_algorithm():
    response = client.get("/predict?symbol=AAPL&algorithm=invalid_model")
    assert response.status_code == 400
    assert "Invalid algorithm" in response.json()["detail"]


def test_predict_linear_regression_success():
    df = _mock_df(120)
    with patch("main.download_stock_data", return_value=df):
        response = client.get("/predict?symbol=AAPL&algorithm=linear_regression&steps=3")

    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "linear_regression"
    assert len(body["predictions"]) == 3
    assert len(body["history"]) == 120
    assert "model_metrics" in body
    assert "mae" in body["model_metrics"]


def test_predict_compare_success():
    df = _mock_df(120)
    with patch("main.download_stock_data", return_value=df):
        response = client.get("/predict/compare?symbol=AAPL")

    assert response.status_code == 200
    body = response.json()
    assert "comparison" in body
    assert "best_algorithm" in body
    assert "recommended_algorithm" in body
