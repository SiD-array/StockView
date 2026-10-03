"""
Tests for the Finnhub client and the provider fallback logic.

All network calls are mocked, so these run offline and without an API key.
"""

from unittest.mock import MagicMock, patch

import pytest

import data
from errors import ProviderError, SymbolNotFoundError, UpstreamRateLimitError
from providers import finnhub


def _response(status: int, payload):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    data._quote_cache.clear()
    data._name_cache.clear()


def test_quote_is_normalized():
    payload = {"c": 101.5, "d": 1.5, "dp": 1.5, "h": 102, "l": 99, "o": 100, "pc": 100, "t": 1700000000}
    with patch.object(finnhub._session, "get", return_value=_response(200, payload)) as get:
        quote = finnhub.get_quote("AAPL")

    assert quote["price"] == 101.5
    assert quote["previous_close"] == 100
    assert quote["change_percent"] == 1.5
    # Key must go in a header, never in the URL (URLs end up in logs).
    assert get.call_args.kwargs["headers"]["X-Finnhub-Token"] == "test-key"


def test_unknown_symbol_raises_not_found():
    zeros = {"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}
    with patch.object(finnhub._session, "get", return_value=_response(200, zeros)):
        with pytest.raises(SymbolNotFoundError):
            finnhub.get_quote("NOTREAL")


@pytest.mark.parametrize(
    "status, error",
    [(429, UpstreamRateLimitError), (401, ProviderError), (403, ProviderError), (500, ProviderError)],
)
def test_http_errors_are_mapped(status, error):
    with patch.object(finnhub._session, "get", return_value=_response(status, {})):
        with pytest.raises(error):
            finnhub.get_quote("AAPL")


def test_news_sorted_newest_first_and_limited():
    payload = [
        {"headline": "old", "datetime": 1, "url": "u1", "source": "s", "summary": ""},
        {"headline": "new", "datetime": 3, "url": "u3", "source": "s", "summary": ""},
        {"headline": "mid", "datetime": 2, "url": "u2", "source": "s", "summary": ""},
    ]
    with patch.object(finnhub._session, "get", return_value=_response(200, payload)):
        news = finnhub.get_company_news("AAPL", limit=2)

    assert [n["headline"] for n in news] == ["new", "mid"]


def test_get_quote_falls_back_to_yfinance_when_finnhub_fails():
    yf_quote = {"price": 50.0, "open": 49, "high": 51, "low": 48, "previous_close": 49,
                "change": 1, "change_percent": 2.0, "volume": 1000}
    with patch.object(finnhub, "get_quote", side_effect=ProviderError("down")), \
         patch.object(data, "_quote_from_yfinance", return_value=yf_quote):
        quote = data.get_quote("AAPL")

    assert quote["source"] == "yfinance"
    assert quote["price"] == 50.0


def test_get_quote_uses_finnhub_when_configured():
    fh_quote = {"price": 10.0, "open": 9, "high": 11, "low": 8, "previous_close": 9,
                "change": 1, "change_percent": 11.1, "timestamp": 1}
    with patch.object(finnhub, "get_quote", return_value=fh_quote), \
         patch.object(data, "_volume_from_yfinance", return_value=123):
        quote = data.get_quote("msft")

    assert quote["source"] == "finnhub"
    assert quote["volume"] == 123


def test_get_quote_uses_yfinance_without_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY")
    with patch.object(finnhub, "get_quote") as fh, \
         patch.object(data, "_quote_from_yfinance", return_value={"price": 1.0}):
        quote = data.get_quote("AAPL")

    fh.assert_not_called()
    assert quote["source"] == "yfinance"
