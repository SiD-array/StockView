"""
Thin client for the Finnhub REST API (https://finnhub.io/docs/api).

Free plan covers what we use here (US stocks):
  - /quote           real-time price snapshot
  - /stock/profile2  company name, logo, exchange, industry
  - /company-news    ticker-specific news
"""

from datetime import date, timedelta

import requests  # type: ignore

from config import get_finnhub_api_key
from errors import ProviderError, SymbolNotFoundError, UpstreamRateLimitError

BASE_URL = "https://finnhub.io/api/v1"
TIMEOUT_SECONDS = 8

# One shared session reuses TCP/TLS connections across requests.
_session = requests.Session()


def is_configured() -> bool:
    return bool(get_finnhub_api_key())


def _get(path: str, params: dict) -> dict | list:
    """GET a Finnhub endpoint and map HTTP errors to our exception types."""
    api_key = get_finnhub_api_key()
    if not api_key:
        raise ProviderError("FINNHUB_API_KEY is not configured.")

    try:
        response = _session.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"X-Finnhub-Token": api_key},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ProviderError(f"Finnhub request failed: {exc}") from exc

    if response.status_code == 429:
        raise UpstreamRateLimitError("Finnhub rate limit reached.")
    if response.status_code in (401, 403):
        raise ProviderError(
            f"Finnhub denied access ({response.status_code}). "
            "Check FINNHUB_API_KEY, or the symbol may need a paid plan (non-US)."
        )
    if response.status_code != 200:
        raise ProviderError(f"Finnhub error {response.status_code}: {response.text[:200]}")

    return response.json()


def get_quote(symbol: str) -> dict:
    """
    Return a normalized quote.

    Finnhub returns {c, d, dp, h, l, o, pc, t}. Unknown symbols come back
    as all zeros, so a zero timestamp means "not found".
    """
    raw = _get("/quote", {"symbol": symbol})
    if not raw or not raw.get("t"):
        raise SymbolNotFoundError(f"No quote found for symbol '{symbol}'.")

    return {
        "price": raw["c"],
        "open": raw["o"],
        "high": raw["h"],
        "low": raw["l"],
        "previous_close": raw["pc"],
        "change": raw["d"],
        "change_percent": raw["dp"],
        "timestamp": raw["t"],
    }


def get_company_profile(symbol: str) -> dict:
    """Return company profile; empty dict if Finnhub has none."""
    raw = _get("/stock/profile2", {"symbol": symbol})
    if not raw:
        return {}
    return {
        "name": raw.get("name"),
        "logo": raw.get("logo"),
        "exchange": raw.get("exchange"),
        "industry": raw.get("finnhubIndustry"),
        "market_cap_millions": raw.get("marketCapitalization"),
        "website": raw.get("weburl"),
    }


def get_company_news(symbol: str, days: int = 7, limit: int = 10) -> list[dict]:
    """Return recent ticker-specific news, newest first."""
    today = date.today()
    raw = _get(
        "/company-news",
        {
            "symbol": symbol,
            "from": (today - timedelta(days=days)).isoformat(),
            "to": today.isoformat(),
        },
    )
    articles = sorted(raw or [], key=lambda a: a.get("datetime", 0), reverse=True)
    return [
        {
            "headline": a.get("headline", ""),
            "summary": a.get("summary", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "image": a.get("image") or None,
            "published_at_unix": a.get("datetime", 0),
        }
        for a in articles[:limit]
        if a.get("headline")
    ]
