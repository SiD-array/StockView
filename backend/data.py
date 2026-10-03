"""
Market data access layer.

The rest of the app calls these functions and never talks to a provider
directly. That lets us choose the best source per data type:

  - Quotes & company names: Finnhub (official API) when FINNHUB_API_KEY is
    set, otherwise yfinance.
  - Historical candles: yfinance (not on Finnhub's free plan), cached with a
    stale-on-rate-limit fallback.
"""

import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
from yfinance.exceptions import YFRateLimitError  # type: ignore

from cache import TTLCache
from config import (
    COMPANY_NAME_CACHE_TTL_SECONDS,
    DATA_CACHE_TTL_SECONDS,
    MIN_HISTORY_ROWS,
    QUOTE_CACHE_TTL_SECONDS,
)
from errors import ProviderError, SymbolNotFoundError, UpstreamRateLimitError
from providers import finnhub

__all__ = [
    "UpstreamRateLimitError",
    "download_stock_data",
    "get_company_name",
    "get_quote",
]

_data_cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_SECONDS, max_size=256, name="data")
_name_cache = TTLCache(ttl_seconds=COMPANY_NAME_CACHE_TTL_SECONDS, max_size=1024, name="company")
_quote_cache = TTLCache(ttl_seconds=QUOTE_CACHE_TTL_SECONDS, max_size=512, name="quote")


def _normalize(symbol: str) -> str:
    return symbol.upper().strip()


# ---------------------------------------------------------------------------
# Historical candles (yfinance)
# ---------------------------------------------------------------------------

def download_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
    min_rows: int = MIN_HISTORY_ROWS,
) -> pd.DataFrame:
    """
    Download historical OHLCV data using yfinance.

    If Yahoo rate-limits the request, the most recent cached copy (even if
    expired) is returned instead of failing.

    Raises:
        ValueError: If no data is returned or row count is insufficient.
        UpstreamRateLimitError: If rate-limited and no cached copy exists.
    """
    symbol = _normalize(symbol)
    cache_key = f"{symbol}:{period}:{interval}"
    if use_cache:
        cached = _data_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

    try:
        data = yf.Ticker(symbol).history(period=period, interval=interval)
    except YFRateLimitError as exc:
        stale = _data_cache.get_stale(cache_key)
        if stale is not None:
            print(f"[data] Rate limited for {cache_key}; serving stale cache.")
            return stale.copy()
        raise UpstreamRateLimitError(
            "Yahoo Finance is rate limiting requests. Please try again in a minute."
        ) from exc

    if data.empty:
        raise SymbolNotFoundError(f"No data found for symbol '{symbol}'.")

    if len(data) < min_rows:
        raise ValueError(
            f"Insufficient data for '{symbol}': need at least {min_rows} rows, "
            f"got {len(data)}."
        )

    if use_cache:
        _data_cache.set(cache_key, data.copy())

    return data


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

def _quote_from_yfinance(symbol: str) -> dict:
    data = download_stock_data(symbol, period="5d", interval="1d", min_rows=1)
    latest = data.iloc[-1]
    price = float(latest["Close"])
    prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else None
    return {
        "price": price,
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "previous_close": prev_close,
        "change": price - prev_close if prev_close else None,
        "change_percent": (price - prev_close) / prev_close * 100 if prev_close else None,
        "volume": int(latest["Volume"]),
    }


def _volume_from_yfinance(symbol: str) -> int | None:
    """Finnhub's free quote has no volume; borrow it from cached daily candles."""
    try:
        data = download_stock_data(symbol, period="5d", interval="1d", min_rows=1)
        return int(data["Volume"].iloc[-1])
    except Exception:
        return None


def get_quote(symbol: str) -> dict:
    """
    Return the latest quote for a symbol, tagged with its `source`.

    Order: fresh cache -> Finnhub -> yfinance -> stale cache.
    """
    symbol = _normalize(symbol)
    cached = _quote_cache.get(symbol)
    if cached is not None:
        return cached

    try:
        if finnhub.is_configured():
            try:
                quote = finnhub.get_quote(symbol)
                quote["volume"] = _volume_from_yfinance(symbol)
                quote["source"] = "finnhub"
            except (ProviderError, UpstreamRateLimitError) as exc:
                print(f"[data] Finnhub quote failed for {symbol} ({exc}); using yfinance.")
                quote = {**_quote_from_yfinance(symbol), "source": "yfinance"}
        else:
            quote = {**_quote_from_yfinance(symbol), "source": "yfinance"}
    except UpstreamRateLimitError:
        stale = _quote_cache.get_stale(symbol)
        if stale is not None:
            return stale
        raise

    _quote_cache.set(symbol, quote)
    return quote


# ---------------------------------------------------------------------------
# Company name
# ---------------------------------------------------------------------------

def get_company_name(symbol: str) -> str:
    """
    Return the company's long name, cached for a long time.

    Uses Finnhub's profile endpoint when configured; otherwise yfinance's
    `.info` (Yahoo's most heavily rate-limited endpoint). Never raises:
    falls back to the ticker symbol.
    """
    symbol = _normalize(symbol)
    cached = _name_cache.get_stale(symbol)
    if cached is not None:
        return cached

    name = None
    try:
        if finnhub.is_configured():
            name = finnhub.get_company_profile(symbol).get("name")
        if not name:
            name = yf.Ticker(symbol).info.get("longName")
    except Exception as exc:  # rate limit, network, parsing...
        print(f"[data] Could not fetch company name for {symbol}: {exc}")

    if not name:
        return symbol

    _name_cache.set(symbol, name)
    return name
