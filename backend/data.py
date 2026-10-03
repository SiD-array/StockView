"""Stock data download via yfinance with caching and rate-limit resilience."""

import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
from yfinance.exceptions import YFRateLimitError  # type: ignore

from cache import TTLCache
from config import (
    COMPANY_NAME_CACHE_TTL_SECONDS,
    DATA_CACHE_TTL_SECONDS,
    MIN_HISTORY_ROWS,
)

_data_cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_SECONDS, max_size=256)
_name_cache = TTLCache(ttl_seconds=COMPANY_NAME_CACHE_TTL_SECONDS, max_size=1024)


class UpstreamRateLimitError(Exception):
    """Raised when Yahoo Finance rate-limits us and no cached fallback exists."""


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
    symbol = symbol.upper().strip()
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
        raise ValueError(f"No data found for symbol '{symbol}'.")

    if len(data) < min_rows:
        raise ValueError(
            f"Insufficient data for '{symbol}': need at least {min_rows} rows, "
            f"got {len(data)}."
        )

    if use_cache:
        _data_cache.set(cache_key, data.copy())

    return data


def get_company_name(symbol: str) -> str:
    """
    Return the company's long name, cached for a long time.

    `Ticker.info` hits Yahoo's quoteSummary endpoint, which is the most
    aggressively rate-limited one. Failures are non-fatal: we fall back
    to the ticker symbol.
    """
    symbol = symbol.upper().strip()
    cached = _name_cache.get_stale(symbol)
    if cached is not None:
        return cached

    try:
        name = yf.Ticker(symbol).info.get("longName") or symbol
    except Exception as exc:  # rate limit, network, parsing...
        print(f"[data] Could not fetch company name for {symbol}: {exc}")
        return symbol

    _name_cache.set(symbol, name)
    return name
