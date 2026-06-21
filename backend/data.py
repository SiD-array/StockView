"""Stock data download via yfinance with caching."""

import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore

from cache import TTLCache
from config import DATA_CACHE_TTL_SECONDS, MIN_HISTORY_ROWS

_data_cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_SECONDS)


def download_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
    min_rows: int = MIN_HISTORY_ROWS,
) -> pd.DataFrame:
    """
    Download historical OHLCV data using yfinance.

    Raises:
        ValueError: If no data is returned or row count is insufficient.
    """
    cache_key = f"{symbol.upper()}:{period}:{interval}"
    if use_cache:
        cached = _data_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)

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
