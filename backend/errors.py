"""Shared exception types for upstream data providers."""


class UpstreamRateLimitError(Exception):
    """An upstream provider (Yahoo, Finnhub, ...) is rate limiting us."""


class ProviderError(Exception):
    """An upstream provider failed (bad key, no access, outage, bad response)."""


class SymbolNotFoundError(ValueError):
    """The requested ticker symbol does not exist at the provider."""
