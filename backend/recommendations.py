"""Symbol-specific algorithm recommendations from offline evaluation."""

from pathlib import Path
import csv

# Defaults from evaluation_results.csv (1y daily, 80/20 chronological split)
SYMBOL_BEST_ALGORITHM: dict[str, str] = {
    "AAPL": "linear_regression",
    "MSFT": "lightgbm",
    "TSLA": "lightgbm",
    "GOOGL": "linear_regression",
    "AMZN": "random_forest",
}

DEFAULT_ALGORITHM = "lightgbm"

RESULTS_PATH = Path(__file__).resolve().parent.parent / "evaluation_results.csv"


def _load_from_csv() -> dict[str, str]:
    """Load per-symbol best algorithm from evaluation_results.csv if available."""
    if not RESULTS_PATH.exists():
        return {}

    symbol_rows: dict[str, list[dict]] = {}
    with RESULTS_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("symbol", "").upper()
            if not symbol or symbol == "AVERAGE":
                continue
            symbol_rows.setdefault(symbol, []).append(row)

    recommendations: dict[str, str] = {}
    for symbol, rows in symbol_rows.items():
        best = max(rows, key=lambda r: float(r.get("r2", "-inf")))
        recommendations[symbol] = best["model_key"]
    return recommendations


_CACHED_RECOMMENDATIONS = {**SYMBOL_BEST_ALGORITHM, **_load_from_csv()}


def get_recommended_algorithm(symbol: str) -> str:
    """Return the best-performing algorithm for a symbol based on offline evaluation."""
    return _CACHED_RECOMMENDATIONS.get(symbol.upper(), DEFAULT_ALGORITHM)


def get_all_recommendations() -> dict[str, str]:
    """Return all symbol-to-algorithm recommendations."""
    return dict(_CACHED_RECOMMENDATIONS)
