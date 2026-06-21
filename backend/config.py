"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

# yfinance must be configured before import
os.environ.setdefault("YFINANCE_DISABLE_CURL_CFFI", "1")

# Load backend/.env for local development (Render/Vercel use dashboard env vars)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(_env_path)
    except ImportError:
        pass

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "").strip().strip('"').strip("'")
NEWS_URL = os.environ.get("NEWS_URL", "https://newsapi.org/v2/everything").strip()


def get_news_api_key() -> str:
    """Read NEWS_API_KEY at call time so Render env updates apply after redeploy."""
    return os.environ.get("NEWS_API_KEY", "").strip().strip('"').strip("'")

# Comma-separated origins, e.g. "https://stock-view-ebon.vercel.app,http://localhost:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

TRAIN_RATIO = 0.8
CNN_SEQUENCE_LENGTH = 10
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 16
MIN_HISTORY_ROWS = 50
MIN_FEATURE_ROWS = 30

MODEL_CACHE_TTL_SECONDS = int(os.environ.get("MODEL_CACHE_TTL_SECONDS", "3600"))
DATA_CACHE_TTL_SECONDS = int(os.environ.get("DATA_CACHE_TTL_SECONDS", "300"))

VALID_ALGORITHMS = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "cnn",
]
