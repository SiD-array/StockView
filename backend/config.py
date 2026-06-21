"""Application configuration loaded from environment variables."""

import os

# yfinance must be configured before import
os.environ.setdefault("YFINANCE_DISABLE_CURL_CFFI", "1")

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
NEWS_URL = "https://newsapi.org/v2/everything"

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
