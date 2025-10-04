import requests # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import xgboost as xgb # type: ignore
import lightgbm as lgb # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import ta # type: ignore
from fastapi import FastAPI, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
import yfinance as yf # type: ignore
import pandas as pd # type: ignore
import numpy as np  # pyright: ignore[reportMissingImports]
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "StockView API is running"}

MARKETAUX_TOKEN = "ZPFB0QLRLQwWBjDzZ6Th1EmAqLnGGU8mx7njToJo"
NEWS_API_KEY = "2f024385176b41ff9f13d916c7f6b742" 
NEWS_URL = "https://newsapi.org/v2/everything"

# Feature Engineering Functions
def create_technical_indicators(df):
    """Create comprehensive technical indicators for stock prediction"""
    # Price-based indicators
    df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # Momentum indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_histogram'] = ta.trend.macd(df['Close'])
    
    # Volatility indicators
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # Price change features
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Pct'] = (df['Open'] - df['Close']) / df['Close']
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    return df

def prepare_features(df, target_col='Close'):
    """Prepare features for machine learning models"""
    # Create technical indicators
    df = create_technical_indicators(df)
    
    # Select feature columns (reduced set to avoid NaN issues)
    feature_cols = [
        'SMA_5', 'SMA_10', 'SMA_20',
        'RSI', 'MACD',
        'BB_upper', 'BB_lower', 'BB_middle',
        'Volume_SMA', 'Price_Change', 'High_Low_Pct', 'Open_Close_Pct',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
        'Volume_lag_1', 'Volume_lag_2'
    ]
    
    # Add time-based features
    df['Day_of_week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Day_of_month'] = df.index.day
    
    feature_cols.extend(['Day_of_week', 'Month', 'Day_of_month'])
    
    # Remove rows with NaN values
    df_clean = df.dropna()
    
    if len(df_clean) < 30:  # Reduced minimum requirement
        return None, None, None
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    return X, y, feature_cols

def create_cnn_sequences(data, sequence_length=10):
    """Create sequences for CNN time series prediction"""
    sequences = []
    targets = []
    
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
        targets.append(data[i])
    
    return np.array(sequences), np.array(targets)

def build_cnn_model(input_shape):
    """Build CNN model for time series prediction"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_random_forest(X, y):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {'mse': mse, 'mae': mae, 'r2': r2}

def train_xgboost(X, y):
    """Train XGBoost model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {'mse': mse, 'mae': mae, 'r2': r2}

def train_lightgbm(X, y):
    """Train LightGBM model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {'mse': mse, 'mae': mae, 'r2': r2}

def train_cnn(X, y, sequence_length=10):
    """Train CNN model for time series prediction"""
    # Prepare sequences
    X_seq, y_seq = create_cnn_sequences(X.values, sequence_length)
    
    if len(X_seq) < 20:  # Need minimum sequences
        return None, None
    
    # Split data
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Reshape for CNN (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build and train model
    model = build_cnn_model((sequence_length, 1))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Evaluate model
    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {'mse': mse, 'mae': mae, 'r2': r2}

@app.get("/price")
def get_price(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        info = stock.info
        company_name = info.get("longName", symbol)
        if data.empty:
            raise HTTPException(status_code=404, detail="Stock symbol not found")

        latest = data.iloc[-1]
        return {
            "company": company_name,
            "symbol": symbol.upper(),
            "price": round(latest["Close"], 2),
            "open": round(latest["Open"], 2),
            "high": round(latest["High"], 2),
            "low": round(latest["Low"], 2),
            "volume": int(latest["Volume"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/history")
def get_history(symbol: str, range: str = "1d", interval: str = "5m"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=range, interval=interval)
        if data.empty:
            raise HTTPException(status_code=404, detail="No chart data found.")

        # Calculate SMA 10
        data["SMA_10"] = data["Close"].rolling(window=10).mean()

        # Calculate Z-score for anomaly detection
        window = 20 if interval.endswith("m") else 50 if interval.endswith("h") else 100
        mean_price = data["Close"].rolling(window=window).mean()
        std_price = data["Close"].rolling(window=window).std()
        data["Zscore"] = (data["Close"] - mean_price) / std_price

        # Mark anomalies (Z-score > 2 or < -2)
        data["Anomaly"] = data["Zscore"].apply(lambda z: abs(z) > 2 if not pd.isna(z) else False)

        chart_data = []
        for index, row in data.iterrows():
             # Convert timezone-aware datetime to local time if needed
            if hasattr(index, 'tz_localize') or hasattr(index, 'tz_convert'):
                # If timezone-aware, convert to local time
                if index.tz is not None:
                    local_time = index.tz_convert('America/New_York')  # Convert to Eastern Time
                else:
                    local_time = index
            else:
                local_time = index

            # Format time based on interval and range
            if interval.endswith("m"):
                # For minute intervals, show day and time
                if range in ["1d"]:
                    time_str = local_time.strftime("%H:%M")  # e.g., 14:30
                else:
                    time_str = local_time.strftime("%m/%d %H:%M")  # e.g., 12/16 14:30
            elif interval.endswith("h"):
                time_str = local_time.strftime("%d %b %H:%M")  # e.g., 16 Jul 15:00
            else:
                time_str = local_time.strftime("%m/%d")  # e.g., 12/16
            
            chart_data.append({
                "time": time_str,
                "timestamp": int(local_time.timestamp()),
                "price": round(row["Close"], 2),
                "sma_10": round(row["SMA_10"], 2) if not pd.isna(row["SMA_10"]) else None,
                "volume": int(row["Volume"]),
                "anomaly": bool(row["Anomaly"]),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2)
            })

        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/news")
def get_news(symbol: str, limit: int = 5):
    try:
        # Fetch news articles
        params = {
            "q": symbol,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
            "apiKey": NEWS_API_KEY,
        }
        response = requests.get(NEWS_URL, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="News API error")

        articles = response.json().get("articles", [])

        # Sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        news_data = []
        for article in articles:
            headline = article["title"]
            sentiment_score = analyzer.polarity_scores(headline)["compound"]

            sentiment = "Neutral"
            if sentiment_score > 0.2:
                sentiment = "Positive"
            elif sentiment_score < -0.2:
                sentiment = "Negative"

            news_data.append({
                "headline": headline,
                "url": article["url"],
                "published_at": article["publishedAt"],
                "sentiment": sentiment,
                "sentiment_score": sentiment_score
            })

        return {"symbol": symbol, "news": news_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/predict")
def predict(symbol: str, period: str = "3mo", interval: str = "1d", steps: int = 5, algorithm: str = "random_forest"):
    """
    Advanced stock price prediction with multiple algorithms
    
    Args:
        symbol: Stock symbol
        period: Data period (1mo, 3mo, 6mo, 1y, 2y)
        interval: Data interval (1d, 1h, 5m)
        steps: Number of future predictions
        algorithm: Algorithm to use (linear_regression, random_forest, xgboost, lightgbm, cnn)
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for prediction.")

        # Ensure we have enough data
        if len(data) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for reliable prediction. Need at least 50 data points.")

        # Prepare features and target
        X, y, feature_cols = prepare_features(data.copy())
        
        if X is None:
            raise HTTPException(status_code=400, detail="Insufficient data after feature engineering.")

        predictions = []
        model_metrics = {}
        
        if algorithm == "linear_regression":
            # Simple linear regression (original method)
            data_reset = data.reset_index()
            data_reset["Index"] = range(len(data_reset))
            X_simple = data_reset[["Index"]]
            y_simple = data_reset["Close"]
            
            model = LinearRegression()
            model.fit(X_simple, y_simple)
            
            # Predict future values
            future_idx = np.array(range(len(data_reset), len(data_reset) + steps)).reshape(-1, 1)
            predictions = model.predict(future_idx)
            
            # Calculate simple metrics
            y_pred_train = model.predict(X_simple)
            mse = mean_squared_error(y_simple, y_pred_train)
            mae = mean_absolute_error(y_simple, y_pred_train)
            r2 = r2_score(y_simple, y_pred_train)
            model_metrics = {'mse': mse, 'mae': mae, 'r2': r2}
            
        elif algorithm == "random_forest":
            model, metrics = train_random_forest(X, y)
            model_metrics = metrics
            
            # Make predictions
            last_features = X.iloc[-1:].values
            for i in range(steps):
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                # Update features for next prediction (simplified)
                last_features = np.roll(last_features, -1, axis=1)
                last_features[0, -1] = pred
                
        elif algorithm == "xgboost":
            model, metrics = train_xgboost(X, y)
            model_metrics = metrics
            
            # Make predictions
            last_features = X.iloc[-1:].values
            for i in range(steps):
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                # Update features for next prediction (simplified)
                last_features = np.roll(last_features, -1, axis=1)
                last_features[0, -1] = pred
                
        elif algorithm == "lightgbm":
            model, metrics = train_lightgbm(X, y)
            model_metrics = metrics
            
            # Make predictions
            last_features = X.iloc[-1:].values
            for i in range(steps):
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                # Update features for next prediction (simplified)
                last_features = np.roll(last_features, -1, axis=1)
                last_features[0, -1] = pred
                
        elif algorithm == "cnn":
            # For CNN, we use price sequences
            price_data = data['Close'].values
            model, metrics = train_cnn(price_data, price_data, sequence_length=10)
            
            if model is None:
                raise HTTPException(status_code=400, detail="Insufficient data for CNN model.")
            
            model_metrics = metrics
            
            # Make predictions using the last sequence
            last_sequence = price_data[-10:].reshape(1, 10, 1)
            for i in range(steps):
                pred = model.predict(last_sequence, verbose=0)[0][0]
                predictions.append(pred)
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred
                
        else:
            raise HTTPException(status_code=400, detail="Invalid algorithm. Choose from: linear_regression, random_forest, xgboost, lightgbm, cnn")

        # Format historical data
        history = []
        for idx, row in data.iterrows():
            time_str = idx.strftime("%b %d")
            history.append({
                "time": time_str,
                "price": round(row["Close"], 2)
            })

        # Format predicted data
        last_date = data.index[-1]
        predicted = []
        for i, pred in enumerate(predictions):
            future_date = last_date + pd.Timedelta(days=i+1)
            predicted.append({
                "time": future_date.strftime("%b %d"),
                "predicted": round(float(pred), 2)
            })

        return {
            "history": history, 
            "predictions": predicted,
            "algorithm": algorithm,
            "model_metrics": model_metrics,
            "feature_importance": get_feature_importance(model, feature_cols, algorithm) if algorithm != "cnn" else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_feature_importance(model, feature_cols, algorithm):
    """Get feature importance for tree-based models"""
    try:
        if algorithm in ["random_forest", "xgboost", "lightgbm"]:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_cols, model.feature_importances_))
                # Sort by importance and return top 10
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                return {k: float(v) for k, v in sorted_importance}
        return None
    except:
        return None

@app.get("/predict/compare")
def compare_algorithms(symbol: str, period: str = "3mo", interval: str = "1d", steps: int = 5):
    """Compare all available algorithms and return their performance metrics"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for prediction.")

        if len(data) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for reliable prediction.")

        X, y, feature_cols = prepare_features(data.copy())
        
        if X is None:
            raise HTTPException(status_code=400, detail="Insufficient data after feature engineering.")

        algorithms = ["linear_regression", "random_forest", "xgboost", "lightgbm"]
        results = {}
        
        for algo in algorithms:
            try:
                if algo == "linear_regression":
                    data_reset = data.reset_index()
                    data_reset["Index"] = range(len(data_reset))
                    X_simple = data_reset[["Index"]]
                    y_simple = data_reset["Close"]
                    
                    model = LinearRegression()
                    model.fit(X_simple, y_simple)
                    y_pred = model.predict(X_simple)
                    mse = mean_squared_error(y_simple, y_pred)
                    mae = mean_absolute_error(y_simple, y_pred)
                    r2 = r2_score(y_simple, y_pred)
                    results[algo] = {'mse': mse, 'mae': mae, 'r2': r2}
                    
                elif algo == "random_forest":
                    _, metrics = train_random_forest(X, y)
                    results[algo] = metrics
                    
                elif algo == "xgboost":
                    _, metrics = train_xgboost(X, y)
                    results[algo] = metrics
                    
                elif algo == "lightgbm":
                    _, metrics = train_lightgbm(X, y)
                    results[algo] = metrics
                    
            except Exception as e:
                results[algo] = {'error': str(e)}
        
        # Find best algorithm based on R² score
        best_algo = None
        best_r2 = -float('inf')
        for algo, metrics in results.items():
            if 'error' not in metrics and metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_algo = algo
        
        return {
            "comparison": results,
            "best_algorithm": best_algo,
            "best_r2_score": best_r2
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

