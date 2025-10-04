# Advanced Machine Learning Features for Stock Price Prediction

## Overview

Your StockView application has been upgraded with advanced machine learning algorithms to provide more accurate and sophisticated stock price predictions. The system now supports multiple algorithms with comprehensive technical indicators and model comparison capabilities.

## New Features

### 🤖 Advanced ML Algorithms

1. **Random Forest** (Default)
   - Ensemble method using multiple decision trees
   - Handles non-linear relationships well
   - Provides feature importance rankings

2. **XGBoost**
   - Gradient boosting algorithm optimized for performance
   - Excellent for tabular data
   - High accuracy with proper tuning

3. **LightGBM**
   - Fast gradient boosting framework
   - Efficient memory usage
   - Good for large datasets

4. **CNN (Convolutional Neural Network)**
   - Deep learning approach for time series
   - Captures complex patterns in price sequences
   - Uses 1D convolutions for temporal data

5. **Linear Regression** (Original)
   - Simple baseline model
   - Fast and interpretable
   - Good for trend analysis

### 📊 Technical Indicators

The system now includes 25+ technical indicators:

**Price-based Indicators:**
- Simple Moving Averages (SMA 5, 10, 20)
- Exponential Moving Averages (EMA 12, 26)

**Momentum Indicators:**
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- MACD Signal and Histogram

**Volatility Indicators:**
- Bollinger Bands (Upper, Lower, Middle)
- Average True Range (ATR)

**Volume Indicators:**
- Volume Simple Moving Average
- On-Balance Volume (OBV)

**Price Change Features:**
- Price Change Percentage
- High-Low Percentage
- Open-Close Percentage

**Lagged Features:**
- Previous closing prices (1, 2, 3, 5 days)
- Previous volumes (1, 2, 3, 5 days)

**Time-based Features:**
- Day of week
- Month
- Day of month

### 🔬 Model Evaluation & Comparison

**Performance Metrics:**
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

**Model Comparison:**
- Automatic comparison of all algorithms
- Best algorithm recommendation based on R² score
- Side-by-side performance metrics

## How to Use

### 1. Algorithm Selection

1. Search for a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Select your preferred algorithm from the dropdown:
   - **Random Forest** (recommended for most cases)
   - **XGBoost** (for high accuracy)
   - **LightGBM** (for speed)
   - **CNN** (for complex patterns)
   - **Linear Regression** (for simplicity)

### 2. Generate Predictions

1. Click "Add Predictions" to generate 5-day price forecasts
2. View the prediction line on the chart (purple dashed line)
3. Check model performance metrics below the chart

### 3. Compare Algorithms

1. Click "Compare Models" to see performance comparison
2. View R² scores, MAE, and MSE for all algorithms
3. See which algorithm performs best for the current stock

### 4. Interpret Results

**R² Score Interpretation:**
- 0.8-1.0: Excellent fit
- 0.6-0.8: Good fit
- 0.4-0.6: Moderate fit
- 0.0-0.4: Poor fit

**Feature Importance:**
- For tree-based models, see which technical indicators are most important
- Helps understand what drives the predictions

## API Endpoints

### Enhanced Prediction Endpoint
```
GET /predict?symbol=AAPL&period=3mo&interval=1d&steps=5&algorithm=random_forest
```

**Parameters:**
- `symbol`: Stock symbol
- `period`: Data period (1mo, 3mo, 6mo, 1y, 2y)
- `interval`: Data interval (1d, 1h, 5m)
- `steps`: Number of future predictions (default: 5)
- `algorithm`: Algorithm to use (linear_regression, random_forest, xgboost, lightgbm, cnn)

**Response includes:**
- Historical data
- Predictions
- Model metrics (R², MAE, MSE)
- Feature importance (for tree-based models)

### Algorithm Comparison Endpoint
```
GET /predict/compare?symbol=AAPL&period=3mo&interval=1d&steps=5
```

**Response includes:**
- Performance comparison of all algorithms
- Best algorithm recommendation
- Individual metrics for each model

## Installation & Setup

### 1. Install New Dependencies

```bash
cd backend
pip install -r requirements.txt
```

New dependencies added:
- `tensorflow==2.15.0` (for CNN)
- `xgboost==2.0.3` (for XGBoost)
- `lightgbm==4.1.0` (for LightGBM)
- `ta==0.10.2` (for technical indicators)

### 2. Restart Backend Server

```bash
cd backend
python main.py
```

### 3. Frontend Updates

The frontend has been automatically updated with:
- Algorithm selection dropdown
- Model performance metrics display
- Algorithm comparison interface
- Enhanced prediction visualization

## Best Practices

### 1. Algorithm Selection Guidelines

- **Random Forest**: Good default choice, handles most scenarios well
- **XGBoost**: Use when you need highest accuracy and have sufficient data
- **LightGBM**: Use for faster training on large datasets
- **CNN**: Use for stocks with complex temporal patterns
- **Linear Regression**: Use for simple trend analysis or as baseline

### 2. Data Requirements

- Minimum 50 data points for reliable predictions
- 3+ months of data recommended for best results
- Daily intervals work best for most algorithms

### 3. Model Interpretation

- Higher R² scores indicate better model fit
- Feature importance helps understand prediction drivers
- Compare multiple algorithms to find the best fit for each stock

## Performance Considerations

### 1. Training Time
- **Linear Regression**: < 1 second
- **Random Forest**: 1-5 seconds
- **XGBoost/LightGBM**: 2-10 seconds
- **CNN**: 10-30 seconds

### 2. Memory Usage
- All models are trained in-memory
- CNN requires more memory for sequence data
- Models are not persisted between requests

### 3. Accuracy Expectations
- R² scores typically range from 0.3-0.8
- Higher volatility stocks may have lower scores
- Predictions become less accurate further into the future

## Troubleshooting

### Common Issues

1. **"Insufficient data" error**
   - Ensure stock has at least 50 trading days of data
   - Try a longer period (3mo or 6mo)

2. **CNN model fails**
   - CNN requires more data than other models
   - Try Random Forest or XGBoost instead

3. **Low R² scores**
   - This is normal for stock prediction
   - Try different algorithms
   - Consider that stock prices are inherently unpredictable

### Performance Tips

1. **For faster predictions**: Use Random Forest or LightGBM
2. **For highest accuracy**: Use XGBoost with 3+ months of data
3. **For complex patterns**: Try CNN with 6+ months of data

## Disclaimer

⚠️ **IMPORTANT**: These predictions are for educational and research purposes only. Stock prices are influenced by countless unpredictable factors including market sentiment, news, economic conditions, and global events that no model can fully account for. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

## Future Enhancements

Potential improvements for future versions:
- LSTM/GRU models for better sequence modeling
- Ensemble methods combining multiple algorithms
- Real-time model retraining
- Sentiment analysis integration
- Model persistence and caching
- Hyperparameter optimization
- Cross-validation for more robust evaluation
