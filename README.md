# 📈 StockView

**StockView** is a **desktop web application** for real-time stock market analysis. It combines interactive data visualization, anomaly detection, news sentiment insights, and machine learning–based price predictions into a single dashboard.

🚀 **Live Deployment**:

* **Frontend (Vercel)** → [View App](https://stock-view-five.vercel.app/)
* **Backend (Railway)**

---

## ✨ Features

* **Real-time Stock Data** → Current price, open, high, low, and volume.
* **Interactive Charts** → Zoom/pan with SMA, anomaly highlights, and prediction overlays.
* **Technical Indicators** → SMA, RSI, MACD, Bollinger Bands, ATR, and more.
* **Anomaly Detection** → Detects unusual price movements via Z-score.
* **Machine Learning Predictions** →

  * Algorithms: Linear Regression, Random Forest, XGBoost, LightGBM, CNN.
  * Metrics: R², MAE, MSE.
  * Algorithm comparison dashboard.
* **News & Sentiment Analysis** → Headlines with Positive/Neutral/Negative labels and sentiment charts.
* **Watchlist** → Add/remove stock symbols, synced with Firebase.

---

## 🏗️ Tech Stack

### Frontend (React + Vite)

* React + Hooks for state management
* Recharts for interactive visualization
* TailwindCSS for UI styling
* Firebase (Cloud Firestore) for watchlist persistence

### Backend (FastAPI)

* FastAPI for REST API
* Yahoo Finance (`yfinance`) for stock data
* VADER Sentiment Analyzer for news sentiment
* ML Models: scikit-learn, XGBoost, LightGBM, TensorFlow (CNN)
* Technical indicators with `ta` library

---

## 📂 Project Structure

```
StockView/
│── frontend/
│   ├── App.jsx        # React app (UI and chart logic)
│   └── ...
│
│── backend/
│   ├── main.py        # FastAPI backend with endpoints
│   └── ...
│
│── README.md
```

---

## 🖼️ Preview

<p align="center">
  <img src="https://drive.google.com/file/d/1lsEs0Cs0GFRApPek2OzNgqEwi2ZoGeX3/view?usp=sharing" alt="StockView Dashboard Preview" />
</p>

<p align="center">
  <img src="https://drive.google.com/file/d/1Z-LmdCnPWaf3LfgkIIRF175qDiVf6Gsk/view?usp=sharing" alt="StockView Predictions and Anomalies" />
</p>

<p align="center">
  <img src="https://drive.google.com/file/d/1d1RtUMWFrL_qUSD32agxIXi6V-GrXkRv/view?usp=sharing" alt="StockView Predictions and Anomalies" />
</p>

<p align="center">
  <img src="https://drive.google.com/file/d/1sUdMNBVV9i-_LdwQxBGWaYJ3SG5cSjxK/view?usp=sharing" alt="StockView Predictions and Anomalies" />
</p>

> Replace the placeholder images above with actual screenshots of your app for a better showcase.

---

## ⚙️ Local Development

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs at → `http://localhost:8000`

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Runs at → `http://localhost:5173`

---

## 🖥️ Usage

1. Search for a stock symbol (e.g., `AAPL`, `TSLA`).
2. Explore real-time price charts with SMA and anomalies.
3. Toggle **ML predictions** and **model comparison**.
4. Check news headlines with sentiment analysis.
5. Save your favorite stocks in the **watchlist**.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
Predictions and insights are **not financial advice**.

---

## 📌 Notes

* Optimized for **desktop browsers only** (not mobile responsive).
* Requires valid API keys (News API, Firebase, etc.).
* Works best on Chrome, Edge, or Firefox.
