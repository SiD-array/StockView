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
  <img src="https://private-user-images.githubusercontent.com/192932780/497589963-e8ce19cb-865a-4e03-beae-8cb2c346bd06.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk3MDExMDUsIm5iZiI6MTc1OTcwMDgwNSwicGF0aCI6Ii8xOTI5MzI3ODAvNDk3NTg5OTYzLWU4Y2UxOWNiLTg2NWEtNGUwMy1iZWFlLThjYjJjMzQ2YmQwNi5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAwNVQyMTQ2NDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zY2FlYmNmMjljMDUwYzg3YTAzMDM3MWQxYWMzMWU2MjhiMGNjZjA4OTRjMDc0ZTY4MmFkMzVjMzZlNDFiYjEwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.tbZpP3zqGcKsuDSN0q9l4X3MsWJNxTytOUSSWrkN11g" alt="StockView Dashboard Preview" />
</p>

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/192932780/497589962-9edb6668-a98d-4237-8833-7380cf8bbdac.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk3MDExMDUsIm5iZiI6MTc1OTcwMDgwNSwicGF0aCI6Ii8xOTI5MzI3ODAvNDk3NTg5OTYyLTllZGI2NjY4LWE5OGQtNDIzNy04ODMzLTczODBjZjhiYmRhYy5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAwNVQyMTQ2NDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lOTY1YWU4YjFhMGQ0NjJjNGUwOTAzMDk4YTY5ZDAxYWYxNWI0OWQ3NDAxNTA4NTkzY2EwY2Q1OWVkNGFkNWRlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.6PE03k13nSFFE1bAgYaZrvonmoH85QY4AZgKkD4BEkI" alt="StockView Predictions and Anomalies" />
</p>

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/192932780/497589965-bd4e8e6d-2a08-46a1-a38a-0f24885c9ec2.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk3MDExMDUsIm5iZiI6MTc1OTcwMDgwNSwicGF0aCI6Ii8xOTI5MzI3ODAvNDk3NTg5OTY1LWJkNGU4ZTZkLTJhMDgtNDZhMS1hMzhhLTBmMjQ4ODVjOWVjMi5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAwNVQyMTQ2NDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hMTMxMzdjM2ZmZjQ3MDMxZjg2NzE3ZjBmYjlhYWJhNTdjOWYzZDBjMDEwZWM4ZGEwNDA3ZDEzOThlOWE4Y2I5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.Zd6wV2joFogUfKbPMQIiJbrpmsO4m2eSVjK1RQSoWS8" alt="StockView Predictions and Anomalies" />
</p>

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/192932780/497589964-0b804850-c252-4c07-8320-8af08b6ab562.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTk3MDExMDUsIm5iZiI6MTc1OTcwMDgwNSwicGF0aCI6Ii8xOTI5MzI3ODAvNDk3NTg5OTY0LTBiODA0ODUwLWMyNTItNGMwNy04MzIwLThhZjA4YjZhYjU2Mi5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAwNVQyMTQ2NDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODdkMDVmNjNhZGMzOGQzY2VlODAyODIzZjQ3NDg2ZjA3NWJmOGRhMjFjYmM5MTVkOGFiYmE5N2U0MDA5NjQyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.cN8Vh6F1gbw9Kf8-Mny87XAhIyeTatFQlxh0zkIRUJQ" alt="StockView Predictions and Anomalies" />
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
