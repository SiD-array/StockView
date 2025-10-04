import axios from "axios";
import { useState, useEffect, useCallback, useMemo } from "react";
import { Info } from "lucide-react";
import { db } from "./firebase";
import { collection, getDocs, addDoc, deleteDoc, doc } from "firebase/firestore";
import { Cell, Bar, BarChart, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Brush, Legend, CartesianGrid, ReferenceDot } from 'recharts';

// Main App Component

function App() {
  const [symbol, setSymbol] = useState("AAPL");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chartData, setChartData] = useState([]);
  const [range, setRange] = useState({ value: "1d", interval: "5m" });
  const [lastUpdate, setLastUpdate] = useState(null);
  const [news, setNews] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [viewMode, setViewMode] = useState("chart");
  const [showPopup, setShowPopup] = useState(false);
  const [popupMessage, setPopupMessage] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [showPredictions, setShowPredictions] = useState(false);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("random_forest");
  const [modelMetrics, setModelMetrics] = useState(null);
  const [algorithmComparison, setAlgorithmComparison] = useState(null);
  const [showComparison, setShowComparison] = useState(false);

  // Fetch watchlist
  const fetchWatchlist = async () => {
    const querySnapshot = await getDocs(collection(db, "watchlist"));
    const items = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
    setWatchlist(items);
  };

  // Add to watchlist
  const addToWatchlist = async (symbol) => {
    if (!symbol) return;

    // Prevent duplicates
    if (watchlist.some(item => item.symbol === symbol)) {
      alert("Symbol already in watchlist!");
      return;
    }

    try {
      await addDoc(collection(db, "watchlist"), { symbol });
      await fetchWatchlist();
      setPopupMessage(`✅ ${symbol} added to watchlist successfully!`);
      setShowPopup(true);
    } catch (error) {
      alert("Error adding to watchlist. Please try again.");
    }
  };


  // Remove from watchlist
  const removeFromWatchlist = async (id) => {
    await deleteDoc(doc(db, "watchlist", id));
    fetchWatchlist();
  };

  function HelpModal() {
    const [open, setOpen] = useState(false);

    return (
      <>
        {/* Toggle Button */}
        <div className="flex justify-end mb-4">
          <button
            onClick={() => setOpen(true)}
            className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-1 rounded-md hover:bg-blue-700"
          >
            Help
          </button>
        </div>

        {open && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-xl shadow-lg w-96">
              <h2 className="text-xl font-semibold mb-3">📘 About StockView</h2>
              <p className="text-sm mb-3">
                StockView is a real-time stock analysis dashboard. You can search any stock symbol,
                choose time ranges, and interact with charts (zoom/pan).
              </p>

              <h3 className="text-md font-medium">Features:</h3>
              <ul className="list-disc list-inside text-sm mb-4 text-gray-700">
                <li><b>SMA 10:</b> Simple Moving Average smoothing short-term trends.</li>
                <li><b>Anomalies:</b> Points where price deviates unusually (high Z-score).</li>
                <li><b>Interactive Chart:</b> Zoom and pan to explore timeframes.</li>
                <li><b>Tips:</b> Use 1D/5D/1M/6M/1Y buttons to switch time ranges.</li>
              </ul>

              <button
                onClick={() => setOpen(false)}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </>
    );
  }

  // Custom tooltip to show anomaly information
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold">{`Time: ${data.time}`}</p>
          <p className="text-blue-600">{`Price: ${data.price}`}</p>
          <p className="text-gray-600">{`Open: ${data.open} | High: ${data.high} | Low: ${data.low}`}</p>
          {data.sma_10 && <p className="text-yellow-600">{`SMA 10: ${data.sma_10}`}</p>}
          {data.anomaly && (
            <p className="text-red-600 font-semibold">⚠️ Anomaly Detected!</p>
          )}
          <p className="text-gray-600">{`Volume: ${data.volume?.toLocaleString()}`}</p>
        </div>
      );
    }
    return null;
  };

  const handleSearch = async () => {
    if (!symbol.trim()) {
      setData(null);
      setChartData([]);
      setError("Please enter a stock symbol.");
      return; // stop further execution
    }

    setLoading(true);
    setError("");
    try {
      await fetchStock(symbol, range.value, range.interval);
      // Fetch news
      const res = await axios.get(`${API_URL}/news?symbol=${symbol}`);
      setNews(res.data.news || []);
      setPredictions([]);
      setShowPredictions(false);
    } catch (err) {
      setPopupMessage("⚠️ Stock symbol not found. Please enter a valid symbol.");
      setShowPopup(true);
      setError("Stock not found or API error" + err.message);
      setNews([]); // clear old news

    }
    setLoading(false);
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (symbol) {
        fetchStock(symbol, range.value, range.interval);
      }
    }, 600000);
    return () => clearInterval(intervalId);
  }, [symbol, range]);

  // Load watchlist on component mount
  useEffect(() => {
    fetchWatchlist();
  }, []);

  // Get API URL from environment or use localhost as fallback
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchStock = useCallback(async (symbolOverride = symbol, selectedRange = range.value, selectedInterval = range.interval) => {
    try {
      setError("");

      const [priceRes, chartRes] = await Promise.all([
        fetch(`${API_URL}/price?symbol=${symbolOverride}`),
        fetch(`${API_URL}/history?symbol=${symbolOverride}&range=${selectedRange}&interval=${selectedInterval}`)
      ]);

      if (!priceRes.ok || !chartRes.ok) {
        setData(null);
        setChartData([]);
        throw new Error("Stock not found");
      }

      const newPrice = await priceRes.json();
      const newChart = await chartRes.json();

      // Only update if data actually changed - replace the JSON.stringify comparison
      setData(prevData => {
        if (!prevData || prevData.price !== newPrice.price || prevData.volume !== newPrice.volume) {
          return newPrice;
        }
        return prevData;
      });

      setChartData(prevChart => {
        if (!prevChart.length || prevChart.length !== newChart.length ||
          prevChart[prevChart.length - 1]?.price !== newChart[newChart.length - 1]?.price) {
          return newChart;
        }
        return prevChart;
      });

      setLastUpdate(new Date().toLocaleTimeString());

    } catch (err) {
      setError(err.message || "Failed to fetch stock data");
    }
  }, [symbol, range.value, range.interval]);

  const fetchPredictions = useCallback(async (symbolOverride = symbol, algorithm = selectedAlgorithm) => {
    setPredictionLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict?symbol=${symbolOverride}&period=3mo&interval=1d&steps=5&algorithm=${algorithm}`);
      if (!response.ok) throw new Error("Failed to fetch predictions");

      const predictionData = await response.json();

      // Combine historical and predicted data for the same chart
      const combinedData = [
        ...predictionData.history.map(item => ({ ...item, predicted: null })),
        ...predictionData.predictions.map(item => ({ ...item, price: null }))
      ];

      setPredictions(combinedData);
      setModelMetrics(predictionData.model_metrics);
    } catch (err) {
      setError("Failed to load predictions: " + err.message);
    } finally {
      setPredictionLoading(false);
    }
  }, [symbol, selectedAlgorithm]);

  const fetchAlgorithmComparison = useCallback(async (symbolOverride = symbol) => {
    try {
      const response = await fetch(`${API_URL}/predict/compare?symbol=${symbolOverride}&period=3mo&interval=1d&steps=5`);
      if (!response.ok) throw new Error("Failed to fetch algorithm comparison");

      const comparisonData = await response.json();
      setAlgorithmComparison(comparisonData);
    } catch (err) {
      setError("Failed to load algorithm comparison: " + err.message);
    }
  }, [symbol]);

  const anomalyData = useMemo(() => chartData.filter(d => d.anomaly), [chartData]);

  return (
    <div className="min-h-screen bg-gray-100 p-2 sm:p-4 lg:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-center items-center mb-4 sm:mb-6 relative p-2 sm:p-4">
        <HelpModal />
        <button
          onClick={() => setViewMode(viewMode === "chart" ? "watchlist" : "chart")}
          className="bg-yellow-500 text-white px-3 py-2 rounded-md hover:bg-yellow-600 absolute top-2 right-2 sm:top-3 sm:right-20 text-sm sm:text-base"
        >
          {viewMode === "chart" ? "⭐ Watchlist" : "📊 Back to Chart"}
        </button>
        <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold text-center text-blue-600 mt-8 sm:mt-0">
          📈 Stock Analysis Dashboard
        </h1>
      </div>
      
      {/* Main Content */}
      <div className="flex flex-col xl:flex-row gap-4 p-2 sm:p-4 min-h-screen">

        {/* Sidebar */}
        <div className="w-full xl:w-1/4 bg-white p-3 sm:p-4 rounded-xl shadow-md space-y-3 sm:space-y-4 overflow-y-auto max-h-96 xl:max-h-full">
          <h2 className="text-lg sm:text-xl font-semibold">📊 Stock Insights</h2>
          <p className="text-sm sm:text-base">
            This tool provides real-time stock visualization with interactive charts,
            zoom & pan, anomaly detection, and sentiment overlays.
          </p>

          <div>
            <h3 className="text-sm sm:text-md font-medium mt-3 sm:mt-4">Popular Stocks</h3>
            <div className="grid grid-cols-2 sm:grid-cols-1 gap-1 sm:gap-0">
              <span className="text-xs sm:text-sm text-gray-700">AAPL – Apple</span>
              <span className="text-xs sm:text-sm text-gray-700">MSFT – Microsoft</span>
              <span className="text-xs sm:text-sm text-gray-700">GOOGL – Alphabet</span>
              <span className="text-xs sm:text-sm text-gray-700">TSLA – Tesla</span>
              <span className="text-xs sm:text-sm text-gray-700">AMZN – Amazon</span>
            </div>
          </div>

          <div>
            <h3 className="text-sm sm:text-md font-medium mt-3 sm:mt-4">Quick Tips</h3>
            <ul className="list-disc list-inside text-xs sm:text-sm text-gray-700 space-y-1">
              <li>Enter a valid stock symbol to search</li>
              <li>Use zoom/pan to explore trends</li>
              <li>Watch out for sudden spikes (anomalies)</li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm sm:text-md font-medium mt-3 sm:mt-4">Features</h3>
            <ul className="list-disc list-inside text-xs sm:text-sm text-gray-700 space-y-1">
              <li><b>SMA:</b> Moving averages for trend analysis</li>
              <li><b>Anomaly Detection:</b> Unusual price movements</li>
              <li><b>ML Predictions:</b> 5 advanced algorithms</li>
              <li><b>Technical Indicators:</b> RSI, MACD, Bollinger Bands</li>
              <li><b>Model Comparison:</b> Performance metrics</li>
            </ul>
          </div>

          {lastUpdate && (
            <div className="mt-4 text-xs text-gray-500">
              Last updated: {lastUpdate}
            </div>
          )}

        </div>

        {viewMode === "chart" ? (
          <>
            {/* Chart Content */}
            <div className="w-full xl:w-3/4 max-h-full overflow-auto bg-white p-3 sm:p-4 lg:p-6 rounded-xl shadow-md space-y-2">
              {/* Search Bar */}
              <div className="flex flex-col sm:flex-row gap-2 mb-4">
                <input
                  type="text"
                  className="flex-1 border border-gray-300 p-2 sm:p-3 rounded-md text-sm sm:text-base"
                  value={symbol}
                  placeholder="Enter stock symbol (e.g., AAPL)"
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleSearch();
                    }
                  }}
                />
                <button
                  onClick={handleSearch}
                  className="bg-blue-600 text-white px-4 py-2 sm:py-3 rounded-md hover:bg-blue-700 text-sm sm:text-base font-medium"
                >
                  Search
                </button>
              </div>

              {loading && <p className="text-center text-gray-500">Loading...</p>}

              {error && (<p className="text-center text-red-500 text-sm">{error}</p>)}

              {data && (
                <div className="flex flex-col sm:flex-row justify-center mt-2 gap-2 sm:gap-3">
                  <select
                    value={selectedAlgorithm}
                    onChange={(e) => setSelectedAlgorithm(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-md text-sm sm:text-base flex-1 sm:flex-none"
                  >
                    <option value="linear_regression">Linear Regression</option>
                    <option value="random_forest">Random Forest</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="lightgbm">LightGBM</option>
                    <option value="cnn">CNN</option>
                  </select>
                  <button
                    onClick={() => {
                      if (!showPredictions) {
                        fetchPredictions();
                      }
                      setShowPredictions(!showPredictions);
                    }}
                    className={`px-3 sm:px-4 py-2 rounded-md border text-sm sm:text-base ${showPredictions
                      ? "bg-purple-600 text-white"
                      : "bg-white text-purple-600 border-purple-600"
                      }`}
                  >
                    {predictionLoading ? "Loading..." : showPredictions ? "Hide Predictions" : "Add Predictions"}
                  </button>
                  <button
                    onClick={() => {
                      if (!showComparison) {
                        fetchAlgorithmComparison();
                      }
                      setShowComparison(!showComparison);
                    }}
                    className={`px-3 py-2 rounded-md border text-sm sm:text-base ${showComparison
                      ? "bg-green-600 text-white"
                      : "bg-white text-green-600 border-green-600"
                      }`}
                  >
                    {showComparison ? "Hide Comparison" : "Compare Models"}
                  </button>
                </div>
              )}

              {data && (
                <div className="text-center">
                  <h2 className="text-lg sm:text-xl lg:text-2xl font-bold mb-2">{data.company}</h2>
                  <p className="text-base sm:text-lg text-green-600 transition-all duration-500 ease-in-out">Current Price: ${data.price}</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-5 mt-1 mb-3 font-semibold">
                    <p className="text-xs sm:text-sm text-gray-700">Open: ${data.open}</p>
                    <p className="text-xs sm:text-sm text-gray-700">High: ${data.high}</p>
                    <p className="text-xs sm:text-sm text-gray-700">Low: ${data.low}</p>
                    <p className="text-xs sm:text-sm text-gray-700">Volume: {data.volume.toLocaleString()}</p>
                  </div>
                </div>
              )}

              <div className="flex gap-1 sm:gap-2 justify-center mt-4 flex-wrap">
                {[
                  { label: "1D", value: "1d", interval: "5m" },
                  { label: "5D", value: "5d", interval: "30m" },
                  { label: "1M", value: "1mo", interval: "1d" },
                  { label: "6M", value: "6mo", interval: "1d" },
                  { label: "1Y", value: "1y", interval: "1d" },
                ].map(({ label, value, interval }) => (
                  <button
                    key={value}
                    onClick={() => {
                      setRange({ value, interval });
                      fetchStock(symbol, value, interval);  // update chart immediately
                    }}
                    className={`px-2 sm:px-3 py-1 sm:py-2 rounded-md border text-xs sm:text-sm ${range.value === value
                      ? "bg-blue-600 text-white"
                      : "bg-white text-blue-600 border-blue-600"
                      }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {chartData.length > 0 && (
                <div className="bg-white mt-4 p-2 sm:p-4 rounded-xl shadow-md w-full mx-auto">
                  <h3 className="text-base sm:text-lg font-semibold text-center mb-2">Intraday Price Chart</h3>
                  <ResponsiveContainer width="100%" height={200} className="sm:h-[250px]">
                    <LineChart data={showPredictions && predictions.length > 0 ? predictions : chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        tick={{ fontSize: 8 }} 
                        angle={-45} 
                        textAnchor="end" 
                        interval={chartData.length > 50 ? Math.floor(chartData.length / 10) : 0}
                        className="text-xs"
                      />
                      <YAxis domain={['auto', 'auto']} tick={{ fontSize: 8 }} />
                      <Tooltip content={CustomTooltip} />
                      <Legend
                        formatter={(value) => {
                          if (value === "sma_10") {
                            return (
                              <span className="flex items-center gap-1">
                                SMA 10
                                <div className="relative group">
                                  <Info size={14} className="text-gray-500 cursor-pointer" />
                                  <div className="absolute left-5 top-0 hidden group-hover:block bg-gray-800 text-white text-xs p-2 rounded w-48 z-10">
                                    Simple Moving Average over the last 10 data points. Smooths price trends.
                                  </div>
                                </div>
                              </span>
                            );
                          }
                          if (value === "price") return "Price";
                          return value;
                        }}
                      />

                      {/* Price Line */}
                      <Line type="monotone" dataKey="price" stroke="#2563eb" strokeWidth={2} dot={false} />
                      {/* SMA 10 Line */}
                      <Line type="monotone" dataKey="sma_10" stroke="#f59e0b" dot={false} strokeWidth={2} name="SMA 10" />
                      {/* Predicted Price Line */}
                      {showPredictions && (
                        <Line
                          type="monotone"
                          dataKey="predicted"
                          stroke="#9333ea"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={{ fill: '#9333ea', r: 4 }}
                          name="Predicted Price"
                          connectNulls={false}
                        />
                      )}
                      {/* Anomaly markers */}
                      {anomalyData.map((point, index) => (
                        <ReferenceDot
                          key={`anomaly-${point.timestamp || index}`}
                          x={point.time}
                          y={point.price}
                          r={4}
                          fill="#ef4444"
                          stroke="#dc2626"
                          strokeWidth={2}
                        />
                      ))}
                      <Brush className="mt-2" dataKey="time" height={30} stroke="#8884d8" />
                    </LineChart>
                  </ResponsiveContainer>

                  {/* Conditional Summary - Anomaly or Prediction */}
                  {showPredictions ? (
                    <div className="mt-4 p-3 bg-orange-50 border border-orange-300 rounded-lg">
                      <h4 className="font-semibold text-orange-800 mb-2">📊 Price Predictions</h4>
                      <div className="text-sm text-orange-700 space-y-2">
                        <p>
                          Showing 5-day price predictions using <strong>{selectedAlgorithm.replace('_', ' ').toUpperCase()}</strong> algorithm with advanced technical indicators.
                        </p>
                        {modelMetrics && (
                          <div className="bg-blue-50 border border-blue-300 p-2 rounded text-blue-800">
                            <strong>Model Performance:</strong> R² = {modelMetrics.r2?.toFixed(3)}, MAE = ${modelMetrics.mae?.toFixed(2)}, MSE = {modelMetrics.mse?.toFixed(2)}
                          </div>
                        )}
                        <div className="bg-red-100 border border-red-300 p-2 rounded text-red-800">
                          <strong>⚠️ IMPORTANT DISCLAIMER:</strong> These predictions are for educational purposes only and should NOT be used for investment decisions. Stock prices are influenced by countless unpredictable factors including market sentiment, news, economic conditions, and global events that this model cannot account for. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.
                        </div>
                      </div>
                    </div>
                  ) : (
                    chartData.filter(d => d.anomaly).length > 0 && (
                      <div className="mt-4 p-2 bg-red-50 border border-red-200 rounded-lg">
                        <h4 className="font-semibold text-red-800 mb-2">⚠️ Anomalies Detected</h4>
                        <p className="text-sm text-red-700">
                          Found {chartData.filter(d => d.anomaly).length} price anomalies in the current timeframe.
                          The red points represent significant deviations from the recent price trend.
                        </p>
                      </div>
                    )
                  )}
                </div>
              )}

              {showPredictions && predictions.history && (
                <div className="bg-white mt-6 p-4 rounded-xl shadow-md max-w-3xl mx-auto">
                  <h3 className="text-lg font-semibold text-center mb-2">Price Predictions (Next 5 Days)</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={[...predictions.history, ...predictions.predictions]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                      <YAxis domain={['auto', 'auto']} />
                      <Tooltip />
                      <Legend />

                      {/* Historical prices */}
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#2563eb"
                        strokeWidth={2}
                        dot={false}
                        name="Historical Price"
                      />

                      {/* Predicted prices */}
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#9333ea"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={{ fill: '#9333ea', r: 4 }}
                        name="Predicted Price"
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <div className="mt-4 text-xs text-gray-500 text-center">
                    Note: Predictions are based on {selectedAlgorithm.replace('_', ' ')} and should not be used as financial advice.
                  </div>
                </div>
              )}

              {/* Algorithm Comparison Section */}
              {showComparison && algorithmComparison && (
                <div className="bg-white mt-4 sm:mt-6 p-3 sm:p-4 rounded-xl shadow-md w-full mx-auto">
                  <h3 className="text-base sm:text-lg font-semibold text-center mb-3 sm:mb-4">🔬 Algorithm Performance Comparison</h3>
                  
                  <div className="mb-3 sm:mb-4 p-2 sm:p-3 bg-green-50 border border-green-300 rounded-lg">
                    <p className="text-green-800 text-xs sm:text-sm">
                      <strong>Best Algorithm:</strong> {algorithmComparison.best_algorithm?.replace('_', ' ').toUpperCase()} 
                      (R² = {algorithmComparison.best_r2_score?.toFixed(3)})
                    </p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                    {Object.entries(algorithmComparison.comparison).map(([algorithm, metrics]) => (
                      <div key={algorithm} className="border border-gray-200 rounded-lg p-2 sm:p-3">
                        <h4 className="font-semibold text-gray-800 mb-2 text-sm sm:text-base">
                          {algorithm.replace('_', ' ').toUpperCase()}
                        </h4>
                        {metrics.error ? (
                          <p className="text-red-600 text-xs sm:text-sm">Error: {metrics.error}</p>
                        ) : (
                          <div className="space-y-1 text-xs sm:text-sm">
                            <div className="flex justify-between">
                              <span>R² Score:</span>
                              <span className={`font-medium ${metrics.r2 > 0.7 ? 'text-green-600' : metrics.r2 > 0.4 ? 'text-yellow-600' : 'text-red-600'}`}>
                                {metrics.r2?.toFixed(3)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>MAE:</span>
                              <span className="font-medium">${metrics.mae?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>MSE:</span>
                              <span className="font-medium">{metrics.mse?.toFixed(2)}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="mt-3 sm:mt-4 p-2 sm:p-3 bg-blue-50 border border-blue-300 rounded-lg">
                    <h5 className="font-semibold text-blue-800 mb-2 text-sm sm:text-base">📈 Algorithm Descriptions:</h5>
                    <div className="text-xs sm:text-sm text-blue-700 space-y-1">
                      <p><strong>Linear Regression:</strong> Simple trend-based prediction using time as the only feature.</p>
                      <p><strong>Random Forest:</strong> Ensemble method using multiple decision trees with technical indicators.</p>
                      <p><strong>XGBoost:</strong> Gradient boosting algorithm optimized for performance and accuracy.</p>
                      <p><strong>LightGBM:</strong> Fast gradient boosting framework with efficient memory usage.</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            {/* Watchlist Section */}
            <div className="w-full xl:w-3/4 max-h-full overflow-auto bg-white p-3 sm:p-4 lg:p-6 rounded-xl shadow-md space-y-2">
              <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">⭐ Your Watchlist</h3>
              <ul className="space-y-2">
                {watchlist.map(item => (
                  <li
                    key={item.id}
                    className="flex justify-between items-center border-b pb-2 cursor-pointer hover:bg-gray-100 p-2 rounded-md"
                  >
                    {/* Clickable Symbol */}
                    <span
                      className="font-medium text-blue-600 hover:underline text-sm sm:text-base"
                      onClick={async () => {
                        // Clear old data first
                        setData(null);
                        setChartData([]);
                        setNews([]);
                        setError("");

                        // Clear predictions
                        setPredictions([]);
                        setShowPredictions(false);

                        // Set new symbol
                        setSymbol(item.symbol);

                        // Show loading state
                        setLoading(true);

                        try {
                          // Fetch new data
                          await fetchStock(item.symbol, range.value, range.interval);

                          // Fetch news for the new symbol
                          const res = await axios.get(`${API_URL}/news?symbol=${item.symbol}`);
                          setNews(res.data.news || []);

                        } catch (err) {
                          setPopupMessage("⚠️ Error loading stock data. Please try again.");
                          setShowPopup(true);
                          setError("Failed to load stock data: " + err.message);
                        }

                        setLoading(false);
                        setViewMode("chart");
                      }}
                    >
                      {item.symbol}
                    </span>

                    {/* Remove button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation(); // prevent triggering the chart load
                        removeFromWatchlist(item.id);
                      }}
                      className="text-red-500 text-xs sm:text-sm px-2 py-1 rounded hover:bg-red-50"
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>


              {/* Watchlist Add Bar */}
              <div className="flex flex-col sm:flex-row gap-2 mt-4">
                <input
                  type="text"
                  placeholder="Add symbol (e.g. AAPL)"
                  className="flex-1 border border-gray-300 p-2 sm:p-3 rounded-md text-sm sm:text-base"
                  onKeyDown={async (e) => {
                    if (e.key === "Enter") {
                      const symbol = e.target.value.trim().toUpperCase();

                      if (!symbol) {
                        alert("Please enter a symbol");
                        return;
                      }

                      try {
                        const response = await fetch(`${API_URL}/price?symbol=${symbol}`);
                        if (response.ok) {
                          await addToWatchlist(symbol);
                          e.target.value = "";
                        } else {
                          alert("Invalid stock symbol. Please try again.");
                        }
                      } catch (error) {
                        alert("Error validating symbol. Please try again.");
                      }
                    }
                  }}
                />
                <button
                  onClick={async () => {
                    const input = document.querySelector('input[placeholder="Add symbol (e.g. AAPL)"]');
                    const symbol = input.value.trim().toUpperCase();

                    if (!symbol) {
                      alert("Please enter a symbol");
                      return;
                    }

                    // Validate symbol by checking if it exists
                    try {
                      const response = await fetch(`${API_URL}/price?symbol=${symbol}`);
                      if (response.ok) {
                        await addToWatchlist(symbol);
                        input.value = "";
                      } else {
                        alert("Invalid stock symbol. Please try again.");
                      }
                    } catch (error) {
                      alert("Error validating symbol. Please try again.");
                    }
                  }}
                  className="bg-blue-600 text-white px-3 py-2 sm:py-3 rounded-md hover:bg-blue-700 text-sm sm:text-base font-medium"
                >
                  Add
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {showPopup && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-6 rounded-xl shadow-lg max-w-sm text-center">
            <h2 className="text-lg font-semibold mb-2">Important</h2>
            <p className="text-sm text-gray-600 mb-4">{popupMessage}</p>
            <button
              onClick={() => setShowPopup(false)}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
            >
              OK
            </button>
          </div>
        </div>
      )}

      {/* News & Sentiment Card */}
      {news.length > 0 && (
        <div className="flex flex-col xl:flex-row gap-4 overflow-hidden p-2 sm:p-4">
          <div className="bg-white p-3 sm:p-4 lg:p-7 rounded-xl shadow-md w-full">
            <h3 className="text-base sm:text-lg font-semibold text-center mb-3 sm:mb-4">📰 News Sentiment & Headlines</h3>

            <div className="flex flex-col xl:flex-row gap-4 sm:gap-6 p-2 sm:p-3">
              {/* Left side: Sentiment Summary + Chart */}
              <div className="w-full xl:w-1/2 space-y-3 sm:space-y-4 p-1 sm:p-2">
                {/* Sentiment Summary */}
                <div className="flex justify-around text-xs sm:text-sm font-medium">
                  <span className="text-green-600">Positive: {news.filter(n => n.sentiment === "Positive").length}</span>
                  <span className="text-yellow-600">Neutral: {news.filter(n => n.sentiment === "Neutral").length}</span>
                  <span className="text-red-600">Negative: {news.filter(n => n.sentiment === "Negative").length}</span>
                </div>

                {/* Sentiment Chart */}
                <ResponsiveContainer width="100%" height={200} className="sm:h-[250px]">
                  <BarChart data={news.map(n => ({
                    time: new Date(n.published_at).toLocaleTimeString("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                    }),
                    sentiment_score: n.sentiment_score,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={[-1, 1]} />
                    <Tooltip />
                    <Bar dataKey="sentiment_score">
                      {news.map((n, idx) => (
                        <Cell
                          key={idx}
                          fill={
                            n.sentiment === "Positive"
                              ? "#16a34a" // green
                              : n.sentiment === "Negative"
                                ? "#dc2626" // red
                                : "#9ca3af" // gray
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Right side: Headlines List */}
              <div className="w-full xl:w-1/2 p-1 sm:p-2">
                <h4 className="text-sm sm:text-md font-semibold mb-2">Recent Headlines</h4>
                <ul className="space-y-2 sm:space-y-3">
                  {news.map((article, idx) => (
                    <li key={idx} className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b pb-2 gap-2">
                      <div className="flex-1 min-w-0">
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline font-medium text-xs sm:text-sm block truncate"
                        >
                          {article.headline}
                        </a>
                        <p className="text-xs text-gray-500">
                          {new Date(article.published_at).toLocaleString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-semibold whitespace-nowrap ${article.sentiment === "Positive"
                          ? "bg-green-100 text-green-700"
                          : article.sentiment === "Negative"
                            ? "bg-red-100 text-red-700"
                            : "bg-gray-200 text-gray-700"
                          }`}
                      >
                        {article.sentiment}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
