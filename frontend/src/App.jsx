import axios from "axios";
import { useState, useEffect, useCallback, useMemo } from "react";
import { TrendingUp, TrendingDown, Activity, BarChart3, Star, HelpCircle, X, Search, Plus, Trash2, ExternalLink, Clock, Zap, Brain } from "lucide-react";
import { db } from "./firebase";
import { collection, getDocs, addDoc, deleteDoc, doc } from "firebase/firestore";
import { Cell, Bar, BarChart, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Brush, Legend, CartesianGrid, ReferenceDot, Area, AreaChart } from 'recharts';

// Get API URL from environment or use localhost as fallback
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================
// CUSTOM COMPONENTS
// ============================================

// Loading Spinner Component
const LoadingSpinner = ({ size = "md", text = "Loading..." }) => {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8"
  };

  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className={`${sizeClasses[size]} border-2 border-dark-600 border-t-accent-cyan rounded-full animate-spin`} />
      <span className="text-gray-400 text-sm">{text}</span>
    </div>
  );
};

// Skeleton Loader
const Skeleton = ({ className = "" }) => (
  <div className={`skeleton ${className}`} />
);

// Price Change Badge
const PriceChangeBadge = ({ value, showIcon = true }) => {
  const isPositive = value >= 0;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-sm font-mono font-medium ${
      isPositive ? 'bg-gain/20 text-gain' : 'bg-loss/20 text-loss'
    }`}>
      {showIcon && (isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />)}
      {isPositive ? '+' : ''}{value?.toFixed(2)}%
    </span>
  );
};

// Help Modal Component
const HelpModal = ({ open, onClose }) => {
  if (!open) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content max-w-lg" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold gradient-text">About StockView</h2>
          <button onClick={onClose} className="p-2 hover:bg-dark-700 rounded-lg transition-colors">
            <X size={20} className="text-gray-400" />
          </button>
        </div>

        <p className="text-gray-300 mb-6">
          StockView is a real-time stock analysis dashboard with interactive charts,
          ML predictions, and sentiment analysis.
        </p>

        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
              <Zap size={18} className="text-accent-cyan" /> Features
            </h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-accent-cyan mt-1">•</span>
                <span><strong className="text-white">SMA 10:</strong> Simple Moving Average for trend analysis</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-loss mt-1">•</span>
                <span><strong className="text-white">Anomalies:</strong> Unusual price movements detected via Z-score</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-purple mt-1">•</span>
                <span><strong className="text-white">ML Predictions:</strong> Multiple algorithms including XGBoost & CNN</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-gain mt-1">•</span>
                <span><strong className="text-white">Sentiment:</strong> News headlines with AI-powered sentiment scores</span>
              </li>
            </ul>
          </div>

          <div className="pt-4 border-t border-dark-600">
            <h3 className="text-lg font-semibold text-white mb-2">Quick Tips</h3>
            <ul className="space-y-1 text-sm text-gray-400">
              <li>• Use 1D/5D/1M/6M/1Y to switch time ranges</li>
              <li>• Drag the brush below the chart to zoom</li>
              <li>• Click stocks in watchlist to quickly switch</li>
            </ul>
          </div>
        </div>

        <button onClick={onClose} className="btn-primary w-full mt-6">
          Got it
        </button>
      </div>
    </div>
  );
};

// Popup/Toast Component
const Popup = ({ show, message, onClose }) => {
  if (!show) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content text-center" onClick={e => e.stopPropagation()}>
        <p className="text-gray-200 mb-6 whitespace-pre-line">{message}</p>
        <button onClick={onClose} className="btn-primary">
          OK
        </button>
      </div>
    </div>
  );
};

// Custom Tooltip for Charts
const CustomChartTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="glass-card p-4 border-accent-cyan/30 min-w-[200px]">
        <p className="font-semibold text-white mb-2">{data.time}</p>
        <div className="space-y-1.5 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Price:</span>
            <span className="font-mono text-accent-cyan font-semibold">${data.price}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Open:</span>
            <span className="font-mono text-gray-300">${data.open}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">High:</span>
            <span className="font-mono text-gain">${data.high}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Low:</span>
            <span className="font-mono text-loss">${data.low}</span>
          </div>
          {data.sma_10 && (
            <div className="flex justify-between">
              <span className="text-gray-400">SMA 10:</span>
              <span className="font-mono text-chart-orange">${data.sma_10}</span>
            </div>
          )}
          <div className="flex justify-between">
            <span className="text-gray-400">Volume:</span>
            <span className="font-mono text-gray-300">{data.volume?.toLocaleString()}</span>
          </div>
          {data.anomaly && (
            <div className="mt-2 px-2 py-1 bg-loss/20 rounded text-loss text-xs font-semibold">
              ⚠️ Anomaly Detected
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

// Stat Card Component
const StatCard = ({ label, value, prefix = "", suffix = "", mono = true }) => (
  <div className="stat-card">
    <span className="stat-label">{label}</span>
    <span className={`stat-value ${mono ? 'font-mono' : ''}`}>
      {prefix}{typeof value === 'number' ? value.toLocaleString() : value}{suffix}
    </span>
  </div>
);

// Range Button Component
const RangeButton = ({ label, active, onClick }) => (
  <button
    onClick={onClick}
    className={active ? 'range-btn-active' : 'range-btn-inactive'}
  >
    {label}
  </button>
);

// ============================================
// MAIN APP COMPONENT
// ============================================

function App() {
  // State
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
  const [showHelp, setShowHelp] = useState(false);

  // Fetch watchlist
  const fetchWatchlist = async () => {
    const querySnapshot = await getDocs(collection(db, "watchlist"));
    const items = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
    setWatchlist(items);
  };

  // Add to watchlist
  const addToWatchlist = async (sym) => {
    if (!sym) return;
    if (watchlist.some(item => item.symbol === sym)) {
      setPopupMessage("Symbol already in watchlist!");
      setShowPopup(true);
      return;
    }

    try {
      await addDoc(collection(db, "watchlist"), { symbol: sym });
      await fetchWatchlist();
      setPopupMessage(`✅ ${sym} added to watchlist!`);
      setShowPopup(true);
    } catch (error) {
      setPopupMessage("Error adding to watchlist. Please try again.");
      setShowPopup(true);
    }
  };

  // Remove from watchlist
  const removeFromWatchlist = async (id) => {
    await deleteDoc(doc(db, "watchlist", id));
    fetchWatchlist();
  };

  // Fetch stock data
  const fetchStock = useCallback(async (symbolOverride = symbol, selectedRange = range.value, selectedInterval = range.interval) => {
    try {
      setError("");

      const [priceRes, chartRes] = await Promise.all([
        fetch(`${API_URL}/price?symbol=${symbolOverride}`),
        fetch(`${API_URL}/history?symbol=${symbolOverride}&range=${selectedRange}&interval=${selectedInterval}`)
      ]);

      if (!priceRes.ok) {
        const errorText = await priceRes.text();
        throw new Error(`API Error: ${priceRes.status} - ${errorText || 'Stock not found'}`);
      }

      if (!chartRes.ok) {
        const errorText = await chartRes.text();
        throw new Error(`API Error: ${chartRes.status} - ${errorText || 'Chart data not found'}`);
      }

      const newPrice = await priceRes.json();
      const newChart = await chartRes.json();

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
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError(`Cannot connect to backend. Please check if the server is running.`);
      } else {
        setError(err.message || "Failed to fetch stock data");
      }
    }
  }, [symbol, range.value, range.interval]);

  // Search handler
  const handleSearch = async () => {
    if (!symbol.trim()) {
      setData(null);
      setChartData([]);
      setError("Please enter a stock symbol.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      await fetchStock(symbol, range.value, range.interval);
      const res = await axios.get(`${API_URL}/news?symbol=${symbol}`);
      setNews(res.data.news || []);
      setPredictions([]);
      setShowPredictions(false);
    } catch (err) {
      if (err.message && err.message.includes('Network Error')) {
        setPopupMessage(`Cannot connect to backend API.`);
      } else if (err.message && err.message.includes('404')) {
        setPopupMessage("Stock symbol not found. Please enter a valid symbol.");
      } else {
        setPopupMessage(`Error: ${err.message || 'Unknown error occurred'}`);
      }
      setShowPopup(true);
      setError(err.message || "Stock not found or API error");
      setNews([]);
    }
    setLoading(false);
  };

  // Fetch predictions
  const fetchPredictions = useCallback(async (symbolOverride = symbol, algorithm = selectedAlgorithm) => {
    setPredictionLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict?symbol=${symbolOverride}&period=6mo&interval=1d&steps=5&algorithm=${algorithm}`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const predictionData = await response.json();
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

  // Fetch algorithm comparison
  const fetchAlgorithmComparison = useCallback(async (symbolOverride = symbol) => {
    try {
      const response = await fetch(`${API_URL}/predict/compare?symbol=${symbolOverride}&period=6mo&interval=1d&steps=5`);
      if (!response.ok) throw new Error("Failed to fetch algorithm comparison");
      const comparisonData = await response.json();
      setAlgorithmComparison(comparisonData);
    } catch (err) {
      setError("Failed to load algorithm comparison: " + err.message);
    }
  }, [symbol]);

  // Auto-refresh
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (symbol) fetchStock(symbol, range.value, range.interval);
    }, 600000);
    return () => clearInterval(intervalId);
  }, [symbol, range, fetchStock]);

  // Load watchlist on mount
  useEffect(() => {
    fetchWatchlist();
  }, []);

  // Memoized anomaly data
  const anomalyData = useMemo(() => chartData.filter(d => d.anomaly), [chartData]);

  // Range options
  const rangeOptions = [
    { label: "1D", value: "1d", interval: "5m" },
    { label: "5D", value: "5d", interval: "30m" },
    { label: "1M", value: "1mo", interval: "1d" },
    { label: "6M", value: "6mo", interval: "1d" },
    { label: "1Y", value: "1y", interval: "1d" },
  ];

  // Algorithm options
  const algorithmOptions = [
    { value: "linear_regression", label: "Linear Regression" },
    { value: "random_forest", label: "Random Forest" },
    { value: "xgboost", label: "XGBoost" },
    { value: "lightgbm", label: "LightGBM" },
    { value: "cnn", label: "CNN (Deep Learning)" },
  ];

  return (
    <div className="min-h-screen bg-dark-900 bg-grid">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent-cyan/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-chart-purple/5 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-dark-700/50 bg-dark-900/80 backdrop-blur-xl sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-cyan to-chart-purple flex items-center justify-center">
                <Activity size={24} className="text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">StockView</h1>
                <p className="text-xs text-gray-500">Real-Time Analysis</p>
              </div>
            </div>

            {/* Header Actions */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setViewMode(viewMode === "chart" ? "watchlist" : "chart")}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                  viewMode === "watchlist"
                    ? "bg-chart-orange text-dark-900"
                    : "bg-dark-700 text-gray-300 hover:bg-dark-600"
                }`}
              >
                <Star size={18} />
                {viewMode === "chart" ? "Watchlist" : "Back to Chart"}
              </button>
              <button
                onClick={() => setShowHelp(true)}
                className="p-2.5 bg-dark-700 rounded-xl text-gray-400 hover:text-white hover:bg-dark-600 transition-colors"
              >
                <HelpCircle size={20} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <div className="flex flex-col lg:flex-row gap-6">
          
          {/* Sidebar */}
          <aside className="w-full lg:w-72 flex-shrink-0">
            <div className="glass-card p-6 sticky top-24">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <BarChart3 size={20} className="text-accent-cyan" />
                Stock Insights
              </h2>
              
              <p className="text-sm text-gray-400 mb-6">
                Real-time visualization with interactive charts, anomaly detection, and ML predictions.
              </p>

              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Popular Stocks</h3>
                <div className="flex flex-wrap gap-2">
                  {["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"].map(s => (
                    <button
                      key={s}
                      onClick={() => {
                        setSymbol(s);
                        setTimeout(() => handleSearch(), 100);
                      }}
                      className="px-3 py-1.5 text-xs font-mono bg-dark-700 text-gray-300 rounded-lg hover:bg-accent-cyan hover:text-dark-900 transition-all"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">Features</h3>
                <ul className="space-y-2 text-xs text-gray-400">
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-chart-orange" />
                    SMA Trend Lines
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-loss" />
                    Anomaly Detection
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-chart-purple" />
                    ML Price Predictions
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-gain" />
                    Sentiment Analysis
                  </li>
                </ul>
              </div>

              {lastUpdate && (
                <div className="flex items-center gap-2 text-xs text-gray-500 pt-4 border-t border-dark-600">
                  <Clock size={14} />
                  Updated: {lastUpdate}
                </div>
              )}
            </div>
          </aside>

          {/* Main Panel */}
          <div className="flex-1 min-w-0">
            {viewMode === "chart" ? (
              <div className="space-y-6 animate-fade-in">
                {/* Search Bar */}
                <div className="glass-card p-4">
                  <div className="flex gap-3">
                    <div className="relative flex-1">
                      <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" size={20} />
                      <input
                        type="text"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                        onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                        placeholder="Enter stock symbol (e.g., AAPL)"
                        className="input-field pl-12"
                      />
                    </div>
                    <button onClick={handleSearch} className="btn-primary flex items-center gap-2">
                      <Search size={18} />
                      Search
                    </button>
                  </div>
                </div>

                {/* Loading State */}
                {loading && (
                  <div className="glass-card p-12">
                    <LoadingSpinner text="Fetching stock data..." />
                  </div>
                )}

                {/* Error State */}
                {error && (
                  <div className="glass-card p-4 border-loss/30 bg-loss/5">
                    <p className="text-loss text-center">{error}</p>
                  </div>
                )}

                {/* Stock Data Display */}
                {data && !loading && (
                  <>
                    {/* Stock Header */}
                    <div className="glass-card p-6 animate-slide-up">
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
                        <div>
                          <div className="flex items-center gap-3 mb-1">
                            <h2 className="text-2xl font-bold text-white">{data.company}</h2>
                            <span className="px-2 py-0.5 bg-dark-700 rounded text-xs font-mono text-gray-400">
                              {data.symbol}
                            </span>
                          </div>
                          <div className="flex items-baseline gap-3">
                            <span className="text-4xl font-bold font-mono text-white">
                              ${data.price}
                            </span>
                          </div>
                        </div>

                        <button
                          onClick={() => addToWatchlist(symbol)}
                          className="btn-secondary flex items-center gap-2 self-start"
                        >
                          <Star size={18} />
                          Add to Watchlist
                        </button>
                      </div>

                      {/* Stats Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <StatCard label="Open" value={data.open} prefix="$" />
                        <StatCard label="High" value={data.high} prefix="$" />
                        <StatCard label="Low" value={data.low} prefix="$" />
                        <StatCard label="Volume" value={data.volume} />
                      </div>
                    </div>

                    {/* Prediction Controls */}
                    <div className="glass-card p-4 animate-slide-up delay-100">
                      <div className="flex flex-wrap items-center gap-3">
                        <select
                          value={selectedAlgorithm}
                          onChange={(e) => setSelectedAlgorithm(e.target.value)}
                          className="select-field"
                        >
                          {algorithmOptions.map(opt => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>

                        <button
                          onClick={() => {
                            if (!showPredictions) fetchPredictions();
                            setShowPredictions(!showPredictions);
                          }}
                          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium transition-all ${
                            showPredictions
                              ? "bg-chart-purple text-white shadow-lg shadow-chart-purple/25"
                              : "bg-dark-700 text-chart-purple border border-chart-purple/30 hover:bg-chart-purple/10"
                          }`}
                        >
                          <Brain size={18} />
                          {predictionLoading ? "Loading..." : showPredictions ? "Hide Predictions" : "Show Predictions"}
                        </button>

                        <button
                          onClick={() => {
                            if (!showComparison) fetchAlgorithmComparison();
                            setShowComparison(!showComparison);
                          }}
                          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium transition-all ${
                            showComparison
                              ? "bg-gain text-dark-900"
                              : "bg-dark-700 text-gain border border-gain/30 hover:bg-gain/10"
                          }`}
                        >
                          <Zap size={18} />
                          {showComparison ? "Hide Comparison" : "Compare Models"}
                        </button>
                      </div>
                    </div>

                    {/* Range Selector */}
                    <div className="flex gap-2 justify-center animate-slide-up delay-200">
                      {rangeOptions.map(({ label, value, interval }) => (
                        <RangeButton
                          key={value}
                          label={label}
                          active={range.value === value}
                          onClick={() => {
                            setRange({ value, interval });
                            fetchStock(symbol, value, interval);
                          }}
                        />
                      ))}
                    </div>

                    {/* Price Chart */}
                    {chartData.length > 0 && (
                      <div className="chart-container animate-slide-up delay-300">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Activity size={20} className="text-accent-cyan" />
                          Price Chart
                        </h3>
                        
                        <ResponsiveContainer width="100%" height={350}>
                          <AreaChart data={showPredictions && predictions.length > 0 ? predictions : chartData}>
                            <defs>
                              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
                              </linearGradient>
                              <linearGradient id="predictGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#a855f7" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="#a855f7" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(71, 85, 105, 0.3)" />
                            <XAxis
                              dataKey="time"
                              tick={{ fontSize: 11, fill: '#94a3b8' }}
                              tickLine={{ stroke: '#475569' }}
                              axisLine={{ stroke: '#475569' }}
                            />
                            <YAxis
                              domain={['auto', 'auto']}
                              tick={{ fontSize: 11, fill: '#94a3b8' }}
                              tickLine={{ stroke: '#475569' }}
                              axisLine={{ stroke: '#475569' }}
                              tickFormatter={(v) => `$${v}`}
                            />
                            <Tooltip content={<CustomChartTooltip />} />
                            <Legend
                              wrapperStyle={{ paddingTop: '20px' }}
                              formatter={(value) => (
                                <span className="text-gray-300 text-sm">
                                  {value === "sma_10" ? "SMA 10" : value === "price" ? "Price" : value}
                                </span>
                              )}
                            />
                            
                            <Area
                              type="monotone"
                              dataKey="price"
                              stroke="#06b6d4"
                              strokeWidth={2}
                              fill="url(#priceGradient)"
                              dot={false}
                              name="Price"
                            />
                            <Line
                              type="monotone"
                              dataKey="sma_10"
                              stroke="#f97316"
                              strokeWidth={2}
                              dot={false}
                              name="SMA 10"
                            />
                            
                            {showPredictions && (
                              <Area
                                type="monotone"
                                dataKey="predicted"
                                stroke="#a855f7"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                                fill="url(#predictGradient)"
                                dot={{ fill: '#a855f7', r: 4 }}
                                name="Predicted"
                                connectNulls={false}
                              />
                            )}
                            
                            {anomalyData.map((point, index) => (
                              <ReferenceDot
                                key={`anomaly-${point.timestamp || index}`}
                                x={point.time}
                                y={point.price}
                                r={6}
                                fill="#f43f5e"
                                stroke="#fff"
                                strokeWidth={2}
                              />
                            ))}
                            
                            <Brush
                              dataKey="time"
                              height={40}
                              stroke="#06b6d4"
                              fill="#1e293b"
                              tickFormatter={() => ''}
                            />
                          </AreaChart>
                        </ResponsiveContainer>

                        {/* Prediction Info */}
                        {showPredictions && modelMetrics && (
                          <div className="mt-6 p-4 bg-chart-purple/10 border border-chart-purple/30 rounded-xl">
                            <h4 className="font-semibold text-chart-purple mb-3 flex items-center gap-2">
                              <Brain size={18} />
                              ML Prediction Results
                            </h4>
                            <div className="grid grid-cols-3 gap-4 mb-4">
                              <div className="text-center p-3 bg-dark-800 rounded-lg">
                                <p className="text-xs text-gray-400 mb-1">R² Score</p>
                                <p className="font-mono font-bold text-white">{modelMetrics.r2?.toFixed(3)}</p>
                              </div>
                              <div className="text-center p-3 bg-dark-800 rounded-lg">
                                <p className="text-xs text-gray-400 mb-1">MAE</p>
                                <p className="font-mono font-bold text-white">${modelMetrics.mae?.toFixed(2)}</p>
                              </div>
                              <div className="text-center p-3 bg-dark-800 rounded-lg">
                                <p className="text-xs text-gray-400 mb-1">MSE</p>
                                <p className="font-mono font-bold text-white">{modelMetrics.mse?.toFixed(2)}</p>
                              </div>
                            </div>
                            <p className="text-xs text-gray-400 bg-dark-800 p-3 rounded-lg">
                              ⚠️ <strong>Disclaimer:</strong> Predictions are for educational purposes only. Not financial advice.
                            </p>
                          </div>
                        )}

                        {/* Anomaly Summary */}
                        {!showPredictions && anomalyData.length > 0 && (
                          <div className="mt-6 p-4 bg-loss/10 border border-loss/30 rounded-xl">
                            <h4 className="font-semibold text-loss mb-2">⚠️ {anomalyData.length} Anomalies Detected</h4>
                            <p className="text-sm text-gray-400">
                              Significant price deviations from the moving average detected in this timeframe.
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Algorithm Comparison */}
                    {showComparison && algorithmComparison && (
                      <div className="glass-card p-6 animate-slide-up">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Zap size={20} className="text-gain" />
                          Algorithm Comparison
                        </h3>
                        
                        <div className="p-3 bg-gain/10 border border-gain/30 rounded-xl mb-6">
                          <p className="text-gain text-sm">
                            <strong>Best Algorithm:</strong> {algorithmComparison.best_algorithm?.replace('_', ' ').toUpperCase()} 
                            <span className="font-mono ml-2">(R² = {algorithmComparison.best_r2_score?.toFixed(3)})</span>
                          </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {Object.entries(algorithmComparison.comparison).map(([algorithm, metrics]) => (
                            <div
                              key={algorithm}
                              className={`p-4 rounded-xl border ${
                                algorithm === algorithmComparison.best_algorithm
                                  ? 'border-gain/50 bg-gain/5'
                                  : 'border-dark-600 bg-dark-800/50'
                              }`}
                            >
                              <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
                                {algorithm.replace('_', ' ').toUpperCase()}
                                {algorithm === algorithmComparison.best_algorithm && (
                                  <span className="badge-gain text-[10px]">BEST</span>
                                )}
                              </h4>
                              {metrics.error ? (
                                <p className="text-loss text-sm">Error: {metrics.error}</p>
                              ) : (
                                <div className="space-y-2 text-sm">
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">R² Score:</span>
                                    <span className={`font-mono font-medium ${
                                      metrics.r2 > 0.7 ? 'text-gain' : metrics.r2 > 0.4 ? 'text-chart-orange' : 'text-loss'
                                    }`}>
                                      {metrics.r2?.toFixed(3)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">MAE:</span>
                                    <span className="font-mono text-gray-300">${metrics.mae?.toFixed(2)}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">MSE:</span>
                                    <span className="font-mono text-gray-300">{metrics.mse?.toFixed(2)}</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            ) : (
              /* Watchlist View */
              <div className="glass-card p-6 animate-fade-in">
                <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                  <Star size={24} className="text-chart-orange" />
                  Your Watchlist
                </h2>

                {watchlist.length === 0 ? (
                  <div className="text-center py-12">
                    <Star size={48} className="text-dark-600 mx-auto mb-4" />
                    <p className="text-gray-400">Your watchlist is empty</p>
                    <p className="text-sm text-gray-500 mt-2">Add stocks to track them here</p>
                  </div>
                ) : (
                  <div className="space-y-3 mb-6">
                    {watchlist.map(item => (
                      <div key={item.id} className="watchlist-item">
                        <button
                          onClick={async () => {
                            setData(null);
                            setChartData([]);
                            setNews([]);
                            setError("");
                            setPredictions([]);
                            setShowPredictions(false);
                            setSymbol(item.symbol);
                            setLoading(true);

                            try {
                              await fetchStock(item.symbol, range.value, range.interval);
                              const res = await axios.get(`${API_URL}/news?symbol=${item.symbol}`);
                              setNews(res.data.news || []);
                            } catch (err) {
                              setError("Failed to load stock data: " + err.message);
                            }

                            setLoading(false);
                            setViewMode("chart");
                          }}
                          className="font-mono font-semibold text-accent-cyan hover:text-accent-cyan-light transition-colors"
                        >
                          {item.symbol}
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromWatchlist(item.id);
                          }}
                          className="p-2 text-gray-500 hover:text-loss hover:bg-loss/10 rounded-lg transition-all"
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Add to Watchlist */}
                <div className="flex gap-3 pt-4 border-t border-dark-600">
                  <input
                    type="text"
                    placeholder="Add symbol (e.g. NVDA)"
                    className="input-field flex-1"
                    onKeyDown={async (e) => {
                      if (e.key === "Enter") {
                        const sym = e.target.value.trim().toUpperCase();
                        if (!sym) return;
                        try {
                          const response = await fetch(`${API_URL}/price?symbol=${sym}`);
                          if (response.ok) {
                            await addToWatchlist(sym);
                            e.target.value = "";
                          } else {
                            setPopupMessage("Invalid stock symbol");
                            setShowPopup(true);
                          }
                        } catch {
                          setPopupMessage("Error validating symbol");
                          setShowPopup(true);
                        }
                      }
                    }}
                  />
                  <button
                    onClick={async () => {
                      const input = document.querySelector('input[placeholder="Add symbol (e.g. NVDA)"]');
                      const sym = input.value.trim().toUpperCase();
                      if (!sym) return;
                      try {
                        const response = await fetch(`${API_URL}/price?symbol=${sym}`);
                        if (response.ok) {
                          await addToWatchlist(sym);
                          input.value = "";
                        } else {
                          setPopupMessage("Invalid stock symbol");
                          setShowPopup(true);
                        }
                      } catch {
                        setPopupMessage("Error validating symbol");
                        setShowPopup(true);
                      }
                    }}
                    className="btn-primary flex items-center gap-2"
                  >
                    <Plus size={18} />
                    Add
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* News Section */}
        {news.length > 0 && viewMode === "chart" && (
          <div className="mt-8 glass-card p-6 animate-slide-up">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              📰 News & Sentiment Analysis
            </h3>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Sentiment Chart */}
              <div>
                <div className="flex justify-around text-sm font-medium mb-4">
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-gain" />
                    Positive: {news.filter(n => n.sentiment === "Positive").length}
                  </span>
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-gray-500" />
                    Neutral: {news.filter(n => n.sentiment === "Neutral").length}
                  </span>
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-loss" />
                    Negative: {news.filter(n => n.sentiment === "Negative").length}
                  </span>
                </div>

                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={news.map(n => ({
                    time: new Date(n.published_at).toLocaleTimeString("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                    }),
                    sentiment_score: n.sentiment_score,
                    sentiment: n.sentiment
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(71, 85, 105, 0.3)" />
                    <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <YAxis domain={[-1, 1]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="sentiment_score" radius={[4, 4, 0, 0]}>
                      {news.map((n, idx) => (
                        <Cell
                          key={idx}
                          fill={
                            n.sentiment === "Positive" ? "#10b981"
                            : n.sentiment === "Negative" ? "#f43f5e"
                            : "#64748b"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* News Headlines */}
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-4">Recent Headlines</h4>
                <div className="space-y-3 max-h-[250px] overflow-y-auto pr-2">
                  {news.map((article, idx) => (
                    <a
                      key={idx}
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="news-item group"
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-gray-200 group-hover:text-accent-cyan transition-colors line-clamp-2">
                          {article.headline}
                        </p>
                        <p className="text-xs text-gray-500 mt-1 flex items-center gap-2">
                          <Clock size={12} />
                          {new Date(article.published_at).toLocaleString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className={
                          article.sentiment === "Positive" ? "badge-gain"
                          : article.sentiment === "Negative" ? "badge-loss"
                          : "badge-neutral"
                        }>
                          {article.sentiment}
                        </span>
                        <ExternalLink size={14} className="text-gray-500 group-hover:text-accent-cyan" />
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Modals */}
      <HelpModal open={showHelp} onClose={() => setShowHelp(false)} />
      <Popup show={showPopup} message={popupMessage} onClose={() => setShowPopup(false)} />
    </div>
  );
}

export default App;
