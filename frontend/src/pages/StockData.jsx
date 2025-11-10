import { useEffect, useState, useRef } from "react";
import axios from "axios";
import { useLocation } from "react-router-dom";
import {
  ChartCanvas,
  Chart,
  CandlestickSeries,
  XAxis,
  YAxis,
  CrossHairCursor,
  MouseCoordinateX,
  MouseCoordinateY,
  EdgeIndicator,
  LineSeries,
  CurrentCoordinate,
  discontinuousTimeScaleProvider,
  HoverTooltip,
} from "react-financial-charts";
import { format, addDays } from "date-fns";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  Tooltip,
  CartesianGrid,
  XAxis as RechartsXAxis,
  YAxis as RechartsYAxis,
} from "recharts";
import { Heart } from "lucide-react";

const backendUrl =import.meta.env.VITE_APP_BACKEND_URL; 

const DURATION_OPTIONS = [
  { label: "1 Month", value: 30 },
  { label: "3 Months", value: 90 },
  { label: "6 Months", value: 180 },
  { label: "1 Year", value: 365 },
  { label: "2 Years", value: 730 },
];

const PREDICT_OPTIONS = [
  { label: "Next Day", value: "day" },
  { label: "Next Week", value: "week" },
  { label: "Next Month", value: "month" },
];

function formatDate(date) {
  if (!date) return "";
  if (typeof date === "string" || typeof date === "number") date = new Date(date);
  return format(date, "MMM dd,yyyy"); // Formats as "Jan 01,2023"
}

function formatShortDate(date) {
  if (!date) return "";
  if (typeof date === "string" || typeof date === "number") date = new Date(date);
  return format(date, "MMM dd"); // Formats as "Jan 01"
}

function formatMonthYear(date) {
  if (!date) return "";
  if (typeof date === "string" || typeof date === "number") date = new Date(date);
  // Corrected and robust format string to display Month and Year properly
  return format(date, "MMM yyyy"); // Formats as "Jan 2023"
}

function StockData({ width = 1200, ratio = 1 }) {
  const location = useLocation();
  const initialSymbol = location.state?.symbol || "GOOGL";
  const [symbol, setSymbol] = useState(initialSymbol);
  const [search, setSearch] = useState(initialSymbol);
  const [duration, setDuration] = useState(365);
  const [records, setRecords] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictHorizon, setPredictHorizon] = useState("month");
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState("candlestick");
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [fetchingData, setFetchingData] = useState(false);
  const [isLiked, setIsLiked] = useState(false);

  const addSearchedTickerToWatchlist = (tickerSymbol) => {
    try {
      let watchlist = JSON.parse(localStorage.getItem('userWatchlist')) || [];
      if (!watchlist.includes(tickerSymbol)) {
        watchlist = [tickerSymbol, ...watchlist.slice(0, 4)];
        localStorage.setItem('userWatchlist', JSON.stringify(watchlist));
      }
    } catch (e) {
      console.error("Failed to update watchlist in localStorage", e);
    }
  };

  const toggleWishlist = (tickerSymbol) => {
    try {
      let wishlist = JSON.parse(localStorage.getItem('userWishlist')) || [];
      if (wishlist.includes(tickerSymbol)) {
        wishlist = wishlist.filter(s => s !== tickerSymbol);
        setIsLiked(false);
      } else {
        wishlist.push(tickerSymbol);
        setIsLiked(true);
      }
      localStorage.setItem('userWishlist', JSON.stringify(wishlist));
    } catch (e) {
      console.error("Failed to update wishlist in localStorage", e);
    }
  };

  useEffect(() => {
    try {
      const wishlist = JSON.parse(localStorage.getItem('userWishlist')) || [];
      setIsLiked(wishlist.includes(symbol));
    } catch (e) {
      console.error("Failed to read wishlist from localStorage", e);
      setIsLiked(false);
    }
  }, [symbol]);
  
  // Filter records based on selected duration
  const getFilteredRecords = (records, duration) => {
    if (!records || records.length === 0) return [];
    
    // Sort records by date (newest first)
    const sortedRecords = [...records].sort((a, b) => new Date(b.date) - new Date(a.date));
    
    // Take only the number of records specified by duration
    const filteredRecords = sortedRecords.slice(0, duration);
    
    // Sort back to chronological order (oldest first)
    return filteredRecords.sort((a, b) => new Date(a.date) - new Date(b.date));
  };

  const filteredRecords = getFilteredRecords(records, duration);

  const chartData = filteredRecords.map((r) => ({
    date: new Date(r.date),
    open: r.open,
    high: r.high,
    low: r.low,
    close: r.close,
    volume: r.volume,
    predicted: r.predicted || false,
  }));

  const chartDataWithPrediction = [...chartData];
  if (prediction?.close_series?.length > 0) {
    const lastHistoricalDate = chartData.length > 0 ? chartData[chartData.length - 1].date : null;

    const futurePredictions = prediction.close_series.filter(p => {
        const predictionDate = new Date(p.date);
        return lastHistoricalDate ? predictionDate > lastHistoricalDate : true;
    });

    chartDataWithPrediction.push(
      ...futurePredictions.map(p => ({
        date: new Date(p.date),
        open: p.close,
        high: p.close,
        low: p.close,
        close: p.close,
        volume: 0,
        predicted: true,
      }))
    );
  }
  
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError("");
      setPrediction(null);
      try {
        const res = await axios.get(`${backendUrl}/stock/data/${symbol}`, {
          params: { limit: duration, days: duration },
        });
        if (res.data.success) {
          const processedRecords = res.data.data.records
            .map((r) => ({
              ...r,
              date: new Date(r.date),
              open: parseFloat(r.open) || 0,
              high: parseFloat(r.high) || 0,
              low: parseFloat(r.low) || 0,
              close: parseFloat(r.close) || 0,
              volume: parseInt(r.volume) || 0,
            }))
            .sort((a, b) => a.date - b.date);

          setRecords(processedRecords);
          setStatistics(res.data.data.statistics);
          addSearchedTickerToWatchlist(symbol);
        } else {
          setRecords([]);
          setStatistics(null);
          setError(res.data.message || "No data found for this symbol.");
        }
      } catch (err) {
        setRecords([]);
        setStatistics(null);
        setError("Failed to fetch stock data. Please check your connection and try again.");
      }
      setLoading(false);
    }
    fetchData();
  }, [symbol, duration]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!search.trim()) return;
    const newSymbol = search.trim().toUpperCase();
    setSymbol(newSymbol);
    setPrediction(null);
    setError("");
  };

  const handleFetchData = async () => {
    setFetchingData(true);
    setError("");
    let stockMarket = 'US';
    if (symbol.toUpperCase().endsWith('.NS')) {
      stockMarket = 'IN';
    } else if (symbol.toUpperCase() === 'SBIN') {
      stockMarket = 'IN';
    }
    try {
      const res = await axios.post(`${backendUrl}/stock/fetch`, {
        symbol: symbol,
        months: 24,
        market: stockMarket,
      });
      if (res.data.success) {
        fetchData();
        setPrediction(null);
      } else {
        setError(res.data.message || "Failed to fetch fresh data.");
      }
    } catch (err) {
      setError("Failed to fetch fresh data from the market.");
    }
    setFetchingData(false);
  };

  const handlePredict = async () => {
    setPredicting(true);
    setPrediction(null);
    setError("");
    try {
      const res = await axios.get(`${backendUrl}/stock/predict/${symbol}?horizon=month`);

      if (res.data.success) {
        setPrediction(res.data.prediction);
      } else {
        setError(res.data.message || "Prediction failed. Please ensure you have sufficient historical data.");
      }
    } catch (err) {
      setError("Prediction service is currently unavailable.");
    }
    setPredicting(false);
  };

  const handleChartToggle = () => {
    setChartType(chartType === "candlestick" ? "line" : "candlestick");
  };

  const allPrices = chartDataWithPrediction.flatMap(d => [d.open, d.high, d.low, d.close]).filter(p => p > 0);
  const minPrice = allPrices.length > 0 ? Math.min(...allPrices) : 0;
  const maxPrice = allPrices.length > 0 ? Math.max(...allPrices) : 100;
  
  let priceRange = maxPrice - minPrice;

  if (priceRange === 0 && allPrices.length > 0) {
    priceRange = minPrice * 0.1;
    if (priceRange === 0) priceRange = 1;
  } else if (allPrices.length === 0) {
    priceRange = 100;
  }

  const yAxisMin = Math.max(0, minPrice - priceRange * 0.1);
  const yAxisMax = maxPrice + priceRange * 0.1;

  const xScaleProvider = discontinuousTimeScaleProvider.inputDateAccessor((d) => d.date);
  const { data, xScale, xAccessor, displayXAccessor } = xScaleProvider(chartDataWithPrediction);
  const getTickValues = (data, duration) => {
    if (!data || data.length === 0) return [];
    
    const tickValues = [];
    const dataLength = data.length;
    
    if (duration <= 30) {
      // For 1 month or less, show weekly ticks
      const interval = Math.max(1, Math.floor(dataLength / 4));
      for (let i = 0; i < dataLength; i += interval) {
        tickValues.push(data[i].date);
      }
    } else if (duration <= 90) {
      // For 3 months or less, show bi-weekly ticks
      const interval = Math.max(1, Math.floor(dataLength / 6));
      for (let i = 0; i < dataLength; i += interval) {
        tickValues.push(data[i].date);
      }
    } else if (duration <= 180) {
      // For 6 months or less, show monthly ticks
      let currentMonth = -1;
      let currentYear = -1;
      data.forEach((d) => {
        const date = d.date;
        if (date.getMonth() !== currentMonth || date.getFullYear() !== currentYear) {
          tickValues.push(date);
          currentMonth = date.getMonth();
          currentYear = date.getFullYear();
        }
      });
    } else {
      // For longer periods, show quarterly ticks
      const interval = Math.max(1, Math.floor(dataLength / 8));
      for (let i = 0; i < dataLength; i += interval) {
        tickValues.push(data[i].date);
      }
    }
    
    // Always include the last data point
    if (dataLength > 0 && !tickValues.includes(data[dataLength - 1].date)) {
      tickValues.push(data[dataLength - 1].date);
    }
    
    return tickValues;
  };

  const tickValues = getTickValues(data, duration);
  
  // Dynamic tick formatter based on duration
  const getTickFormatter = (duration) => {
    if (duration <= 30) {
      return formatShortDate; // "Jan 01"
    } else if (duration <= 180) {
      return formatMonthYear; // "Jan 2023"
    } else {
      return formatMonthYear; // "Jan 2023"
    }
  };
  
  const tickFormatter = getTickFormatter(duration);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Stock Analysis Dashboard
          </h1>
          <p className="text-gray-400">Advanced LSTM-powered stock prediction and analysis</p>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 mb-8 border border-gray-700/50 shadow-2xl">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
              <div className="lg:col-span-2">
                <label className="block text-sm font-medium text-gray-300 mb-2">Stock Symbol</label>
                <input
                  type="text"
                  className="w-full px-4 py-3 rounded-xl bg-gray-700/50 text-white border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 placeholder-gray-400"
                  placeholder="e.g., AAPL, GOOGL, TSLA"
                  value={search}
                  onChange={(e) => setSearch(e.target.value.toUpperCase())}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Time Period</label>
                <select
                  className="w-full px-4 py-3 rounded-xl bg-gray-700/50 text-white border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                >
                  {DURATION_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value} className="bg-gray-800">
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

                <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Prediction</label>
                <select
                  className="w-full px-4 py-3 rounded-xl bg-gray-700/50 text-white border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200"
                  value={predictHorizon}
                  onChange={(e) => setPredictHorizon(e.target.value)}
                  disabled
                >
                  {PREDICT_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value} className="bg-gray-800">
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="lg:col-span-2 flex gap-2">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? "Loading..." : "Search"}
                </button>
                <button
                  type="button"
                  onClick={handleFetchData}
                  disabled={fetchingData}
                  className="px-4 py-3 rounded-xl bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 disabled:opacity-50"
                >
                  {fetchingData ? "refreshing" : "refresh"}
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-3 pt-4 border-t border-gray-700/50">
              <button
                type="button"
                onClick={handlePredict}
                disabled={predicting || !records.length}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-medium shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predicting ? "🧠 Predicting..." : "🧠 AI Predict"}
              </button>
              <button
                type="button"
                onClick={handleChartToggle}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white font-medium shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
              >
                📊 {chartType === "candlestick" ? "Line Chart" : "Candlestick"}
              </button>
              <button
                type="button"
                onClick={() => toggleWishlist(symbol)}
                className={`px-6 py-2 rounded-lg text-white font-medium shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center
                  ${isLiked ? 'bg-red-600 hover:bg-red-700' : 'bg-gray-600 hover:bg-gray-700'}`}
              >
                <Heart size={18} className="mr-2" fill={isLiked ? "white" : "none"} stroke={isLiked ? "white" : "currentColor"} />
                {isLiked ? "Remove from Wishlist" : "Add to Wishlist"}
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-900/50 backdrop-blur-lg border border-red-500/50 rounded-xl text-red-200 shadow-lg">
            <div className="flex items-center">
              <span className="text-red-400 mr-2">⚠️</span>
              {error}
            </div>
          </div>
        )}

        {records.length > 0 && (
          <div className="mb-8">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-3xl font-bold text-white mb-2">{symbol}</h2>
                {statistics?.price_stats && (
                  <div className="flex items-center space-x-4 text-lg">
                    <span className="text-2xl font-bold text-white">
                      ${statistics.price_stats.current.toFixed(2)}
                    </span>
                    <span className={`font-semibold ${
                      statistics.price_stats.change >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {statistics.price_stats.change >= 0 ? '+' : ''}
                      ${statistics.price_stats.change} ({statistics.price_stats.change_percent}%)
                    </span>
                  </div>
                )}
              </div>
              <div className="text-right text-gray-400 mt-4 md:mt-0">
                <p>Last Updated: {formatDate(new Date())}</p>
                <p>{filteredRecords.length} data points (from {records.length} total)</p>
              </div>
            </div>
          </div>
        )}

        {statistics && (
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-white mb-4">Market Statistics ({duration} Days)</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
              <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 backdrop-blur-lg rounded-xl p-4 border border-blue-500/30">
                <p className="text-blue-300 text-sm font-medium">Current</p>
                <p className="text-white text-xl font-bold">${statistics.price_stats?.current.toFixed(2)}</p>
              </div>
              <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 backdrop-blur-lg rounded-xl p-4 border border-green-500/30">
                <p className="text-green-300 text-sm font-medium">High</p>
                <p className="text-white text-xl font-bold">${statistics.price_stats?.highest.toFixed(2)}</p>
              </div>
              <div className="bg-gradient-to-br from-red-600/20 to-red-800/20 backdrop-blur-lg rounded-xl p-4 border border-red-500/30">
                <p className="text-red-300 text-sm font-medium">Low</p>
                <p className="text-white text-xl font-bold">${statistics.price_stats?.lowest.toFixed(2)}</p>
              </div>
              <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
                <p className="text-purple-300 text-sm font-medium">Average</p>
                <p className="text-white text-xl font-bold">${statistics.price_stats?.average}</p>
              </div>
              
              <div className="bg-gradient-to-br from-yellow-600/20 to-yellow-800/20 backdrop-blur-lg rounded-xl p-4 border border-yellow-500/30">
                <p className="text-yellow-300 text-sm font-medium">Avg Volume</p>
                <p className="text-white text-lg font-bold">
                  {statistics.volume_stats?.average ? 
                    (statistics.volume_stats.average / 1000000).toFixed(1) + 'M' : 'N/A'}
                </p>
              </div>
              <div className="bg-gradient-to-br from-teal-600/20 to-teal-800/20 backdrop-blur-lg rounded-xl p-4 border border-teal-500/30">
                <p className="text-teal-300 text-sm font-medium">Positive Days</p>
                <p className="text-white text-xl font-bold">{statistics.performance_stats?.positive_days || 0}</p>
              </div>
              <div className="bg-gradient-to-br from-orange-600/20 to-orange-800/20 backdrop-blur-lg rounded-xl p-4 border border-orange-500/30">
                <p className="text-orange-300 text-sm font-medium">Negative Days</p>
                <p className="text-white text-xl font-bold">{statistics.performance_stats?.negative_days || 0}</p>
              </div>
              <div className="bg-gradient-to-br from-indigo-600/20 to-indigo-800/20 backdrop-blur-lg rounded-xl p-4 border border-indigo-500/30">
                <p className="text-indigo-300 text-sm font-medium">Win Rate</p>
                <p className="text-white text-xl font-bold">{statistics.performance_stats?.positive_ratio || 0}%</p>
              </div>
            </div>
          </div>
        )}

        {prediction && (
          <div className="mb-8">
            <div className="bg-gradient-to-r from-yellow-600/20 via-yellow-500/20 to-amber-600/20 backdrop-blur-lg rounded-2xl p-6 border border-yellow-500/30 shadow-2xl">
              <div className="flex items-center mb-4">
                <span className="text-2xl mr-3">🎯</span>
                <h3 className="text-xl font-bold text-white">
                  LSTM AI Prediction ({PREDICT_OPTIONS.find(opt => opt.value === predictHorizon)?.label})
                </h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {prediction.predicted_close !== null && (
                  <>
                    <div className="text-center">
                      <p className="text-yellow-200 text-sm font-medium">Predicted Close (Day 1)</p>
                      <p className="text-white text-2xl font-bold">${prediction.predicted_close}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-yellow-200 text-sm font-medium">Predicted Open (Day 1)</p>
                      <p className="text-white text-2xl font-bold">${prediction.predicted_open}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-yellow-200 text-sm font-medium">Predicted High (Day 1)</p>
                      <p className="text-white text-2xl font-bold">${prediction.predicted_high}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-yellow-200 text-sm font-medium">Predicted Low (Day 1)</p>
                      <p className="text-white text-2xl font-bold">${prediction.predicted_low}</p>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {records.length > 0 && (
          <div className="bg-gray-800/30 backdrop-blur-lg rounded-2xl p-6 border border-gray-700/50 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white">
                {chartType === "candlestick" ? "📊 Candlestick Chart" : "📈 Price Trend"}
              </h3>
              <div className="text-sm text-gray-400">
                {prediction && "🟡 Yellow = AI Prediction"}
              </div>
            </div>
            
            <div className="w-full overflow-x-auto">
              {chartType === "candlestick" && data.length > 0 && (
                <div className="min-w-[800px]">
                  <ChartCanvas
                    height={500}
                    width={width}
                    ratio={ratio}
                    margin={{ left: 80, right: 80, top: 20, bottom: 60 }}
                    seriesName={symbol}
                    data={data}
                    xScale={xScale}
                    xAccessor={xAccessor}
                    displayXAccessor={displayXAccessor}
                    xExtents={[xAccessor(data[0]), xAccessor(data[data.length - 1])]}
                  >
                    <Chart 
                      id={1} 
                      yExtents={[yAxisMin, yAxisMax]}
                      padding={{ top: 10, bottom: 10 }}
                    >
                      <XAxis
                        axisAt="bottom"
                        orient="bottom"
                        tickFormat={tickFormatter} 
                        tickValues={tickValues}
                        stroke="#9CA3AF"
                        tickStroke="#9CA3AF"
                        fontSize={11}
                        fontFamily="ui-sans-serif, system-ui, sans-serif" 
                      />
                      <YAxis
                        axisAt="left"
                        orient="left"
                        stroke="#9CA3AF"
                        tickStroke="#9CA3AF"
                        fontSize={11}
                        fontFamily="ui-sans-serif, system-ui, sans-serif" 
                        tickFormat={(d) => `$${d.toFixed(2)}`}
                      />

                      {/* MouseCoordinateX and MouseCoordinateY provide the hover functionality */}
                      <MouseCoordinateX 
                        displayFormat={formatDate} // Shows full date on X-axis hover
                        fill="#374151" // Background color of the coordinate display
                        stroke="#6B7280" // Border color
                        textFill="#E5E7EB" // Text color
                        fontSize={12}
                      />

                      <MouseCoordinateY 
                        displayFormat={(v) => `$${v.toFixed(2)}`} // Shows price value on Y-axis hover
                        fill="#374151" // Background color of the coordinate display
                        stroke="#6B7280" // Border color
                        textFill="#E5E7EB" // Text color
                        fontSize={12}
                      />

                      <CandlestickSeries
                        stroke={(d) => d.predicted ? "#FCD34D" : (d.close > d.open ? "#10B981" : "#EF4444")}
                        wickStroke={(d) => d.predicted ? "#FCD34D" : (d.close > d.open ? "#10B981" : "#EF4444")}
                        fill={(d) => d.predicted ? "#FCD34D" : (d.close > d.open ? "#10B981" : "#EF4444")}
                        opacity={(d) => d.predicted ? 0.9 : 0.8}
                        strokeWidth={(d) => d.predicted ? 2 : 1}
                      />
                      
                      {prediction && (
                        <LineSeries
                          yAccessor={(d) => d.predicted ? d.close : null}
                          stroke="#FCD34D"
                          strokeWidth={3}
                          strokeDasharray="5,5"
                          highlightOnHover={true}
                        />
                      )}
                      
                      <EdgeIndicator
                        itemType="last"
                        orient="right"
                        edgeAt="right"
                        yAccessor={(d) => d.close}
                        fill="#3B82F6"
                        stroke="#3B82F6"
                        strokeWidth={2}
                        textFill="#FFFFFF"
                        fontSize={12}
                      />
                      <CurrentCoordinate 
                        yAccessor={(d) => d.close} 
                        fill="#3B82F6"
                        stroke="#3B82F6"
                        strokeWidth={2}
                        r={4}
                      />
                      
                      <HoverTooltip
                        yAccessor={(d) => {
                          return {
                            date: d.date,
                            open: d.open,
                            high: d.high,
                            low: d.low,
                            close: d.close,
                            volume: d.volume,
                            predicted: d.predicted
                          };
                        }}
                        tooltipContent={({ currentItem }) => {
                          if (!currentItem) return null;
                          const data = currentItem;
                          return (
                            <div style={{
                              backgroundColor: '#1F2937',
                              border: '1px solid #374151',
                              borderRadius: '8px',
                              padding: '12px',
                              color: '#E5E7EB',
                              fontSize: '12px',
                              minWidth: '180px'
                            }}>
                              <div style={{ fontWeight: 'bold', marginBottom: '8px', color: data.predicted ? '#FCD34D' : '#3B82F6' }}>
                                {formatDate(data.date)}
                                {data.predicted && ' (Predicted)'}
                              </div>
                              <div>Open: ${data.open?.toFixed(2)}</div>
                              <div>High: ${data.high?.toFixed(2)}</div>
                              <div>Low: ${data.low?.toFixed(2)}</div>
                              <div>Close: ${data.close?.toFixed(2)}</div>
                              {!data.predicted && <div>Volume: {data.volume?.toLocaleString()}</div>}
                            </div>
                          );
                        }}
                        fontSize={12}
                        fill="#1F2937"
                        stroke="#374151"
                      />
                    </Chart>
                    {/* CrossHairCursor displays the intersecting lines on hover */}
                    <CrossHairCursor stroke="#6B7280" strokeWidth={1} />
                  </ChartCanvas>
                </div>
              )}
              
              {chartType === "line" && chartDataWithPrediction.length > 0 && (
                <div className="min-w-[800px]">
                  <ResponsiveContainer width="100%" height={500}>
                    <LineChart 
                      data={chartDataWithPrediction} 
                      margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                    >
                      <RechartsXAxis
                        dataKey="date"
                        tickFormatter={tickFormatter}
                        tick={{ fill: "#9CA3AF", fontSize: 11 }}
                        stroke="#6B7280"
                        interval="preserveStartEnd"
                        minTickGap={50}
                      />
                      <RechartsYAxis
                        domain={[yAxisMin, yAxisMax]}
                        tick={{ fill: "#9CA3AF", fontSize: 11 }}
                        stroke="#6B7280"
                        tickFormatter={(value) => `$${value.toFixed(2)}`}
                      />
                      <Tooltip
                        labelFormatter={(value) => formatDate(value)}
                        formatter={(value, name) => [
                          value ? `$${parseFloat(value).toFixed(2)}` : 'N/A',
                          name === "close" ? "Close Price" : name
                        ]}
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                          color: '#E5E7EB'
                        }}
                      />

                      <CartesianGrid stroke="#374151" strokeDasharray="3 3" />
                      
                      <Line
                        type="monotone"
                        dataKey={(d) => d.predicted ? null : d.close}
                        stroke="#3B82F6"
                        strokeWidth={2}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                      />
                      
                      {prediction && (
                        <Line
                          type="monotone"
                          dataKey={(d) => d.predicted ? d.close : null}
                          stroke="#FCD34D"
                          strokeWidth={3}
                          strokeDasharray="8 4"
                          dot={{ fill: "#FCD34D", strokeWidth: 2, r: 4 }}
                          connectNulls={false}
                          isAnimationActive={false}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              
              {records.length === 0 && !loading && (
                <div className="text-center py-12">
                  <p className="text-gray-400 text-lg">No data available for {symbol}</p>
                  <p className="text-gray-500 text-sm mt-2">Try fetching fresh data or selecting a different symbol</p>
                </div>
              )}
              
              {loading && (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  <p className="text-gray-400 mt-4">Loading chart data...</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StockData;