import { useEffect, useState } from "react";
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
} from "react-financial-charts";
import { format } from "date-fns";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  Tooltip,
  CartesianGrid,
  XAxis as RechartsXAxis,
  YAxis as RechartsYAxis,
} from "recharts";



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
  { label: "Next Month", value: "month" }, // This will be the default for AI Predict
];

function formatDate(date) {
  if (!date) return "";
  if (typeof date === "string" || typeof date === "number") date = new Date(date);
  return format(date, "MMM dd,yyyy");
}

function formatShortDate(date) {
  if (!date) return "";
  if (typeof date === "string" || typeof date === "number") date = new Date(date);
  return format(date, "MMM dd");
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
  const [predictHorizon, setPredictHorizon] = useState("month"); // Default to "month" for 30 days
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState("candlestick");
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [fetchingData, setFetchingData] = useState(false);

  
 // chartData and chartDataWithPrediction
  const chartData = records.map((r) => ({
    date: new Date(r.date), // Ensure date is a Date object
    open: r.open,
    high: r.high,
    low: r.low,
    close: r.close,
    volume: r.volume,
    predicted: r.predicted || false,
  }));

  const chartDataWithPrediction = [...chartData];
  if (prediction?.close_series?.length > 0) { // Use close_series from the new prediction structure
    // Add predicted data to chartDataWithPrediction
    chartDataWithPrediction.push(
      ...prediction.close_series.map(p => ({ // Use close_series
        date: new Date(p.date),
        open: p.close, // Use close for open, high, low as placeholders for charting
        high: p.close,
        low: p.close,
        close: p.close,
        volume: 0, // Predicted volume is usually 0 or N/A
        predicted: true,
      }))
    );
  }
  
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError("");
      setPrediction(null); // Clear prediction when new historical data is fetched
      try {
        const res = await axios.get(`http://127.0.0.1:5000/stock/data/${symbol}`, {
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
    setPrediction(null); // Clear prediction on new search
    setError("");
  };

  const handleFetchData = async () => {
    setFetchingData(true);
    setError("");
    let stockMarket = 'US'; // Default to US market
    if (symbol.toUpperCase().endsWith('.NS')) {
      stockMarket = 'IN'; // If it ends with .NS, it's Indian
    } else if (symbol.toUpperCase() === 'SBIN') { // Example for SBIN, as it's Indian but might not have .NS entered
      stockMarket = 'IN';
    }
    try {
      const res = await axios.post("http://127.0.0.1:5000/stock/fetch", {
        symbol: symbol,
        months: 24,
        market: stockMarket,
      });
      if (res.data.success) {
        //window.location.reload();
        fetchStockData(symbol);
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
    setPrediction(null); // Clear previous prediction
    setError("");
    try {
      // Always request a "month" (30 days) prediction from the backend
      const res = await axios.get(`http://127.0.0.1:5000/stock/predict/${symbol}?horizon=month`);

      if (res.data.success) {
        setPrediction(res.data.prediction); // Set the entire prediction object
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

 
  // Calculate y-axis extents based on both historical and predicted data
  const allPrices = chartDataWithPrediction.flatMap(d => [d.open, d.high, d.low, d.close]).filter(p => p > 0);
  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);
  const priceRange = maxPrice - minPrice;
  const yAxisMin = Math.max(0, minPrice - priceRange * 0.1);
  const yAxisMax = maxPrice + priceRange * 0.1;

  // Use chartDataWithPrediction for the chart canvas to include predictions
  const xScaleProvider = discontinuousTimeScaleProvider.inputDateAccessor((d) => d.date);
  const { data, xScale, xAccessor, displayXAccessor } = xScaleProvider(chartDataWithPrediction);
    const getMonthlyTickValues = (data) => {
    const tickValues = [];
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
    return tickValues;
  };

  const monthlyTickValues = getMonthlyTickValues(data); 


  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Stock Analysis Dashboard
          </h1>
          <p className="text-gray-400">Advanced LSTM-powered stock prediction and analysis</p>
        </div>

        {/* Controls Panel */}
        <div className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 mb-8 border border-gray-700/50 shadow-2xl">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
              {/* Stock Symbol Input */}
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

              {/* Duration Select */}
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

              {/* Prediction Horizon - Removed as it's now fixed to 30 days for AI Predict */}
               <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Prediction</label>
                <select
                  className="w-full px-4 py-3 rounded-xl bg-gray-700/50 text-white border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200"
                  value={predictHorizon}
                  onChange={(e) => setPredictHorizon(e.target.value)}
                  disabled // Disable as it's fixed for AI Predict
                >
                  {PREDICT_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value} className="bg-gray-800">
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Action Buttons */}
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

            {/* Secondary Controls */}
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
            </div>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 backdrop-blur-lg border border-red-500/50 rounded-xl text-red-200 shadow-lg">
            <div className="flex items-center">
              <span className="text-red-400 mr-2">⚠️</span>
              {error}
            </div>
          </div>
        )}

        {/* Stock Title and Current Info */}
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
                <p>{records.length} data points</p>
              </div>
            </div>
          </div>
        )}

        {/* Statistics Cards */}
        {statistics && (
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-white mb-4">Market Statistics ({duration} Days)</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
              {/* Price Statistics */}
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
              
              {/* Volume and Performance */}
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

        {/* Prediction Results */}
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
                {/* Displaying the first day's prediction for simplicity,
                    as the full series will be shown on the chart */}
                {prediction.predicted_close !== null && ( // Check if predicted_close is not null
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
              {prediction.confidence && ( // Display confidence if available
                <div className="mt-4 text-center">
                  <p className="text-yellow-200 text-sm">Model Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Chart Section */}
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
                        tickFormat={formatShortDate}
                        //ticks={8}
                        tickValues={monthlyTickValues}
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

                      <MouseCoordinateX 
                        displayFormat={formatDate}
                        fill="#374151"
                        stroke="#6B7280"
                        textFill="#E5E7EB"
                        fontSize={12}
                      />

                      <MouseCoordinateY 
                        displayFormat={(v) => `$${v.toFixed(2)}`}
                        fill="#374151"
                        stroke="#6B7280"
                        textFill="#E5E7EB"
                        fontSize={12}
                      />

                      {/* Regular candlesticks */}
                      <CandlestickSeries
                        stroke={(d) => d.predicted ? "#FCD34D" : (d.close > d.open ? "#10B981" : "#EF4444")}
                        wickStroke={(d) => d.predicted ? "#FCD34D" : (d.close > d.open ? "#10B981" : "#EF4444")}
                        fill={(d) => d.predicted ? "#FCD34D" : (d.close > d.open ? "#10B981" : "#EF4444")}
                        opacity={(d) => d.predicted ? 0.9 : 0.8}
                        strokeWidth={(d) => d.predicted ? 2 : 1}
                      />
                      
                      {/* Prediction line for better visibility */}
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
                    </Chart>
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
                        tickFormatter={formatShortDate}
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
                      
                      {/* Historical data line */}
                      <Line
                        type="monotone"
                        dataKey={(d) => d.predicted ? null : d.close}
                        stroke="#3B82F6"
                        strokeWidth={2}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                      />
                      
                      {/* Prediction line */}
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
            </div>
            
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
        )}
      </div>
    </div>
  );
}

export default StockData;