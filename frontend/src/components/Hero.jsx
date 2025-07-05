import { useState, useEffect } from "react";
import { Search, TrendingUp, TrendingDown, ChevronRight, LineChart, BarChart, Users, Clock } from "lucide-react";
import { motion } from "framer-motion";
import axios from "axios";
import { Navigate , useNavigate} from "react-router-dom";

export default function Hero() {

  const navigate = useNavigate();
  const [symbol, setSymbol] = useState("");
  const [isLoaded, setIsLoaded] = useState(false);
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  
  // Initialize with empty array, will be populated by API call
  const [tickerData, setTickerData] = useState([]); 
  
  // State for the dashboard preview stock data
  const [dashboardPreviewStock, setDashboardPreviewStock] = useState({
    symbol: "TSLA",
    price: "$0.00",
    change: "+0.0%",
    trending: "up",
    open: "$0.00",
    volume: "0M",
    high: "$0.00",
    aiPrediction: "HOLD" // Default
  });

  // Simulate loading state for dashboard preview animation
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 800);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Ticker effect - Shift the first item to the end every few seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setTickerData(prevData => {
        if (prevData.length === 0) return prevData; // Do nothing if data is not loaded yet
        const newData = [...prevData];
        const firstItem = newData.shift();
        if (firstItem) newData.push(firstItem);
        return newData;
      });
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);

  // Fetch real-time data for ticker and dashboard preview
  useEffect(() => {
    const fetchRealTimeData = async () => {
      const symbolsToFetch = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT", "META", "NVDA", "JPM"];
      const fetchedTickerData = [];
      
      for (const s of symbolsToFetch) {
        try {
          const response = await axios.get(`http://127.0.0.1:5000/api/stock_info/${s}`);
          if (response.data.success) {
            const data = response.data.data;
            const price = data.current_price ? `$${data.current_price.toFixed(2)}` : 'N/A';
            const changePercent = data.day_change_percent !== null ? `${data.day_change_percent >= 0 ? '+' : ''}${data.day_change_percent.toFixed(2)}%` : 'N/A';
            const trending = data.day_change_percent >= 0 ? "up" : "down";
            
            fetchedTickerData.push({
              symbol: data.symbol,
              price: price,
              change: changePercent,
              trending: trending,
            });

            // If it's TSLA, also update the dashboard preview
            if (s === "TSLA") {
              // Fetch statistics for open, volume, high as well
              const statsResponse = await axios.get(`http://127.0.0.1:5000/api/stock_statistics/${s}`);
              if (statsResponse.data.success) {
                const stats = statsResponse.data.data;
                setDashboardPreviewStock({
                  symbol: data.symbol,
                  price: price,
                  change: changePercent,
                  trending: trending,
                  open: stats.price_stats?.opening ? `$${stats.price_stats.opening.toFixed(2)}` : 'N/A',
                  volume: stats.volume_stats?.total ? `${(stats.volume_stats.total / 1000000).toFixed(1)}M` : 'N/A',
                  high: stats.price_stats?.highest ? `$${stats.price_stats.highest.toFixed(2)}` : 'N/A',
                  aiPrediction: "↗ BUY" // This could also come from an API if available
                });
              }
            }
          }
        } catch (error) {
          console.error(`Error fetching data for ${s}:`, error);
        }
      }
      setTickerData(fetchedTickerData);
    };

    fetchRealTimeData();
    // Refresh data every 5 minutes (adjust as needed)
    const intervalId = setInterval(fetchRealTimeData, 5 * 60 * 1000); 

    return () => clearInterval(intervalId); // Clear interval on unmount
  }, []);


const handleSearch = async () => {
  const trimmedSymbol = symbol.trim().toUpperCase();

  if (!trimmedSymbol) {
    alert("Please enter a stock ticker symbol.");
    return;
  }

  try {
    // First, fetch and store the stock data
    console.log(`Fetching data for ${trimmedSymbol}...`);

    const fetchRes = await axios.post('http://127.0.0.1:5000/stock/fetch', {
      symbol: trimmedSymbol,
      months: 18
    });

    if (fetchRes.data.success) {
      console.log(`Successfully fetched and stored data for ${trimmedSymbol}`);

      // Safely get statistics, or default to an empty object if not present
      // This is the key change to prevent the TypeError
      const fetchedStatistics = fetchRes.data.data && fetchRes.data.data.statistics
        ? fetchRes.data.data.statistics
        : {}; // Provide a default empty object if statistics is undefined

      // Then, get the stored data for display
      const getRes = await axios.get(`http://127.0.0.1:5000/stock/data/${trimmedSymbol}`, {
        params: {
          limit: 365,
          days: 365,
        }
      });

      if (getRes.data.success && getRes.data.data.records.length > 0) {
        // Navigate to stock page with the fetched data
        navigate("/stock", {
          state: {
            stockData: getRes.data.data,
            symbol: trimmedSymbol,
            statistics: fetchedStatistics // Use the safely accessed statistics
          }
        });
      } else {
        alert(`No stored data found for ${trimmedSymbol}. Please try again.`);
      }
    } else {
      alert(fetchRes.data.message || `Failed to fetch data for ${trimmedSymbol}.`);
    }
  } catch (err) {
    console.error("Error fetching stock data:", err);

    // More specific error handling
    if (err.response) {
      // Server responded with error status
      const status = err.response.status;
      const message = err.response.data.message || "Unknown server error";

      if (status === 404) {
        alert(`Stock symbol "${trimmedSymbol}" not found. Please check the symbol and try again.`);
      } else if (status === 400) {
        alert(`Invalid request: ${message}`);
      } else if (status >= 500) {
        alert("Server error. Please try again later.");
      } else {
        alert(`Error: ${message}`);
      }
    } else if (err.request) {
      // Network error
      alert("Network error. Please check your connection and try again.");
    } else {
      alert("An unexpected error occurred. Please try again.");
    }
  }
};

const handleKeyDown = (e) => {
  if (e.key === 'Enter') {
    handleSearch();
  }
};

  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  const stagger = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const scaleIn = {
    hidden: { scale: 0.9, opacity: 0 },
    visible: { 
      scale: 1, 
      opacity: 1,
      transition: { 
        duration: 0.5,
        ease: "easeOut"
      }
    }
  };

  return (
    <section className="relative bg-gradient-to-br from-gray-900 via-gray-850 to-gray-900 text-white py-24 px-6 lg:px-20 overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.1 }}
          transition={{ duration: 1.5 }}
          className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob"
        ></motion.div>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.1 }}
          transition={{ duration: 1.5, delay: 0.2 }}
          className="absolute top-1/3 right-1/4 w-80 h-80 bg-indigo-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000"
        ></motion.div>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.1 }}
          transition={{ duration: 1.5, delay: 0.4 }}
          className="absolute bottom-1/4 right-1/3 w-72 h-72 bg-violet-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-4000"
        ></motion.div>
      </div>
      
      {/* Stock Ticker */}
      <div className="absolute top-0 left-0 w-full bg-gray-900 bg-opacity-80 backdrop-filter backdrop-blur-sm border-b border-gray-800 text-sm font-medium py-2 px-4 overflow-hidden">
        <div className="flex space-x-8 animate-ticker px-4 whitespace-nowrap">
          {tickerData.map((stock, index) => (
            <div key={`${stock.symbol}-${index}`} className="flex items-center">
              <span className="font-semibold">{stock.symbol}:</span>
              <span className="ml-1">{stock.price}</span>
              <span className={`ml-1 flex items-center ${
                stock.trending === "up" ? "text-emerald-400" : "text-rose-400"
              }`}>
                {stock.change}
                {stock.trending === "up" ? (
                  <TrendingUp size={14} className="ml-1" />
                ) : (
                  <TrendingDown size={14} className="ml-1" />
                )}
              </span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Main Content */}
      <div className="pt-16 flex flex-col-reverse lg:flex-row items-center gap-12 relative z-10">
        {/* Left - Textual Content */}
        <motion.div 
          className="flex-1"
          initial="hidden"
          animate="visible"
          variants={stagger}
        >
          <motion.div 
            variants={fadeIn}
            className="inline-block px-3 py-1 mb-6 text-sm font-semibold text-blue-300 bg-blue-900 bg-opacity-30 rounded-full border border-blue-800"
          >
            Stock Analysis Platform
          </motion.div>
          
          <motion.h1 
            variants={fadeIn}
            className="text-4xl lg:text-6xl font-bold leading-tight mb-6 bg-gradient-to-r from-white via-blue-100 to-blue-200 text-transparent bg-clip-text"
          >
            AI-Powered Stock Predictions
          </motion.h1>
          
          <motion.p 
            variants={fadeIn}
            className="text-lg lg:text-xl mb-8 text-gray-300 leading-relaxed max-w-2xl"
          >
            Insights, analytics, and foresight—all in one premium platform.
            Leverage machine learning to stay ahead of market trends and make smarter investments.
          </motion.p>
          
          {/* CTA Buttons */}
          <motion.div 
            variants={stagger}
            className="flex flex-wrap gap-4 mb-8"
          >
            <motion.button 
              variants={fadeIn}
              whileHover={{ y: -4, boxShadow: "0 10px 25px -5px rgba(59, 130, 246, 0.5)" }}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl font-semibold transition-all duration-300 transform focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 group shadow-lg shadow-blue-900/20"
            >
              <span className="flex items-center">
                Get Started
                <ChevronRight size={18} className="ml-1 transition-transform duration-300 group-hover:translate-x-1" />
              </span>
            </motion.button>
          </motion.div>
          
          {/* Search Input */}
          <motion.div 
            variants={fadeIn}
            className="relative max-w-md mb-10"
          >
            <div className={`relative overflow-hidden rounded-xl transition-all duration-200 ${
              isSearchFocused ? "ring-2 ring-blue-500 ring-opacity-60" : ""
            }`}>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => setIsSearchFocused(true)}
                onBlur={() => setIsSearchFocused(false)}
                placeholder="Search Stock Ticker (e.g. AAPL)"
                className="w-full py-4 pl-5 pr-12 bg-gray-800/70 backdrop-blur-sm border border-gray-700 text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none transition-all duration-300"
              />
              <button
                onClick={handleSearch}
                className="absolute right-0 top-0 bottom-0 px-4 bg-blue-600 hover:bg-blue-500 text-white transition-colors duration-300 flex items-center justify-center"
              >
                <Search size={20} />
              </button>
            </div>
            
            {symbol && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-20 backdrop-blur-sm"
              >
                <div className="p-3 border-b border-gray-700">
                  <div className="text-sm text-gray-400">Popular searches</div>
                </div>
                <div className="py-2">
                  {['AAPL', 'MSFT', 'GOOG', 'AMZN'].filter(s => s.includes(symbol.toUpperCase())).map((suggestion) => (
                    <div 
                      key={suggestion} 
                      className="px-3 py-2 hover:bg-gray-700 cursor-pointer flex items-center"
                      onClick={() => setSymbol(suggestion)}
                    >
                      <Search size={14} className="mr-2 text-gray-500" />
                      <span>{suggestion}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
          
          {/* Trust Signals */}
          <motion.div 
            variants={fadeIn}
            className="flex flex-col space-y-4 text-sm text-gray-400"
          >
            <div className="flex items-center space-x-6">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-emerald-400 rounded-full mr-2"></div>
                <span className="font-medium">75%+ prediction accuracy</span>
              </div>
              <div className="flex items-center">
                <Clock size={16} className="mr-2 text-amber-400" />
                <span className="font-medium">1.5+ years of historical data</span>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center">
                <BarChart size={16} className="mr-2 text-purple-400" />
                <span className="font-medium">Powered by LSTM models</span>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Right - Animated Dashboard Preview */}
        <motion.div 
          className="flex-1 w-full relative"
          initial="hidden"
          animate="visible"
          variants={scaleIn}
        >
          <div className="absolute -right-20 -top-20 w-64 h-64 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse"></div>
          
          <div className="relative bg-gray-800/80 rounded-3xl border border-gray-700 shadow-2xl overflow-hidden backdrop-blur-sm">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"></div>
            
            {/* Dashboard Header */}
            <div className="p-4 border-b border-gray-700 flex justify-between items-center">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <div className="text-xs text-gray-400">StockWave Premium Dashboard</div>
              <div className="text-xs text-gray-500">{dashboardPreviewStock.symbol} • Advanced Analytics</div>
            </div>
            
            {/* Dashboard Content */}
            <div className="p-6 h-80">
              {/* Chart UI */}
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-xl font-bold">{dashboardPreviewStock.symbol}</h3>
                  <div className="flex items-center">
                    <span className="text-xl font-semibold mr-2">{dashboardPreviewStock.price}</span>
                    <span className={`flex items-center text-sm font-medium ${
                      dashboardPreviewStock.trending === "up" ? "text-emerald-400" : "text-rose-400"
                    }`}>
                      {dashboardPreviewStock.change} 
                      {dashboardPreviewStock.trending === "up" ? (
                        <TrendingUp size={16} className="ml-1" />
                      ) : (
                        <TrendingDown size={16} className="ml-1" />
                      )}
                    </span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  {["1D", "1W", "1M", "1Y", "ALL"].map((period) => (
                    <button 
                      key={period}
                      className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                        period === "1M" 
                          ? "bg-blue-600 text-white shadow-lg shadow-blue-900/30" 
                          : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                      }`}
                    >
                      {period}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Chart SVG */}
              <div className="h-44 mt-4 relative">
                <svg
                  viewBox="0 0 400 100"
                  className="w-full h-full"
                  preserveAspectRatio="none"
                >
                  {/* Chart gradient background */}
                  <defs>
                    <linearGradient id="chart-gradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.3" />
                      <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  
                  {/* Main chart line */}
                  <motion.path
                    d={`M0,80 C30,75 60,68 100,60 S150,40 200,30 S300,20 350,30 L400,28 V100 H0 Z`}
                    fill="url(#chart-gradient)"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: isLoaded ? 1 : 0 }}
                    transition={{ duration: 1 }}
                  />
                  
                  <motion.path
                    d={`M0,80 C30,75 60,68 100,60 S150,40 200,30 S300,20 350,30 L400,28`}
                    stroke="#3B82F6"
                    strokeWidth="2"
                    fill="none"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: isLoaded ? 1 : 0 }}
                    transition={{ duration: 1.5, ease: "easeInOut" }}
                  />
                  
                  {/* Chart data points */}
                  {isLoaded && (
                    <>
                      <motion.circle 
                        cx="100" cy="60" r="3" fill="#3B82F6"
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.8, duration: 0.4 }}
                      />
                      <motion.circle 
                        cx="200" cy="30" r="3" fill="#3B82F6"
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 1, duration: 0.4 }}
                      />
                      <motion.circle 
                        cx="350" cy="30" r="3" fill="#3B82F6"
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 1.2, duration: 0.4 }}
                      />
                      <circle cx="350" cy="30" r="6" fill="transparent" stroke="#3B82F6" strokeWidth="1" className="animate-ping" />
                    </>
                  )}
                </svg>
                
                {/* Data loading overlay */}
                {!isLoaded && (
                  <div className="absolute inset-0 bg-gray-800 bg-opacity-80 flex items-center justify-center">
                    <div className="w-12 h-12 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
              
              {/* Dashboard Stats */}
              {isLoaded && (
                <motion.div 
                  className="grid grid-cols-4 gap-3 mt-4"
                  initial="hidden"
                  animate="visible"
                  variants={stagger}
                >
                  {[
                    { label: "Open", value: dashboardPreviewStock.open },
                    { label: "Volume", value: dashboardPreviewStock.volume },
                    { label: "High", value: dashboardPreviewStock.high },
                    { label: "AI Prediction", value: dashboardPreviewStock.aiPrediction, highlight: true }
                  ].map((stat, index) => (
                    <motion.div 
                      key={index} 
                      variants={fadeIn}
                      className={`p-3 rounded-lg ${
                        stat.highlight 
                          ? "bg-green-900 bg-opacity-30 border border-green-700" 
                          : "bg-gray-700 bg-opacity-50"
                      }`}
                    >
                      <div className="text-xs text-gray-400">{stat.label}</div>
                      <div className={`text-sm font-bold ${stat.highlight ? "text-green-400" : "text-white"}`}>
                        {stat.value}
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
      
      {/* Add CSS for animations (in a real application this would go in your stylesheet) */}
      <style jsx>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes ticker {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-blob {
          animation: blob 15s infinite alternate;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .animate-ticker {
          animation: ticker 30s linear infinite;
        }
      `}</style>
    </section>
  );
}