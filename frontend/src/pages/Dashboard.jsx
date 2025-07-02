import { useState, useEffect } from "react";
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  Bell, 
  Search,
  Filter, 
  Calendar, 
  PieChart, 
  BarChart3, 
  LineChart, 
  Zap, 
  TrendingUp, 
  ChevronDown, 
  Users,
  RefreshCw,
  Check,
  Plus
} from "lucide-react";
import { motion } from "framer-motion";

import NavBar from "../components/NavBar";
import Footer from "../components/Footer";

// Reusable animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.5 }
  }
};

const staggerChildren = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function Dashboard() {
  // State management
  const [selectedPeriod, setSelectedPeriod] = useState("1M");
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [searchQuery, setSearchQuery] = useState("");
  const [notifications, setNotifications] = useState(3);
  const [watchlist, setWatchlist] = useState([
    { symbol: "AAPL", name: "Apple Inc.", price: "$175.23", change: "+2.3%", trending: "up", prediction: "BUY", score: 89 },
    { symbol: "MSFT", name: "Microsoft Corp.", price: "$336.75", change: "+1.8%", trending: "up", prediction: "HOLD", score: 74 },
    { symbol: "NVDA", name: "NVIDIA Corp.", price: "$781.50", change: "+5.6%", trending: "up", prediction: "STRONG BUY", score: 96 },
    { symbol: "GOOG", name: "Alphabet Inc.", price: "$2,512.44", change: "-0.8%", trending: "down", prediction: "HOLD", score: 68 },
    { symbol: "TSLA", name: "Tesla Inc.", price: "$846.90", change: "+4.2%", trending: "up", prediction: "BUY", score: 85 }
  ]);
  
  const [portfolioData, setPortfolioData] = useState({
    totalValue: "$284,531.20",
    dayChange: "+$5,892.44 (2.1%)",
    trending: "up",
    allocation: [
      { category: "Technology", percentage: 42, color: "#3B82F6" },
      { category: "Healthcare", percentage: 18, color: "#10B981" },
      { category: "Consumer", percentage: 15, color: "#F59E0B" },
      { category: "Financial", percentage: 12, color: "#8B5CF6" },
      { category: "Energy", percentage: 8, color: "#EC4899" },
      { category: "Other", percentage: 5, color: "#6B7280" }
    ],
    performance: [
      { date: "Jan", value: 240000 },
      { date: "Feb", value: 232000 },
      { date: "Mar", value: 245000 },
      { date: "Apr", value: 251000 },
      { date: "May", value: 260000 },
      { date: "Jun", value: 272000 },
      { date: "Jul", value: 284531 }
    ]
  });
  
  // Market insights data
  const [marketInsights, setMarketInsights] = useState([
    { 
      title: "Tech Sector Outperformance", 
      description: "Technology stocks continue to outperform broader market with an average gain of 3.4% this week.",
      category: "Sector Analysis",
      timestamp: "2h ago"
    },
    { 
      title: "Fed Interest Rate Decision", 
      description: "Federal Reserve signals potential rate cut in upcoming meeting, boosting market sentiment.",
      category: "Macro Events",
      timestamp: "4h ago"
    },
    { 
      title: "NVDA Technical Breakout", 
      description: "NVIDIA showing strong bullish pattern after breaking key resistance level at $750.",
      category: "Technical Alert",
      timestamp: "1d ago"
    }
  ]);
  
  // AI predictions data
  const [aiPredictions, setAiPredictions] = useState([
    { symbol: "AAPL", confidence: 89, direction: "up", timeframe: "1 month", reasoning: "Strong product cycle and services growth" },
    { symbol: "JPM", confidence: 76, direction: "up", timeframe: "3 months", reasoning: "Favorable interest rate environment" },
    { symbol: "TSLA", confidence: 82, direction: "up", timeframe: "1 month", reasoning: "Production ramp and margin improvement" },
    { symbol: "AMZN", confidence: 91, direction: "up", timeframe: "3 months", reasoning: "AWS growth acceleration and retail margin expansion" }
  ]);
  
  // Chart data
  const [chartData, setChartData] = useState({
    points: "M0,80 C30,75 60,68 100,60 S150,40 200,30 S300,20 350,30 L400,28",
    area: "M0,80 C30,75 60,68 100,60 S150,40 200,30 S300,20 350,30 L400,28 V100 H0 Z",
    dataPoints: [
      { x: 100, y: 60 },
      { x: 200, y: 30 },
      { x: 350, y: 30 }
    ]
  });
  
  // Simulate loading state
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1200);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Filter watchlist based on search query
  const filteredWatchlist = watchlist.filter(stock => 
    stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) || 
    stock.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <><NavBar /><section className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white min-h-screen py-6 px-4 lg:px-10">
          {/* Background decorative elements */}
          <div className="absolute inset-0 overflow-hidden">
              <div className="absolute top-1/3 left-1/5 w-64 h-64 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob"></div>
              <div className="absolute top-2/3 right-1/4 w-80 h-80 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000"></div>
              <div className="absolute bottom-1/3 right-1/3 w-72 h-72 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-4000"></div>
          </div>

          {/* Dashboard Header */}
          <div className="relative z-10">
              <motion.div
                  className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-8 gap-4"
                  initial="hidden"
                  animate="visible"
                  variants={staggerChildren}
              >
                  <motion.div variants={fadeInUp}>
                      <h1 className="text-3xl font-bold bg-gradient-to-r from-white via-blue-100 to-blue-200 text-transparent bg-clip-text">Market Dashboard</h1>
                      <p className="text-gray-400">Friday, May 17, 2025 • Market is open</p>
                  </motion.div>

                  <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                      <motion.div
                          className="relative"
                          variants={fadeInUp}
                      >
                          <input
                              type="text"
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              placeholder="Search stocks..."
                              className="w-full sm:w-64 py-2 pl-10 pr-4 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-30 transition-all duration-300" />
                          <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                      </motion.div>

                      <motion.div variants={fadeInUp} className="flex items-center gap-3">
                          <button className="p-2 bg-gray-800 border border-gray-700 rounded-xl text-gray-400 hover:text-white hover:border-blue-500 transition-colors duration-300 relative">
                              <Bell size={20} />
                              {notifications > 0 && (
                                  <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 rounded-full text-xs flex items-center justify-center transform translate-x-1 -translate-y-1">
                                      {notifications}
                                  </span>
                              )}
                          </button>

                          <button className="p-2 bg-gray-800 border border-gray-700 rounded-xl text-gray-400 hover:text-white hover:border-blue-500 transition-colors duration-300">
                              <Filter size={20} />
                          </button>

                          <button className="p-2 bg-gray-800 border border-gray-700 rounded-xl text-gray-400 hover:text-white hover:border-blue-500 transition-colors duration-300">
                              <Calendar size={20} />
                          </button>

                          <button className="p-2 bg-blue-600 hover:bg-blue-500 rounded-xl text-white transition-colors duration-300 flex items-center">
                              <RefreshCw size={20} className="animate-slow-spin" />
                          </button>
                      </motion.div>
                  </div>
              </motion.div>

              {/* Dashboard Navigation Tabs */}
              <motion.div
                  className="flex overflow-x-auto scrollbar-hide space-x-2 pb-2 mb-6"
                  initial="hidden"
                  animate="visible"
                  variants={fadeInUp}
              >
                  {[
                      { id: "overview", label: "Overview", icon: PieChart },
                      { id: "portfolio", label: "Portfolio", icon: BarChart3 },
                      { id: "predictions", label: "AI Predictions", icon: Zap },
                      { id: "watchlist", label: "Watchlist", icon: LineChart },
                      { id: "insights", label: "Market Insights", icon: TrendingUp },
                      { id: "community", label: "Community", icon: Users }
                  ].map((tab) => {
                      const Icon = tab.icon;
                      return (
                          <button
                              key={tab.id}
                              onClick={() => setActiveTab(tab.id)}
                              className={`flex items-center px-4 py-2 rounded-xl whitespace-nowrap transition-all duration-300 ${activeTab === tab.id
                                      ? "bg-blue-600 text-white font-medium"
                                      : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                          >
                              <Icon size={16} className="mr-2" />
                              {tab.label}
                          </button>
                      );
                  })}
              </motion.div>
          </div>

          {/* Main Dashboard Area */}
          <div className="relative z-10">
              {/* Dashboard Grid Layout */}
              <motion.div
                  className="grid grid-cols-1 lg:grid-cols-12 gap-6"
                  initial="hidden"
                  animate="visible"
                  variants={staggerChildren}
              >
                  {/* Portfolio Summary Card */}
                  <motion.div
                      className="col-span-1 lg:col-span-8 bg-gray-800 bg-opacity-60 backdrop-filter backdrop-blur-md rounded-2xl border border-gray-700 overflow-hidden"
                      variants={fadeInUp}
                  >
                      <div className="p-6">
                          <div className="flex justify-between items-start mb-6">
                              <div>
                                  <h2 className="text-xl font-bold mb-1">Portfolio Summary</h2>
                                  <div className="flex items-baseline">
                                      <span className="text-2xl font-semibold mr-3">{portfolioData.totalValue}</span>
                                      <span className={`flex items-center text-sm font-medium ${portfolioData.trending === "up" ? "text-green-400" : "text-red-400"}`}>
                                          {portfolioData.dayChange}
                                          {portfolioData.trending === "up" ? (
                                              <ArrowUpRight size={16} className="ml-1" />
                                          ) : (
                                              <ArrowDownRight size={16} className="ml-1" />
                                          )}
                                      </span>
                                  </div>
                              </div>

                              <div className="flex space-x-2">
                                  {["1D", "1W", "1M", "3M", "1Y", "ALL"].map((period) => (
                                      <button
                                          key={period}
                                          onClick={() => setSelectedPeriod(period)}
                                          className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${selectedPeriod === period
                                                  ? "bg-blue-600 text-white"
                                                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
                                      >
                                          {period}
                                      </button>
                                  ))}
                              </div>
                          </div>

                          {/* Portfolio Chart */}
                          <div className="h-60 relative">
                              {!isLoaded ? (
                                  <div className="absolute inset-0 flex items-center justify-center">
                                      <div className="w-12 h-12 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
                                  </div>
                              ) : (
                                  <>
                                      <svg
                                          viewBox="0 0 400 100"
                                          className="w-full h-full"
                                          preserveAspectRatio="none"
                                      >
                                          {/* Chart gradient background */}
                                          <defs>
                                              <linearGradient id="portfolio-gradient" x1="0" y1="0" x2="0" y2="1">
                                                  <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.4" />
                                                  <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
                                              </linearGradient>
                                          </defs>

                                          {/* Chart area fill with animation */}
                                          <motion.path
                                              d={chartData.area}
                                              fill="url(#portfolio-gradient)"
                                              initial={{ opacity: 0 }}
                                              animate={{ opacity: 1 }}
                                              transition={{ duration: 1 }} />

                                          {/* Chart line with drawing animation */}
                                          <motion.path
                                              d={chartData.points}
                                              stroke="#3B82F6"
                                              strokeWidth="2"
                                              fill="none"
                                              initial={{ pathLength: 0, opacity: 0 }}
                                              animate={{ pathLength: 1, opacity: 1 }}
                                              transition={{ duration: 1.5, ease: "easeInOut" }} />

                                          {/* Chart data points with animations */}
                                          {chartData.dataPoints.map((point, index) => (
                                              <motion.circle
                                                  key={index}
                                                  cx={point.x}
                                                  cy={point.y}
                                                  r="3"
                                                  fill="#3B82F6"
                                                  initial={{ scale: 0, opacity: 0 }}
                                                  animate={{ scale: 1, opacity: 1 }}
                                                  transition={{ delay: 0.8 + (index * 0.2) }} />
                                          ))}

                                          {/* Animated pulse on the latest data point */}
                                          <motion.circle
                                              cx={chartData.dataPoints[chartData.dataPoints.length - 1].x}
                                              cy={chartData.dataPoints[chartData.dataPoints.length - 1].y}
                                              r="6"
                                              fill="transparent"
                                              stroke="#3B82F6"
                                              strokeWidth="1.5"
                                              initial={{ scale: 0.8, opacity: 0 }}
                                              animate={{ scale: 1.5, opacity: 0 }}
                                              transition={{
                                                  repeat: Infinity,
                                                  duration: 2,
                                                  repeatType: "loop"
                                              }} />
                                      </svg>

                                      {/* Performance metrics below chart */}
                                      <div className="grid grid-cols-4 mt-4 gap-4">
                                          {[
                                              { label: "Day High", value: "$287,102.31" },
                                              { label: "Day Low", value: "$281,223.65" },
                                              { label: "Week Change", value: "+4.2%" },
                                              { label: "Month Change", value: "+12.7%" }
                                          ].map((metric, index) => (
                                              <motion.div
                                                  key={index}
                                                  className="bg-gray-700 bg-opacity-50 rounded-xl p-3"
                                                  initial={{ opacity: 0, y: 10 }}
                                                  animate={{ opacity: 1, y: 0 }}
                                                  transition={{ delay: 1 + (index * 0.1) }}
                                              >
                                                  <div className="text-xs text-gray-400">{metric.label}</div>
                                                  <div className="text-sm font-bold">{metric.value}</div>
                                              </motion.div>
                                          ))}
                                      </div>
                                  </>
                              )}
                          </div>
                      </div>
                  </motion.div>

                  {/* Portfolio Allocation */}
                  <motion.div
                      className="col-span-1 lg:col-span-4 bg-gray-800 bg-opacity-60 backdrop-filter backdrop-blur-md rounded-2xl border border-gray-700 overflow-hidden"
                      variants={fadeInUp}
                  >
                      <div className="p-6">
                          <div className="flex justify-between items-center mb-4">
                              <h2 className="text-xl font-bold">Asset Allocation</h2>
                              <button className="text-xs text-gray-400 hover:text-white flex items-center">
                                  View Details
                                  <ChevronDown size={14} className="ml-1" />
                              </button>
                          </div>

                          {!isLoaded ? (
                              <div className="h-60 flex items-center justify-center">
                                  <div className="w-10 h-10 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
                              </div>
                          ) : (
                              <div className="relative h-60">
                                  {/* SVG Donut Chart with Animation */}
                                  <svg viewBox="0 0 100 100" className="w-full">
                                      {portfolioData.allocation.map((segment, index) => {
                                          // Calculate the segment positions in the donut chart
                                          const total = portfolioData.allocation.reduce((sum, item) => sum + item.percentage, 0);
                                          const startAngle = portfolioData.allocation
                                              .slice(0, index)
                                              .reduce((sum, item) => sum + (item.percentage / total) * 360, 0);
                                          const endAngle = startAngle + (segment.percentage / total) * 360;

                                          // Convert angles to radians and calculate path
                                          const startRad = (startAngle - 90) * Math.PI / 180;
                                          const endRad = (endAngle - 90) * Math.PI / 180;

                                          const x1 = 50 + 30 * Math.cos(startRad);
                                          const y1 = 50 + 30 * Math.sin(startRad);
                                          const x2 = 50 + 30 * Math.cos(endRad);
                                          const y2 = 50 + 30 * Math.sin(endRad);

                                          const largeArcFlag = endAngle - startAngle <= 180 ? 0 : 1;

                                          const pathData = [
                                              `M 50 50`,
                                              `L ${x1} ${y1}`,
                                              `A 30 30 0 ${largeArcFlag} 1 ${x2} ${y2}`,
                                              `L 50 50`
                                          ].join(' ');

                                          return (
                                              <motion.path
                                                  key={segment.category}
                                                  d={pathData}
                                                  fill={segment.color}
                                                  initial={{ opacity: 0 }}
                                                  animate={{ opacity: 1 }}
                                                  transition={{ duration: 0.5, delay: index * 0.1 + 0.2 }} />
                                          );
                                      })}

                                      {/* Inner circle for donut effect */}
                                      <circle cx="50" cy="50" r="20" fill="#1F2937" />

                                      {/* Central text */}
                                      <text x="50" y="48" textAnchor="middle" fill="white" fontSize="6" fontWeight="bold">
                                          Portfolio
                                      </text>
                                      <text x="50" y="55" textAnchor="middle" fill="#9CA3AF" fontSize="4">
                                          Allocation
                                      </text>
                                  </svg>

                                  {/* Legend */}
                                  <div className="grid grid-cols-2 gap-x-2 gap-y-1 mt-4">
                                      {portfolioData.allocation.map((segment, index) => (
                                          <motion.div
                                              key={segment.category}
                                              className="flex items-center text-xs"
                                              initial={{ opacity: 0, x: -10 }}
                                              animate={{ opacity: 1, x: 0 }}
                                              transition={{ delay: 0.8 + (index * 0.1) }}
                                          >
                                              <div
                                                  className="w-3 h-3 rounded-sm mr-2"
                                                  style={{ backgroundColor: segment.color }}
                                              ></div>
                                              <span className="text-gray-300">{segment.category}</span>
                                              <span className="ml-1 text-gray-400">{segment.percentage}%</span>
                                          </motion.div>
                                      ))}
                                  </div>
                              </div>
                          )}
                      </div>
                  </motion.div>

                  {/* AI Predictions */}
                  <motion.div
                      className="col-span-1 lg:col-span-6 bg-gray-800 bg-opacity-60 backdrop-filter backdrop-blur-md rounded-2xl border border-gray-700 overflow-hidden"
                      variants={fadeInUp}
                  >
                      <div className="p-6">
                          <div className="flex justify-between items-center mb-4">
                              <div className="flex items-center">
                                  <Zap size={20} className="text-amber-400 mr-2" />
                                  <h2 className="text-xl font-bold">AI Predictions</h2>
                              </div>
                              <button className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors">
                                  View All
                              </button>
                          </div>

                          {!isLoaded ? (
                              <div className="h-64 flex items-center justify-center">
                                  <div className="w-10 h-10 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
                              </div>
                          ) : (
                              <div className="space-y-4">
                                  {aiPredictions.map((prediction, index) => (
                                      <motion.div
                                          key={prediction.symbol}
                                          className="bg-gray-700 bg-opacity-40 rounded-xl p-4 border border-gray-600 hover:border-blue-500 transition-colors"
                                          initial={{ opacity: 0, y: 20 }}
                                          animate={{ opacity: 1, y: 0 }}
                                          transition={{ delay: 0.3 + (index * 0.1) }}
                                          whileHover={{ y: -2, transition: { duration: 0.2 } }}
                                      >
                                          <div className="flex justify-between items-start">
                                              <div>
                                                  <div className="text-lg font-bold">{prediction.symbol}</div>
                                                  <div className="text-xs text-gray-400 mb-2">{prediction.timeframe} outlook</div>
                                                  <div className="text-sm">{prediction.reasoning}</div>
                                              </div>

                                              <div className="flex flex-col items-end">
                                                  <div className={`flex items-center ${prediction.direction === "up" ? "text-green-400" : "text-red-400"}`}>
                                                      {prediction.direction === "up" ? (
                                                          <TrendingUp size={16} className="mr-1" />
                                                      ) : (
                                                          <ArrowDownRight size={16} className="mr-1" />
                                                      )}
                                                      <span className="font-medium">
                                                          {prediction.direction === "up" ? "Bullish" : "Bearish"}
                                                      </span>
                                                  </div>

                                                  <div className="mt-1 flex items-center">
                                                      <div className="w-12 h-12 rounded-full flex items-center justify-center bg-blue-900 bg-opacity-30 border-2 border-blue-500 relative">
                                                          <span className="font-bold text-sm">{prediction.confidence}</span>
                                                          <svg width="48" height="48" viewBox="0 0 48 48" className="absolute top-0 left-0">
                                                              <circle
                                                                  cx="24"
                                                                  cy="24"
                                                                  r="20"
                                                                  fill="none"
                                                                  stroke="#3B82F6"
                                                                  strokeWidth="4"
                                                                  strokeDasharray={`${prediction.confidence * 1.26} 126`}
                                                                  strokeDashoffset="-31.5"
                                                                  transform="rotate(-90, 24, 24)" />
                                                          </svg>
                                                      </div>
                                                  </div>
                                              </div>
                                          </div>
                                      </motion.div>
                                  ))}

                                  <motion.button
                                      className="w-full py-3 mt-2 border border-dashed border-gray-600 rounded-xl text-gray-400 hover:text-blue-400 hover:border-blue-500 transition-colors flex items-center justify-center"
                                      initial={{ opacity: 0 }}
                                      animate={{ opacity: 1 }}
                                      transition={{ delay: 1 }}
                                      whileHover={{ scale: 1.01 }}
                                  >
                                      <Plus size={16} className="mr-2" />
                                      Add Custom Prediction Alert
                                  </motion.button>
                              </div>
                          )}
                      </div>
                  </motion.div>

                  {/* Watchlist */}
                  <motion.div
                      className="col-span-1 lg:col-span-6 bg-gray-800 bg-opacity-60 backdrop-filter backdrop-blur-md rounded-2xl border border-gray-700 overflow-hidden"
                      variants={fadeInUp}
                  >
                      <div className="p-6">
                          <div className="flex justify-between items-center mb-4">
                              <h2 className="text-xl font-bold">Watchlist</h2>
                              <button className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-medium transition-colors flex items-center">
                                  <Plus size={14} className="mr-1" />
                                  Add Symbol
                              </button>
                          </div>

                          {!isLoaded ? (
                              <div className="h-64 flex items-center justify-center">
                                  <div className="w-10 h-10 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
                              </div>
                          ) : (
                              <div className="overflow-x-auto">
                                  <table className="w-full min-w-full">
                                      <thead>
                                          <tr className="text-left text-xs text-gray-400 border-b border-gray-700">
                                              <th className="pb-2">Symbol</th>
                                              <th className="pb-2">Price</th>
                                              <th className="pb-2">Change</th>
                                              <th className="pb-2">AI Score</th>
                                              <th className="pb-2">Prediction</th>
                                          </tr>
                                      </thead>
                                      <tbody>
                                          {filteredWatchlist.length > 0 ? (
                                              filteredWatchlist.map((stock, index) => (
                                                  <motion.tr
                                                      key={stock.symbol}
                                                      className="border-b border-gray-700 last:border-b-0 text-sm hover:bg-gray-700 hover:bg-opacity-40 cursor-pointer"
                                                      initial={{ opacity: 0, y: 10 }}
                                                      animate={{ opacity: 1, y: 0 }}
                                                      transition={{ delay: 0.2 + (index * 0.1) }}
                                                  >
                                                      <td className="py-3">
                                                          <div>
                                                              <div className="font-medium">{stock.symbol}</div>
                                                              <div className="text-xs text-gray-400">{stock.name}</div>
                                                          </div>
                                                      </td>
                                                      <td className="py-3 font-medium">{stock.price}</td>
                                                      <td className={`py-3 flex items-center ${stock.trending === "up" ? "text-green-400" : "text-red-400"}`}>
                                                          {stock.change}
                                                          {stock.trending === "up" ? (
                                                              <ArrowUpRight size={14} className="ml-1" />
                                                          ) : (
                                                              <ArrowDownRight size={14} className="ml-1" />
                                                          )}
                                                      </td>
                                                      <td className="py-3">
                                                          <div className="w-10 h-10 rounded-full flex items-center justify-center bg-blue-900 bg-opacity-20 border border-blue-700">
                                                              <span className="font-medium text-sm">{stock.score}</span>
                                                          </div>
                                                      </td>
                                                      <td className="py-3">
                                                          <span className={`px-2 py-1 rounded-md text-xs font-medium ${stock.prediction === "STRONG BUY"
                                                                  ? "bg-green-900 bg-opacity-50 text-green-300"
                                                                  : stock.prediction === "BUY"
                                                                      ? "bg-green-800 bg-opacity-40 text-green-400"
                                                                      : stock.prediction === "HOLD"
                                                                          ? "bg-yellow-800 bg-opacity-40 text-yellow-300"
                                                                          : "bg-red-900 bg-opacity-40 text-red-400"}`}>
                                                              {stock.prediction}
                                                          </span>
                                                      </td>
                                                  </motion.tr>
                                              ))
                                          ) : (
                                              <tr>
                                                  <td colSpan={5} className="py-8 text-center text-gray-400">
                                                      No results found.
                                                  </td>
                                              </tr>
                                          )}
                                      </tbody>
                                  </table>
                              </div>
                          )}
                      </div>
                  </motion.div>

                  {/* Market Insights */}
                  <motion.div
                      className="col-span-1 lg:col-span-6 bg-gray-800 bg-opacity-60 backdrop-filter backdrop-blur-md rounded-2xl border border-gray-700 overflow-hidden"
                      variants={fadeInUp}
                  >
                      <div className="p-6">
                          <div className="flex justify-between items-center mb-4">
                              <div className="flex items-center">
                                  <TrendingUp size={20} className="text-green-400 mr-2" />
                                  <h2 className="text-xl font-bold">Market Insights</h2>
                              </div>
                              <button className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors">
                                  View All
                              </button>
                          </div>

                          {!isLoaded ? (
                              <div className="h-64 flex items-center justify-center">
                                  <div className="w-10 h-10 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
                              </div>
                          ) : (
                              <div className="space-y-4">
                                  {marketInsights.map((insight, index) => (
                                      <motion.div
                                          key={insight.title}
                                          className="bg-gray-700 bg-opacity-40 rounded-xl p-4 border border-gray-600 hover:border-green-500 transition-colors"
                                          initial={{ opacity: 0, y: 20 }}
                                          animate={{ opacity: 1, y: 0 }}
                                          transition={{ delay: 0.3 + (index * 0.1) }}
                                          whileHover={{ y: -2, transition: { duration: 0.2 } }}
                                      >
                                          <div className="flex justify-between items-start">
                                              <div>
                                                  <div className="text-lg font-bold">{insight.title}</div>
                                                  <div className="text-xs text-gray-400 mb-2">{insight.category} • {insight.timestamp}</div>
                                                  <div className="text-sm">{insight.description}</div>
                                              </div>
                                              <div className="flex items-center">
                                                  <Check size={18} className="text-green-400" />
                                              </div>
                                          </div>
                                      </motion.div>
                                  ))}
                              </div>
                          )}
                      </div>
                  </motion.div>
              </motion.div>
          </div>
      </section><Footer /></>
                              );
                            }