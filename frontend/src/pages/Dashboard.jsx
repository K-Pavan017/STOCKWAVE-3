import { useState, useEffect } from "react";
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';
import { 
  Zap, 
  RefreshCw,
  Sun, // Import Sun for light mode
  Moon, // Import Moon for dark mode
  Heart // Used for wishlist icon
} from "lucide-react";
import { motion } from "framer-motion";
import axios from "axios"; 

const backendUrl =import.meta.env.VITE_APP_BACKEND_URL; 
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
  const [isDarkMode, setIsDarkMode] = useState(true); // Default to dark mode
  const [watchlistTickers, setWatchlistTickers] = useState([]);
  const [wishlistTickers, setWishlistTickers] = useState([]);
  const [watchlistData, setWatchlistData] = useState([]);
  const [wishlistData, setWishlistData] = useState([]);
  const [loadingWatchlist, setLoadingWatchlist] = useState(false);
  const [loadingWishlist, setLoadingWishlist] = useState(false);
  

  // Load theme preference from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
      setIsDarkMode(false);
      document.documentElement.classList.remove('dark');
    } else {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    }
  }, []);

  // Effect to apply theme class
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  // Toggle theme
  const toggleTheme = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  // Load watchlist and wishlist tickers from localStorage
  useEffect(() => {
    try {
      const storedWatchlist = JSON.parse(localStorage.getItem('userWatchlist')) || [];
      setWatchlistTickers(storedWatchlist);

      const storedWishlist = JSON.parse(localStorage.getItem('userWishlist')) || [];
      setWishlistTickers(storedWishlist);
    } catch (e) {
      // Optionally handle error in UI
    }
  }, []);

  // Live fetching for Dashboard removed as requested to avoid rate limiting. 
  // Watchlist and Wishlist will show symbols without real-time data.


  // Helper to remove from wishlist directly from dashboard
  const removeFromWishlist = (tickerToRemove) => {
    try {
      let wishlist = JSON.parse(localStorage.getItem('userWishlist')) || [];
      wishlist = wishlist.filter(s => s !== tickerToRemove);
      localStorage.setItem('userWishlist', JSON.stringify(wishlist));
      setWishlistTickers(wishlist); // Update state to trigger re-render
    } catch (e) {
      // Optionally handle error in UI
    }
  };

  // Manual refresh logic removed


  return (
    <>
      <NavBar />
      <section className={`min-h-screen ${isDarkMode ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-gray-100' : 'bg-gradient-to-br from-blue-50 via-blue-100 to-blue-50 text-gray-800'}`}>
          <div className="container mx-auto px-4 py-8">
              <motion.div 
                  className="flex justify-between items-center mb-8"
                  variants={fadeInUp}
                  initial="hidden"
                  animate="visible"
              >
                  <h1 className={`${isDarkMode ? 'text-white' : 'text-gray-900'} text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500`}>
                      Your Stock Hub
                  </h1>
                  <button
                      onClick={toggleTheme}
                      className={`p-2 rounded-full shadow-lg transition-all duration-300
                          ${isDarkMode ? 'bg-gray-700 text-yellow-300 hover:bg-gray-600' : 'bg-white text-blue-600 hover:bg-gray-100'}`}
                  >
                      {isDarkMode ? <Sun size={24} /> : <Moon size={24} />}
                  </button>
              </motion.div>

              <motion.div 
                  className="grid grid-cols-1 md:grid-cols-2 gap-8"
                  variants={staggerChildren}
                  initial="hidden"
                  animate="visible"
              >
                  {/* Watchlist Section */}
                  <motion.div 
                      className={`${isDarkMode ? 'bg-gray-800/50 border-gray-700/50' : 'bg-white/70 border-blue-200/50'} backdrop-blur-lg rounded-2xl p-6 shadow-2xl border`}
                      variants={fadeInUp}
                  >
                      <div className="flex justify-between items-center mb-6">
                          <h2 className={`${isDarkMode ? 'text-white' : 'text-gray-900'} text-2xl font-bold`}>
                              <Zap size={24} className="inline-block mr-2 text-blue-400" /> Watchlist
                          </h2>
                      </div>
                      {watchlistTickers.length === 0 ? (
                          <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'} text-center py-8`}>
                              Your watchlist is empty. Search for stocks in StockData to add them here!
                          </div>
                      ) : (
                          <div className="space-y-4">
                              {watchlistTickers.map((ticker, index) => (
                                  <motion.div
                                      key={ticker}
                                      className={`${isDarkMode ? 'bg-gray-700/30 hover:bg-gray-600/40' : 'bg-blue-100/50 hover:bg-blue-200/60'} rounded-xl p-4 transition-all duration-220 cursor-pointer border ${isDarkMode ? 'border-gray-600/50' : 'border-blue-200/50'}`}
                                      initial={{ opacity: 0, x: -20 }}
                                      animate={{ opacity: 1, x: 0 }}
                                      transition={{ delay: index * 0.05 }}
                                      whileHover={{ scale: 1.01 }}
                                  >
                                      <div className="flex justify-between items-center">
                                          <div>
                                              <div className={`${isDarkMode ? 'text-white' : 'text-gray-900'} text-lg font-bold`}>{ticker}</div>
                                              <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'} text-sm`}>Live data disabled to avoid rate limits</div>
                                          </div>
                                      </div>
                                  </motion.div>
                              ))}
                          </div>
                      )}
                  </motion.div>

                  {/* Wishlist Section */}
                  <motion.div 
                      className={`${isDarkMode ? 'bg-gray-800/50 border-gray-700/50' : 'bg-white/70 border-blue-200/50'} backdrop-blur-lg rounded-2xl p-6 shadow-2xl border`}
                      variants={fadeInUp}
                  >
                      <div className="flex justify-between items-center mb-6">
                          <h2 className={`${isDarkMode ? 'text-white' : 'text-gray-900'} text-2xl font-bold`}>
                              <Heart size={24} className="inline-block mr-2 text-red-400" fill="currentColor" /> Wishlist
                          </h2>
                      </div>
                      {wishlistTickers.length === 0 ? (
                          <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'} text-center py-8`}>
                              Your wishlist is empty. Like stocks in StockData to add them here!
                          </div>
                      ) : (
                          <div className="space-y-4">
                              {wishlistTickers.map((ticker, index) => (
                                  <motion.div
                                      key={ticker}
                                      className={`${isDarkMode ? 'bg-gray-700/30 hover:bg-gray-600/40' : 'bg-blue-100/50 hover:bg-blue-200/60'} rounded-xl p-4 transition-all duration-220 cursor-pointer border ${isDarkMode ? 'border-gray-600/50' : 'border-blue-200/50'} flex items-center justify-between`}
                                      initial={{ opacity: 0, x: -20 }}
                                      animate={{ opacity: 1, x: 0 }}
                                      transition={{ delay: index * 0.05 }}
                                      whileHover={{ scale: 1.01 }}
                                  >
                                      <div>
                                          <div className={`${isDarkMode ? 'text-white' : 'text-gray-900'} text-lg font-bold`}>{ticker}</div>
                                          <div className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'} text-sm`}>Live data disabled</div>
                                      </div>
                                      <div className="text-right flex items-center">
                                          <button 
                                              onClick={(e) => { e.stopPropagation(); removeFromWishlist(ticker); }}
                                              className={`${isDarkMode ? 'text-red-440 hover:text-red-300' : 'text-red-500 hover:text-red-700'} p-1 rounded-full transition-colors duration-200`}
                                              title="Remove from Wishlist"
                                          >
                                              <Heart size={20} fill="currentColor" />
                                          </button>
                                      </div>
                                  </motion.div>
                              ))}
                          </div>
                      )}
                  </motion.div>
              </motion.div>
          </div>
      </section>
      <Footer />
    </>
  );
}
