import { useState, useEffect, Profiler } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import {
  User,
  LogOut,
  Settings,
  ChevronDown,
  Sun,
  Moon,
  Home,
  BarChart3,
  Info,
  Phone,
  MoreHorizontal,
  TrendingUp,
} from "lucide-react";

export default function Navbar() {
  const [darkMode, setDarkMode] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const userEmail = user.email || "user@stockwave.com";
  const userInitial = user.username ? user.username[0].toUpperCase() : "U";

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (!darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  // Add scroll effect for navbar
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // Close profile dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      const profileDropdown = document.getElementById("profile-dropdown");
      const profileButton = document.getElementById("profile-button");

      if (
        profileDropdown &&
        profileButton &&
        !profileDropdown.contains(event.target) &&
        !profileButton.contains(event.target)
      ) {
        setProfileOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const themeClass = darkMode
    ? "bg-gray-900 text-white"
    : "bg-white text-gray-800";

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${themeClass} sticky top-0 z-50 transition-all duration-300 ease-in-out ${
        scrolled
          ? darkMode
            ? "shadow-lg shadow-gray-900/20"
            : "shadow-lg shadow-blue-900/10"
          : ""
      }`}
    >
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo Section */}
          <motion.div
            className="flex-shrink-0 flex items-center"
            whileHover={{ scale: 1.03 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <div className="flex items-center">
              <div className="h-9 w-9 rounded-xl flex items-center justify-center bg-gradient-to-r from-blue-600 to-blue-500">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              <motion.span
                className="ml-2 text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-blue-400"
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                StockWave
              </motion.span>
            </div>
          </motion.div>

          {/* Desktop Navigation Links */}
          <div>
            <div className="ml-10 flex items-center space-x-1">
              <NavLink
                icon={<Home size={18} />}
                text="Home"
                to="/"
                active={true}
                darkMode={darkMode}
              />
              <NavLink
                icon={<BarChart3 size={18} />}
                text="Dashboard"
                to="/stock"
                darkMode={darkMode}
              />
              <NavLink
                icon={<Info size={18} />}
                text="About"
                to="/about"
                darkMode={darkMode}
              />
              <NavLink
                icon={<Phone size={18} />}
                text="Contact Us"
                to="/contact"
                darkMode={darkMode}
              />
              <NavLink
                icon={<User size={18} />}
                text="Portfolio"
                to="/dashboard"
                darkMode={darkMode}
              />
            </div>
          </div>

          {/* Right side items */}
          <div className="flex items-center">
            {/* Dark Mode Toggle */}
            <motion.div className="mr-4" whileTap={{ scale: 0.9 }}>
              <button
                onClick={toggleDarkMode}
                className={`p-2 rounded-full transition-colors duration-300 shadow-sm ${
                  darkMode
                    ? "bg-gray-800 hover:bg-gray-700"
                    : "bg-gray-100 hover:bg-gray-200"
                }`}
                aria-label="Toggle dark mode"
              >
                <motion.div
                  initial={false}
                  animate={{ rotate: darkMode ? 180 : 0 }}
                  transition={{ duration: 0.5, type: "spring" }}
                >
                  {darkMode ? (
                    <Sun size={20} className="text-yellow-400" />
                  ) : (
                    <Moon size={20} className="text-gray-600" />
                  )}
                </motion.div>
              </button>
            </motion.div>

            {/* Profile Dropdown */}
            <div className="ml-4 relative flex-shrink-0">
              <div>
                <motion.button
                  id="profile-button"
                  className={`flex text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                    darkMode ? "focus:ring-blue-500" : "focus:ring-blue-600"
                  }`}
                  onClick={() => setProfileOpen(!profileOpen)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="sr-only">Open user menu</span>
                  <div
                    className={`px-3 py-2 rounded-full flex items-center transition-colors duration-200 ${
                      darkMode
                        ? "bg-gray-800 hover:bg-gray-700"
                        : "bg-blue-50 hover:bg-blue-100"
                    }`}
                  >
                    {/* <User
                      size={18}
                      className={darkMode ? "text-blue-400" : "text-blue-600"}
                    /> */}
                    <div
                      className={`h-8 w-8 rounded-full flex items-center justify-center text-lg font-bold ${
                        darkMode
                          ? "bg-blue-600 text-white"
                          : "bg-blue-200 text-blue-700"
                      }`}
                    >
                      {userInitial}
                    </div>
                    <span className="ml-2 font-medium">
                      {user.username || "Profile"}
                    </span>

                    <span className="ml-2 font-medium">Profile</span>
                    <motion.div
                      animate={{ rotate: profileOpen ? 180 : 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <ChevronDown size={16} className="ml-1" />
                    </motion.div>
                  </div>
                </motion.button>
              </div>

              {/* Profile dropdown panel */}
              <AnimatePresence>
                {profileOpen && (
                  <motion.div
                    id="profile-dropdown"
                    className={`origin-top-right absolute right-0 mt-2 w-56 rounded-xl shadow-lg backdrop-blur-sm ${
                      darkMode
                        ? "bg-gray-800/95 ring-1 ring-gray-700"
                        : "bg-white/95 ring-1 ring-gray-200"
                    } divide-y ${
                      darkMode ? "divide-gray-700" : "divide-gray-100"
                    } focus:outline-none z-10`}
                    initial={{ opacity: 0, y: -20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -20, scale: 0.95 }}
                    transition={{ duration: 0.2, ease: "easeOut" }}
                  >
                    <div className="px-4 py-3">
                      <p className="text-sm">Signed in as</p>
                      <p
                        className={`text-sm font-bold truncate ${
                          darkMode ? "text-white" : "text-gray-900"
                        }`}
                      >
                        {userEmail}
                      </p>
                    </div>
                    <div className="py-1">
                      <ProfileMenuItem
                        icon={<User size={16} />}
                        text="Your Profile"
                        to="/profile"
                        darkMode={darkMode}
                      />
                      <ProfileMenuItem
                        icon={<Settings size={16} />}
                        text="Settings"
                        to="/settings"
                        darkMode={darkMode}
                      />
                      <ProfileMenuItem
                        icon={<LogOut size={16} />}
                        text="Sign out"
                        to="/login"
                        darkMode={darkMode}
                        isSignOut
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </nav>
    </motion.div>
  );
}

// Desktop Navigation Link component
function NavLink({ icon, text, to, active = false, darkMode }) {
  return (
    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
      <Link
        to={to}
        className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 transform group ${
          active
            ? darkMode
              ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white"
              : "bg-gradient-to-r from-blue-50 to-blue-100 text-blue-700"
            : darkMode
            ? "text-gray-300 hover:bg-gray-800 hover:text-white"
            : "text-gray-700 hover:bg-gray-100 hover:text-blue-600"
        }`}
      >
        <motion.span
          className={`mr-1.5 ${active ? "text-current" : ""}`}
          whileHover={{ rotate: [0, -10, 10, -10, 0] }}
          transition={{ duration: 0.5 }}
        >
          {icon}
        </motion.span>
        {text}
      </Link>
    </motion.div>
  );
}

// Profile dropdown menu item component
function ProfileMenuItem({ icon, text, to, isSignOut = false, darkMode }) {
  return (
    <motion.div whileHover={{ x: 4, transition: { duration: 0.2 } }}>
      <Link
        to={to}
        className={`flex items-center px-4 py-2 text-sm transition-colors duration-200 ${
          isSignOut
            ? darkMode
              ? "text-red-400 hover:bg-gray-700"
              : "text-red-600 hover:bg-gray-100"
            : darkMode
            ? "text-gray-300 hover:bg-gray-700 hover:text-white"
            : "text-gray-700 hover:bg-gray-100 hover:text-blue-700"
        }`}
      >
        <motion.span
          className={`mr-2 ${
            isSignOut ? (darkMode ? "text-red-400" : "text-red-600") : ""
          }`}
          whileHover={{ scale: 1.2, rotate: isSignOut ? 15 : 0 }}
          transition={{ duration: 0.2 }}
        >
          {icon}
        </motion.span>
        {text}
      </Link>
    </motion.div>
  );
}
