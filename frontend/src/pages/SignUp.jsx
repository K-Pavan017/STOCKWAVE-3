import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Eye, EyeOff, UserPlus, LogIn, AlertCircle } from 'lucide-react';
import axios from 'axios';

const SignUp = () => {
  const [showPassword, setShowPassword] = useState(false);
  // const [values, setValues] = useState({ email: '', password: '', confirmPassword: '' });
  const [values, setValues] = useState({ username: '', email: '', password: '', confirmPassword: '' });

  const [errors, setErrors] = useState({});
  const [backendError, setBackendError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Animation states
  const [showElements, setShowElements] = useState({
    waves: false,
    logo: false,
    form: false
  });

  useEffect(() => {
    const wavesTimer = setTimeout(() => setShowElements(prev => ({ ...prev, waves: true })), 300);
    const logoTimer = setTimeout(() => setShowElements(prev => ({ ...prev, logo: true })), 800);
    const formTimer = setTimeout(() => setShowElements(prev => ({ ...prev, form: true })), 1200);

    return () => {
      clearTimeout(wavesTimer);
      clearTimeout(logoTimer);
      clearTimeout(formTimer);
    };
  }, []);

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const handleInput = (event) => {
    setValues(prev => ({ ...prev, [event.target.name]: event.target.value }));
    if (errors[event.target.name]) {
      setErrors(prev => ({ ...prev, [event.target.name]: '' }));
    }
    setBackendError('');
  };

  const validateForm = () => {
    let isValid = true;
    const newErrors = {};

    // Username validation
    if (!values.username) {
      newErrors.username = 'Username is required';
      isValid = false;
    } else if (values.username.length < 3) {
      newErrors.username = 'Username must be at least 3 characters';
      isValid = false;
    }

    // Email validation
    if (!values.email) {
      newErrors.email = 'Email is required';
      isValid = false;
    } else if (!/\S+@\S+\.\S+/.test(values.email)) {
      newErrors.email = 'Email address is invalid';
      isValid = false;
    }

    // Password validation
    if (!values.password) {
      newErrors.password = 'Password is required';
      isValid = false;
    } else if (values.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
      isValid = false;
    }

    // Confirm password validation
    if (!values.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
      isValid = false;
    } else if (values.password !== values.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (validateForm()) {
      setIsLoading(true);
      try {
        const res = await axios.post('http://127.0.0.1:5000/signup', {
  username: values.username,
  email: values.email,
  password: values.password
});

        if (res.data.success) {
          navigate('/login');
        } else {
          setBackendError(res.data.message || 'Signup failed');
        }
      } catch (err) {
        console.error('Signup request error:', err);
        setBackendError('Server error. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen w-full flex justify-center items-center relative overflow-hidden bg-gray-950">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-900 to-gray-900 opacity-80" />

      {/* Background image */}
      <div
        className="absolute inset-0 bg-cover bg-center opacity-20"
        style={{ backgroundImage: 'url("/stock.png")' }}
      />

      
      {/* Logo */}
      <motion.div
        className="absolute top-8 md:top-6 left 2 transform -translate-x-1/2"
        initial={{ opacity: 0, y: -30 }}
        animate={showElements.logo ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8 }}
      >
        <div className="flex items-center justify-center">
          <span className="text-4xl md:text-5xl font-bold text-white">Stock</span>
          <span className="text-4xl md:text-5xl font-bold text-cyan-400">Wave</span>
        </div>
      </motion.div>

      {/* Sign Up Form Card */}
      <motion.div
        className="relative z-10 w-11/12 max-w-md mx-auto mt-20"
        initial={{ opacity: 0, y: 30 }}
        animate={showElements.form ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.7 }}
      >
        <div className="bg-gray-800 bg-opacity-70 backdrop-filter backdrop-blur-lg rounded-xl shadow-2xl overflow-hidden border border-gray-700">
          {/* Card Header */}
          <div className="px-8 pt-8 pb-4">
            <h2 className="text-2xl font-bold text-white mb-1">Create Account</h2>
            <p className="text-gray-400">Sign up to get started</p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="px-8 pb-8">
            {/* Backend Error Alert */}
            {backendError && (
              <div className="mb-4 p-3 bg-red-500 bg-opacity-20 border border-red-500 rounded-lg flex items-center">
                <AlertCircle size={18} className="text-red-500 mr-2" />
                <p className="text-red-200 text-sm">{backendError}</p>
              </div>
            )}

            {/* Username Field */}
            <div className="mb-5">
  <label htmlFor="username" className="block text-gray-300 text-sm font-medium mb-2">
    Username
  </label>
  <input
    type="text"
    id="username"
    name="username"
    value={values.username}
    onChange={handleInput}
    className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
      errors.username ? 'border-red-500' : 'border-gray-600'
    } rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition duration-200`}
    placeholder="your_username"
  />
  {errors.username && <p className="mt-1 text-sm text-red-400">{errors.username}</p>}
</div>

            {/* Email Field */}
            <div className="mb-5">
              <label htmlFor="email" className="block text-gray-300 text-sm font-medium mb-2">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={values.email}
                onChange={handleInput}
                className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                  errors.email ? 'border-red-500' : 'border-gray-600'
                } rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition duration-200`}
                placeholder="name@example.com"
              />
              {errors.email && <p className="mt-1 text-sm text-red-400">{errors.email}</p>}
            </div>

            {/* Password Field */}
            <div className="mb-5">
              <label htmlFor="password" className="block text-gray-300 text-sm font-medium mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  id="password"
                  name="password"
                  value={values.password}
                  onChange={handleInput}
                  className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                    errors.password ? 'border-red-500' : 'border-gray-600'
                  } rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition duration-200`}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white focus:outline-none"
                  onClick={togglePasswordVisibility}
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
              {errors.password && <p className="mt-1 text-sm text-red-400">{errors.password}</p>}
            </div>

            {/* Confirm Password Field */}
            <div className="mb-6">
              <label htmlFor="confirmPassword" className="block text-gray-300 text-sm font-medium mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  id="confirmPassword"
                  name="confirmPassword"
                  value={values.confirmPassword}
                  onChange={handleInput}
                  className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                    errors.confirmPassword ? 'border-red-500' : 'border-gray-600'
                  } rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition duration-200`}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white focus:outline-none"
                  onClick={togglePasswordVisibility}
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
              {errors.confirmPassword && <p className="mt-1 text-sm text-red-400">{errors.confirmPassword}</p>}
            </div>

            {/* Sign Up Button */}
            <button
              type="submit"
              disabled={isLoading}
              className={`w-full flex items-center justify-center px-4 py-3 rounded-lg font-medium transition duration-200 ${
                isLoading
                  ? 'bg-gray-600 text-gray-300 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white'
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating...
                </>
              ) : (
                <>
                  <UserPlus size={18} className="mr-2" />
                  Sign Up
                </>
              )}
            </button>

            {/* Terms and Login Link */}
            <div className="mt-8 text-sm text-center">
              <p className="text-gray-400">
                By signing up, you agree to our
                <a href="#" className="text-cyan-400 hover:text-cyan-300 ml-1">Terms of Service</a>
                <span className="mx-1">and</span>
                <a href="#" className="text-cyan-400 hover:text-cyan-300">Privacy Policy</a>
              </p>
              <div className="mt-4 flex items-center justify-center">
                <span className="text-gray-400">Already have an account?</span>
                <Link
                  to="/login"
                  className="ml-2 text-cyan-400 hover:text-cyan-300 flex items-center transition duration-200"
                >
                  Sign in
                  <LogIn size={16} className="ml-1" />
                </Link>
              </div>
            </div>
          </form>
        </div>
      </motion.div>
    </div>
  );
};

export default SignUp;