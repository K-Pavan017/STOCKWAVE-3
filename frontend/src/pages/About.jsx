import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Brain, TrendingUp } from 'lucide-react'; // More relevant icons

import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

const stats = [
  { icon: <LineChart size={28} />, label: 'Data Points Analyzed', value: 'Thousands' },
  { icon: <Brain size={28} />, label: 'AI Prediction Accuracy', value: '70%+' }, // Adjusted accuracy for a prediction model
  { icon: <TrendingUp size={28} />, label: 'Strategic Insights', value: 'Daily' },
];

const AboutStockSage = () => { // Renamed component for clarity
  return (
    <div className="bg-gray-950 text-white min-h-screen font-sans">
      <NavBar />

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-800 to-blue-900 opacity-40 blur-3xl z-0" />
        <div className="max-w-7xl mx-auto px-6 py-20 relative z-10 text-center">
          <motion.h1
            initial={{ opacity: 0, y: -40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-4xl md:text-5xl font-bold text-white mb-4"
          >
            About StockWave: Your AI Investment Edge
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="text-gray-300 text-lg max-w-2xl mx-auto"
          >
            Welcome to StockSage, where cutting-edge Artificial Intelligence meets the dynamic world of stock markets. We empower investors with predictive insights, helping you navigate volatility and make informed decisions.
          </motion.p>
        </div>
      </section>

      {/* Key Features/Tech Section */}
      <section className="py-16 bg-gray-900">
        <div className="max-w-6xl mx-auto px-6 grid grid-cols-1 sm:grid-cols-3 gap-8 text-center">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2, duration: 0.6 }}
              className="bg-gray-800 p-6 rounded-2xl shadow-lg border border-gray-700"
            >
              <div className="text-cyan-400 mb-3">{stat.icon}</div>
              <h3 className="text-3xl font-bold text-white">{stat.value}</h3>
              <p className="text-gray-400">{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Our Approach (LSTM Focus) */}
      <section className="py-20 bg-gray-950">
        <div className="max-w-5xl mx-auto px-6">
          <motion.h2
            initial={{ opacity: 0, y: -30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-3xl md:text-4xl font-bold text-center mb-8"
          >
            How We Predict: The Power of LSTM
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.7 }}
            className="text-lg text-gray-300 text-center max-w-3xl mx-auto leading-relaxed"
          >
            At the core of StockSage is a sophisticated **Long Short-Term Memory (LSTM) neural network model**. Unlike traditional methods, LSTMs excel at understanding sequences and remembering patterns over long periods, making them ideal for time-series data like stock prices. Our model processes vast historical data, identifying complex trends and dependencies to generate future price predictions. This advanced AI capability provides you with a unique edge in anticipating market movements.
          </motion.p>
        </div>
      </section>

      {/* Our Mission/Commitment */}
      <section className="py-20 bg-gray-900"> {/* Changed background to slightly differentiate */}
        <div className="max-w-5xl mx-auto px-6">
          <motion.h2
            initial={{ opacity: 0, y: -30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-3xl md:text-4xl font-bold text-center mb-8"
          >
            Our Mission & Commitment
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.7 }}
            className="text-lg text-gray-300 text-center max-w-3xl mx-auto leading-relaxed"
          >
            We are dedicated to providing accessible, reliable, and intelligent stock market insights. StockSage is built on principles of transparency and continuous improvement. We aim to demystify financial data, making powerful predictive analytics available to everyone, from novice traders to seasoned professionals, fostering a community of informed investors.
          </motion.p>
        </div>
      </section>


      {/* Call To Action */}
      <section className="bg-gradient-to-r from-cyan-700 to-blue-800 py-16 px-6 text-center">
        <motion.h3
          initial={{ scale: 0.8, opacity: 0 }}
          whileInView={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="text-3xl md:text-4xl font-semibold text-white mb-4"
        >
          Ready to make smarter investment decisions?
        </motion.h3>
        <p className="text-white text-lg mb-6">
          Experience the power of AI-driven stock prediction with Stock Wave.
        </p>
        <a
          href="/home"
          className="inline-block px-6 py-3 bg-white text-cyan-700 font-semibold rounded-lg hover:bg-gray-100 transition"
        >
          Explore Predictions
        </a>
      </section>

      <Footer />
    </div>
  );
};

export default AboutStockSage; // Export the renamed component