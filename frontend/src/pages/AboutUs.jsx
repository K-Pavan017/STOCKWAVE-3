import React from 'react';
import { motion } from 'framer-motion';
import { Users, Star, Globe } from 'lucide-react';

import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

const stats = [
  { icon: <Users size={28} />, label: 'Happy Users', value: '200K+' },
  { icon: <Star size={28} />, label: 'Accuracy Rate', value: '92%' },
  { icon: <Globe size={28} />, label: 'Global Reach', value: '120+ Countries' },
];

const AboutUs = () => {
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
            Who We Are
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="text-gray-300 text-lg max-w-2xl mx-auto"
          >
            At StockSage, we're driven by the mission to democratize access to powerful AI-driven stock prediction tools for everyone—from casual investors to financial professionals.
          </motion.p>
        </div>
      </section>

      {/* Stats Section */}
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

      {/* Vision and Team */}
      <section className="py-20 bg-gray-950">
        <div className="max-w-5xl mx-auto px-6">
          <motion.h2
            initial={{ opacity: 0, y: -30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-3xl md:text-4xl font-bold text-center mb-8"
          >
            Our Vision
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.7 }}
            className="text-lg text-gray-300 text-center max-w-3xl mx-auto"
          >
            We believe in making complex data simple. Our AI models scan and learn from millions of market data points so you don't have to. We're committed to transparency, education, and empowering our users to make confident trading decisions.
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
          Ready to level up your investment strategy?
        </motion.h3>
        <p className="text-white text-lg mb-6">
          Join thousands of users who trust StockSage to power their financial decisions.
        </p>
        <a
          href="/signup"
          className="inline-block px-6 py-3 bg-white text-cyan-700 font-semibold rounded-lg hover:bg-gray-100 transition"
        >
          Get Started
        </a>
      </section>

      <Footer />
    </div>
  );
};

export default AboutUs;
