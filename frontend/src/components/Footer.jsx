import React from 'react';
import { motion } from 'framer-motion';
import {
  Mail,
  Phone,
  Facebook,
  Twitter,
  Linkedin,
  Instagram,
  TrendingUp,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';

export default function Footer() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 },
  };

  const currentYear = new Date().getFullYear();

  return (
    <footer className="relative bg-gradient-to-br from-gray-900 via-gray-900 to-gray-950 text-white px-6 py-10 overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-48 h-48 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-5 animate-blob"></div>
        <div className="absolute bottom-1/3 right-1/4 w-64 h-64 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-5 animate-blob animation-delay-2000"></div>
      </div>

      <motion.div
        className="max-w-7xl mx-auto"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.1 }}
      >
        {/* Top section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-10">
          <motion.div variants={itemVariants} className="lg:col-span-2">
            <div className="flex items-center mb-4">
              <div className="h-9 w-9 rounded-xl flex items-center justify-center bg-gradient-to-r from-blue-600 to-blue-500">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              <span className="ml-3 text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-blue-200">
                StockWave
              </span>
            </div>
            <p className="text-gray-400 mb-5 text-sm max-w-lg leading-relaxed">
              AI-powered stock predictions made simple and insightful.
            </p>
            <div className="flex gap-4">
              {[Facebook, Twitter, Linkedin, Instagram].map((Icon, i) => (
                <motion.a
                  href="#"
                  key={i}
                  className="bg-gray-800/50 p-2 rounded-full hover:bg-blue-600/20 transition-colors"
                  whileHover={{ y: -5, scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Icon size={18} className="text-blue-400" />
                </motion.a>
              ))}
            </div>
          </motion.div>

          <motion.div variants={itemVariants} className="lg:col-span-1">
            <h3 className="text-lg font-bold mb-3">Stay Updated</h3>
            <p className="text-gray-400 mb-4 text-sm">
              Get updates on market trends and predictions.
            </p>
            <div className="relative">
              <input
                type="email"
                placeholder="Your email"
                className="w-full px-4 py-2 pr-14 text-sm text-gray-200 bg-gray-800/50 border border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
              />
              <a href="#"><motion.button
                className="absolute right-2 top-1.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 p-2 rounded-lg"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ChevronRight size={16} />
              </motion.button></a>
            </div>
          </motion.div>
        </div>

        {/* Link sections */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-sm">
          {/* Quick Links */}
          <motion.div variants={itemVariants} className="space-y-4">
            <h3 className="font-bold pb-1 relative text-white before:absolute before:bottom-0 before:left-0 before:h-1 before:w-6 before:rounded-full before:bg-blue-500">
              Quick Links
            </h3>
            <ul className="space-y-2">
              {["Home", "Predictions", "About Us", "Pricing", "Blog"].map((text, i) => (
                <motion.li key={i} whileHover={{ x: 5 }}>
                  <a href="#" className="text-gray-400 hover:text-white flex items-center">
                    <ChevronRight size={12} className="mr-2 text-blue-400" />
                    {text}
                  </a>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Resources */}
          <motion.div variants={itemVariants} className="space-y-4">
            <h3 className="font-bold pb-1 relative text-white before:absolute before:bottom-0 before:left-0 before:h-1 before:w-6 before:rounded-full before:bg-blue-500">
              Resources
            </h3>
            <ul className="space-y-2">
              {["Documentation", "API Access", "FAQs & Help", "Support Center"].map((text, i) => (
                <motion.li key={i} whileHover={{ x: 5 }}>
                  <a href="#" className="text-gray-400 hover:text-white flex items-center">
                    <ChevronRight size={12} className="mr-2 text-blue-400" />
                    {text}
                  </a>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Learn Trading */}
          <motion.div variants={itemVariants} className="space-y-4">
            <h3 className="font-bold pb-1 relative text-white before:absolute before:bottom-0 before:left-0 before:h-1 before:w-6 before:rounded-full before:bg-blue-500">
              Learn Trading
            </h3>
            <ul className="space-y-2">
              {["Beginner's Guide", "Market Analysis", "Trading Strategies", "Webinars"].map((text, i) => (
                <motion.li key={i} whileHover={{ x: 5 }}>
                  <a href="#" className="text-gray-400 hover:text-white flex items-center">
                    <ChevronRight size={12} className="mr-2 text-blue-400" />
                    {text}
                  </a>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Contact */}
          <motion.div variants={itemVariants} className="space-y-4">
            <h3 className="font-bold pb-1 relative text-white before:absolute before:bottom-0 before:left-0 before:h-1 before:w-6 before:rounded-full before:bg-blue-500">
              Contact Us
            </h3>
            <ul className="space-y-3">
              <li className="flex items-center text-gray-400">
                <div className="w-7 h-7 bg-blue-600/20 rounded-full flex items-center justify-center mr-3">
                  <Mail size={13} className="text-blue-400" />
                </div>
                <span>pavanppvss.com</span>
              </li>
              <li className="flex items-center text-gray-400">
                <div className="w-7 h-7 bg-blue-600/20 rounded-full flex items-center justify-center mr-3">
                  <Phone size={13} className="text-blue-400" />
                </div>
                <span>+91 7993242204</span>
              </li>
              <li>
                <motion.a
                  href="https://maps.app.goo.gl/WXYa3BoUPNi8k5QX8"
                  className="inline-flex items-center text-blue-400 hover:text-blue-300 font-medium mt-1"
                  whileHover={{ x: 5 }}
                >
                  View our location
                  <ExternalLink size={13} className="ml-2" />
                </motion.a>
              </li>
            </ul>
          </motion.div>
        </div>

        {/* Bottom section */}
        <motion.div
          variants={itemVariants}
          className="border-t border-gray-800 mt-10 pt-6 text-center text-sm"
        >
          <p className="text-gray-500">
            &copy; {currentYear} StockWave. All rights reserved.
          </p>
          <p className="mt-1 text-gray-600 max-w-xl mx-auto">
            Disclaimer: This site provides AI-based predictions. Use at your own risk.
          </p>
        </motion.div>
      </motion.div>

      {/* Animation CSS */}
      <style jsx>{`
        @keyframes blob {
          0% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
          100% {
            transform: translate(0px, 0px) scale(1);
          }
        }
        .animate-blob {
          animation: blob 30s infinite alternate;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
      `}</style>
    </footer>
  );
}
