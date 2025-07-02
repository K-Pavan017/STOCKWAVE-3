import { FaChartLine, FaBrain, FaDesktop } from "react-icons/fa";
import { motion } from "framer-motion";

export default function HowItWorks() {
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { y: 40, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 12
      }
    }
  };
  
  const titleVariants = {
    hidden: { y: -20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { 
        type: "spring", 
        stiffness: 100,
        duration: 0.8 
      }
    }
  };

  return (
    <section className="bg-gradient-to-b from-gray-900 to-gray-800 text-white py-24 px-6 lg:px-20 relative overflow-hidden">
      {/* Background accents */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/3 -left-24 w-64 h-64 bg-blue-600/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 -right-24 w-80 h-80 bg-indigo-600/10 rounded-full blur-3xl"></div>
      </div>

      <div className="max-w-7xl mx-auto text-center relative z-10">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.3 }}
          variants={titleVariants}
          className="mb-16"
        >
          <h2 className="inline-block text-4xl lg:text-5xl font-bold mb-6 bg-gradient-to-r from-blue-300 via-blue-100 to-white text-transparent bg-clip-text">
            How StockWave Predicts Tomorrow's Market
          </h2>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto">
            Our platform combines real-time data, AI algorithms, and an intuitive dashboard to give you market foresight.
          </p>
        </motion.div>

        {/* Feature Grid */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
        >
          {/* Feature 1 */}
          <motion.div 
            variants={itemVariants}
            whileHover={{ y: -8, transition: { duration: 0.3 } }}
            className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl shadow-xl p-8 hover:shadow-2xl transition group"
          >
            <div className="text-blue-500 text-4xl mb-6 group-hover:scale-110 transition duration-300 flex justify-center">
              <div className="bg-blue-900/30 p-4 rounded-xl border border-blue-800/50">
                <FaChartLine />
              </div>
            </div>
            <h3 className="text-xl font-semibold mb-4">Real-Time Market Data</h3>
            <p className="text-gray-300">
              We ingest live data streams from top exchanges to ensure up-to-date predictions and market responsiveness.
            </p>
          </motion.div>

          {/* Feature 2 */}
          <motion.div 
            variants={itemVariants}
            whileHover={{ y: -8, transition: { duration: 0.3 } }}
            className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl shadow-xl p-8 hover:shadow-2xl transition group"
          >
            <div className="text-indigo-500 text-4xl mb-6 group-hover:scale-110 transition duration-300 flex justify-center">
              <div className="bg-indigo-900/30 p-4 rounded-xl border border-indigo-800/50">
                <FaBrain />
              </div>
            </div>
            <h3 className="text-xl font-semibold mb-4">AI & LSTM Algorithms</h3>
            <p className="text-gray-300">
              Our models use deep learning (LSTM) to detect long-term patterns and predict future stock movements.
            </p>
          </motion.div>

          {/* Feature 3 */}
          <motion.div 
            variants={itemVariants}
            whileHover={{ y: -8, transition: { duration: 0.3 } }}
            className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl shadow-xl p-8 hover:shadow-2xl transition group"
          >
            <div className="text-violet-500 text-4xl mb-6 group-hover:scale-110 transition duration-300 flex justify-center">
              <div className="bg-violet-900/30 p-4 rounded-xl border border-violet-800/50">
                <FaDesktop />
              </div>
            </div>
            <h3 className="text-xl font-semibold mb-4">Intuitive Dashboard</h3>
            <p className="text-gray-300">
              Visualize predictions with interactive charts, historical data overlays, and easy navigation.
            </p>
          </motion.div>
        </motion.div>

        {/* Additional Feature - Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="mt-16"
        >
          <a 
            href="#" 
            className="inline-flex items-center bg-gradient-to-r from-blue-600 to-indigo-600 px-8 py-4 rounded-xl font-semibold shadow-lg shadow-blue-900/20 hover:shadow-blue-900/40 transition-all duration-300 transform hover:-translate-y-1 group"
          >
            <span>See how it works</span>
            <span className="ml-2 group-hover:ml-4 transition-all duration-300">→</span>
          </a>
        </motion.div>
      </div>
    </section>
  );
}