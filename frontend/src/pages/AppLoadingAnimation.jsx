import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const AppLoadingAnimation = () => {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Simulate loading progress
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + Math.random() * 15;
        return newProgress >= 100 ? 100 : newProgress;
      });
    }, 300);

    // Navigate to login after animation completes
    const timeout = setTimeout(() => {
      navigate('/login');
    }, 4000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [navigate]);

  // Background wave animation variants
  const waveVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 0.7, 
      scale: 1,
      transition: { 
        duration: 1.5,
        ease: "easeOut"
      }
    }
  };

  // Logo animation variants
  const logoVariants = {
    hidden: { opacity: 0, y: -50 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        delay: 0.5,
        duration: 0.8,
        ease: "easeOut"
      }
    }
  };

  return (
    <div className="h-screen w-full flex flex-col justify-center items-center relative overflow-hidden bg-gray-950">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-900 to-gray-900 opacity-80" />
      
      {/* Dynamic waves animation */}
      <motion.div 
        className="absolute bottom-0 left-0 w-full"
        initial="hidden"
        animate="visible"
        variants={waveVariants}
      >
        <svg viewBox="0 0 1440 320" className="w-full">
          <path 
            fill="rgba(6, 182, 212, 0.3)" 
            d="M0,256L48,240C96,224,192,192,288,197.3C384,203,480,245,576,245.3C672,245,768,203,864,186.7C960,171,1056,181,1152,186.7C1248,192,1344,192,1392,192L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          ></path>
        </svg>
      </motion.div>
      
      <motion.div 
        className="absolute bottom-0 left-0 w-full"
        initial="hidden"
        animate="visible"
        variants={{...waveVariants, visible: {...waveVariants.visible, transition: {delay: 0.2, duration: 1.5}}}}
      >
        <svg viewBox="0 0 1440 320" className="w-full">
          <path 
            fill="rgba(6, 182, 212, 0.2)" 
            d="M0,192L40,181.3C80,171,160,149,240,149.3C320,149,400,171,480,192C560,213,640,235,720,229.3C800,224,880,192,960,181.3C1040,171,1120,181,1200,197.3C1280,213,1360,235,1400,245.3L1440,256L1440,320L1400,320C1360,320,1280,320,1200,320C1120,320,1040,320,960,320C880,320,800,320,720,320C640,320,560,320,480,320C400,320,320,320,240,320C160,320,80,320,40,320L0,320Z"
          ></path>
        </svg>
      </motion.div>
      
      
      {/* Logo */}
      <motion.div 
        className="relative z-10 mb-12"
        variants={logoVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="flex items-center justify-center">
          <span className="text-6xl font-bold text-white">Stock</span>
          <span className="text-6xl font-bold text-cyan-400">Wave</span>
        </div>
        <p className="text-gray-300 text-center mt-2 text-lg">
          AI-Powered Stock Predictions
        </p>
      </motion.div>
      
      {/* Loading bar */}
      <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden relative z-10 mt-8">
        <motion.div 
          className="h-full bg-gradient-to-r from-cyan-400 to-blue-500"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ ease: "easeInOut" }}
        />
      </div>
      
      <p className="text-gray-400 mt-4 relative z-10">
        {progress < 100 ? 'Loading...' : 'Ready!'}
      </p>
    </div>
  );
};

export default AppLoadingAnimation;