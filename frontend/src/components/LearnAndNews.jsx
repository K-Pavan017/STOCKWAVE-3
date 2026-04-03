import { useState, useEffect } from 'react';
import { ExternalLink, BookOpen, Youtube, Instagram, Newspaper, TrendingUp, ChevronRight } from 'lucide-react';

export default function LearnAndNews() {
  const [isVisible, setIsVisible] = useState({
    heading: false,
    tutorials: false,
    news: false,
  });
  
  // Animation timing for staggered effects
  useEffect(() => {
    const timer1 = setTimeout(() => setIsVisible(prev => ({ ...prev, heading: true })), 300);
    const timer2 = setTimeout(() => setIsVisible(prev => ({ ...prev, tutorials: true })), 600);
    const timer3 = setTimeout(() => setIsVisible(prev => ({ ...prev, news: true })), 900);
    
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  return (
    <section className="relative bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white py-20 px-6 lg:px-20 overflow-hidden">
      {/* Background decorative elements - matching Hero component style */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 right-1/4 w-64 h-64 bg-indigo-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob"></div>
        <div className="absolute bottom-1/3 left-1/4 w-80 h-80 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000"></div>
        <div className="absolute top-1/2 right-1/3 w-72 h-72 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-4000"></div>
      </div>

      <div className="max-w-7xl mx-auto space-y-20 relative z-10">
        {/* Heading with animation */}
        <div className={`text-center transition-all duration-1000 transform ${isVisible.heading ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
          <div className="inline-block px-3 py-1 mb-6 text-sm font-semibold text-blue-300 bg-blue-900 bg-opacity-30 rounded-full">
            Exclusive Educational Resources
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white via-blue-100 to-blue-200 text-transparent bg-clip-text">
            Expand Your Trading Knowledge
          </h1>
          
          <p className="text-lg text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Premium educational resources and market insights curated for serious traders
          </p>
        </div>

        {/* Tutorials Section */}
        <div className={`transition-all duration-1000 delay-300 transform ${isVisible.tutorials ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
          <div className="flex items-center justify-center gap-3 mb-10">
            <div className="h-px w-10 bg-gradient-to-r from-transparent to-blue-500 opacity-70"></div>
            <h2 className="text-2xl font-bold text-white">Learn Stock Market & Trading</h2>
            <div className="h-px w-10 bg-gradient-to-l from-transparent to-blue-500 opacity-70"></div>
          </div>
          
          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
            {[
              {
                title: "Stock Market for Beginners",
                description: "Master the fundamentals of stock markets and start your investing journey",
                link: "https://www.youtube.com/watch?v=p7HKvqRI_Bo",
                thumbnail: "/api/placeholder/640/360",
                icon: <Youtube size={24} className="text-red-500" />,
                color: "from-red-600 to-red-400"
              },
              {
                title: "How to Trade Stocks",
                description: "Learn effective trading strategies from industry professionals",
                link: "https://www.youtube.com/watch?v=4CQzOXbkLqY",
                thumbnail: "/api/placeholder/640/360",
                icon: <Youtube size={24} className="text-red-500" />,
                color: "from-red-600 to-red-400"
              },
              {
                title: "Technical Analysis Basics",
                description: "Understand charts, indicators and patterns for better trading decisions",
                link: "https://www.youtube.com/watch?v=eynxyoKgpng",
                thumbnail: "/api/placeholder/640/360",
                icon: <Youtube size={24} className="text-red-500" />,
                color: "from-red-600 to-red-400"
              },
            ].map((video, idx) => (
              <div
                key={idx}
                className="group h-full bg-gray-800 bg-opacity-50 rounded-xl border border-gray-700 overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1"
              >
                <a
                  href={video.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block h-full"
                >
                  <div className="relative">
                    <img src={video.thumbnail} alt={video.title} className="w-full h-48 object-cover transition-transform duration-700 group-hover:scale-105" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent flex items-center justify-center opacity-90 group-hover:opacity-100 transition-opacity duration-300">
                      <div className="bg-white/20 backdrop-blur-sm p-3 rounded-full transform transition-all duration-500 group-hover:scale-110 group-hover:bg-white/30">
                        {video.icon}
                      </div>
                    </div>
                    <div className="absolute top-3 right-3 bg-black/50 backdrop-blur-sm text-xs font-medium py-1 px-2 rounded-full flex items-center">
                      <span className="w-2 h-2 bg-red-500 rounded-full mr-1 animate-pulse"></span>
                      Premium
                    </div>
                  </div>
                  <div className="p-6">
                    <h3 className="font-bold text-xl text-white mb-2 group-hover:text-blue-300 transition-colors duration-300">{video.title}</h3>
                    <p className="text-gray-300">{video.description}</p>
                    <div className="mt-4 flex justify-between items-center">
                      <div className="text-xs text-gray-400">15 min watch</div>
                      <div className="text-blue-400 flex items-center text-sm font-medium opacity-0 transform translate-x-2 transition-all duration-300 group-hover:opacity-100 group-hover:translate-x-0">
                        Watch now <ChevronRight size={16} className="ml-1 transition-transform duration-300 group-hover:translate-x-1" />
                      </div>
                    </div>
                  </div>
                </a>
              </div>
            ))}
          </div>
        </div>

        {/* News & Blogs Section */}
        <div className={`transition-all duration-1000 delay-500 transform ${isVisible.news ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
          <div className="flex items-center justify-center gap-3 mb-10">
            <div className="h-px w-10 bg-gradient-to-r from-transparent to-blue-500 opacity-70"></div>
            <h2 className="text-2xl font-bold text-white">Market News & Trading Buzz</h2>
            <div className="h-px w-10 bg-gradient-to-l from-transparent to-blue-500 opacity-70"></div>
          </div>
          
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[
              {
                title: "How to Invest Your First $1000",
                description: "Beginner-friendly guide from Investopedia on starting with small capital. Learn strategic approaches to maximize your initial investment.",
                link: "https://www.investopedia.com/articles/basics/06/invest1000.asp",
                icon: <BookOpen size={20} />,
                color: "bg-gradient-to-br from-blue-600 to-indigo-600",
                bgColor: "bg-gray-800"
              },
              {
                title: "TradingView IG Insights",
                description: "Catch daily charts, signals, and analysis from TradingView's Instagram. Stay updated with visual market insights and expert commentary.",
                link: "https://www.instagram.com/tradingview/",
                icon: <Instagram size={20} />,
                color: "bg-gradient-to-br from-pink-600 to-purple-600",
                bgColor: "bg-gray-800"
              },
              {
                title: "Latest from Financial Times",
                description: "Live updates on stock indices, interest rates, and tech sector movement. Get professional coverage of global financial markets.",
                link: "https://www.ft.com/markets",
                icon: <Newspaper size={20} />,
                color: "bg-gradient-to-br from-cyan-600 to-blue-600",
                bgColor: "bg-gray-800"
              }
            ].map((item, idx) => (
              <div
                key={idx}
                className="group h-full bg-gray-800 bg-opacity-40 border border-gray-700 rounded-xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1"
              >
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block h-full p-6"
                >
                  <div className="flex items-center justify-between mb-6">
                    <div className={`p-3 rounded-full ${item.color} text-white`}>
                      {item.icon}
                    </div>
                    <div className="text-gray-400 opacity-0 transform translate-x-2 transition-all duration-300 group-hover:opacity-100 group-hover:translate-x-0">
                      <ExternalLink size={16} />
                    </div>
                  </div>
                  <h4 className="font-bold text-xl text-white mb-3 group-hover:text-blue-300 transition-colors duration-300">{item.title}</h4>
                  <p className="text-gray-300">{item.description}</p>
                  <div className="absolute bottom-0 left-0 right-0 h-1 opacity-0 group-hover:opacity-100 transition-opacity duration-700">
                    <div className={`h-full w-0 group-hover:w-full transition-all duration-1000 ease-out ${item.color}`}></div>
                  </div>
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* CSS for animations */}
      <style jsx>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite alternate;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </section>
  );
}