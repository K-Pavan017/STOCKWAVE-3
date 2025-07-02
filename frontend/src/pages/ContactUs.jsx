import { useState, useEffect } from 'react';
import { Phone, Mail, MapPin, ArrowRight, ArrowUpRight, Send } from 'lucide-react';

import Navbar from '../components/NavBar';
import Footer from '../components/Footer';

// Contact Us Page Component
export default function ContactUs() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission here
    console.log(formData);
    // Reset form
    setFormData({
      name: '',
      email: '',
      subject: '',
      message: ''
    });
    alert('Message sent successfully!');
  };
  
  // Animation for elements
  const [visible, setVisible] = useState(false);
  
  useEffect(() => {
    setVisible(true);
  }, []);
  
  const animationClasses = (delay) => `transform transition-all duration-1000 ease-out ${visible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'} ${delay}`;

  return (
    <>
      <Navbar />
      <div className="pt-24 pb-16 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
        {/* Background decorative elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob"></div>
          <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-2000"></div>
          <div className="absolute bottom-1/4 right-1/3 w-72 h-72 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-blob animation-delay-4000"></div>
        </div>
        
        {/* Hero Section */}
        <section className="container mx-auto px-6 py-12 relative z-10">
          <div className={`text-center max-w-3xl mx-auto ${animationClasses("delay-100")}`}>
            <div className="inline-block px-4 py-1 mb-4 text-sm font-semibold text-blue-300 bg-blue-900 bg-opacity-30 rounded-full">
              Get In Touch
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-blue-200 text-transparent bg-clip-text">
              We'd Love to <span className="text-blue-400">Hear From You</span>
            </h1>
            <p className="text-gray-300 text-lg">
              Have questions about our stock prediction technology? Want to learn more about how our AI can enhance your investment strategy? Our team is here to help.
            </p>
          </div>
        </section>

        {/* Contact Options */}
        <section className="container mx-auto px-6 py-8 relative z-10">
          <div className={`grid grid-cols-1 md:grid-cols-3 gap-8 mb-16 ${animationClasses("delay-300")}`}>
            {[
              {
                icon: <Phone size={24} className="text-blue-400" />,
                title: "Call Us",
                info: "+1 (555) 123-4567",
                description: "Available Mon-Fri, 9AM-5PM ET"
              },
              {
                icon: <Mail size={24} className="text-blue-400" />,
                title: "Email Us",
                info: "support@stockai.tech",
                description: "We'll respond within 24 hours"
              },
              {
                icon: <MapPin size={24} className="text-blue-400" />,
                title: "Visit Us",
                info: "123 Innovation Drive",
                description: "Boston, MA 02110"
              }
            ].map((option, index) => (
              <div key={index} className="bg-gray-800 rounded-xl border border-gray-700 shadow-md p-8 text-center transition-transform hover:scale-105">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-900 bg-opacity-30 rounded-full mb-6">
                  {option.icon}
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{option.title}</h3>
                <div className="text-blue-300 font-medium mb-2">{option.info}</div>
                <p className="text-gray-400">{option.description}</p>
              </div>
            ))}
          </div>

          {/* Contact Form and Map */}
          <div className={`grid grid-cols-1 lg:grid-cols-2 gap-12 ${animationClasses("delay-500")}`}>
            <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-md overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"></div>
              <div className="p-8">
                <h2 className="text-2xl font-bold text-white mb-6">Send Us a Message</h2>
                <form onSubmit={handleSubmit}>
                  <div className="space-y-6">
                    <div>
                      <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-1">Your Name</label>
                      <input
                        type="text"
                        id="name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-white placeholder-gray-400"
                        placeholder="John Doe"
                        required />
                    </div>

                    <div>
                      <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-1">Your Email</label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-white placeholder-gray-400"
                        placeholder="example@email.com"
                        required />
                    </div>

                    <div>
                      <label htmlFor="subject" className="block text-sm font-medium text-gray-300 mb-1">Subject</label>
                      <input
                        type="text"
                        id="subject"
                        name="subject"
                        value={formData.subject}
                        onChange={handleChange}
                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-white placeholder-gray-400"
                        placeholder="How can we help you?"
                        required />
                    </div>

                    <div>
                      <label htmlFor="message" className="block text-sm font-medium text-gray-300 mb-1">Your Message</label>
                      <textarea
                        id="message"
                        name="message"
                        value={formData.message}
                        onChange={handleChange}
                        rows="5"
                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition text-white placeholder-gray-400"
                        placeholder="Tell us what you need..."
                        required
                      ></textarea>
                    </div>

                    <button
                      type="submit"
                      className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-600 text-white px-6 py-3 rounded-lg font-medium transition transform hover:scale-105 flex items-center justify-center"
                    >
                      Send Message
                      <Send size={18} className="ml-2" />
                    </button>
                  </div>
                </form>
              </div>
            </div>
            
            {/* Right side could be a map or additional content */}
            <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-md p-8">
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"></div>
              <h2 className="text-2xl font-bold text-white mb-6">Our Location</h2>
              <div className="bg-gray-700 h-64 rounded-lg mb-6 flex items-center justify-center">
                <MapPin size={48} className="text-blue-400 opacity-50" />
                <span className="text-gray-400 ml-2">Map placeholder</span>
              </div>
              <div className="text-gray-300">
                <p className="mb-2"><strong>Address:</strong> 123 Innovation Drive, Boston, MA 02110</p>
                <p className="mb-2"><strong>Hours:</strong> Monday-Friday, 9AM-5PM ET</p>
                <p><strong>Parking:</strong> Available on premises</p>
              </div>
            </div>
          </div>
        </section>

        {/* FAQ Section */}
        <section className="container mx-auto px-6 py-16 relative z-10">
          <div className={`text-center max-w-3xl mx-auto mb-12 ${animationClasses("delay-700")}`}>
            <h2 className="text-3xl font-bold text-white mb-4">Frequently Asked Questions</h2>
            <p className="text-gray-300">
              Find answers to common questions about our stock prediction platform and services.
            </p>
          </div>

          <div className={`grid grid-cols-1 md:grid-cols-2 gap-8 ${animationClasses("delay-900")}`}>
            {[
              {
                question: "How accurate are your stock predictions?",
                answer: "Our AI models consistently achieve over 90% accuracy in backtesting. Real-world performance may vary based on market conditions, but we continuously improve our algorithms to maintain high accuracy rates."
              },
              {
                question: "How is my data protected?",
                answer: "We implement bank-level encryption to protect all user data. We do not share your personal information with third parties, and all predictions are generated using anonymized data models."
              },
              {
                question: "Can I integrate StockAI with my existing trading platform?",
                answer: "Yes! We offer API access that allows seamless integration with most popular trading platforms and financial software. Contact our support team for specific integration guidance."
              },
              {
                question: "What financial markets do you cover?",
                answer: "We currently provide predictions for major US stock markets (NYSE, NASDAQ), with plans to expand to international markets and additional financial instruments in the near future."
              },
              {
                question: "How often are predictions updated?",
                answer: "Our standard service provides daily prediction updates. Premium users receive real-time updates throughout trading hours, with alerts for significant market shifts."
              },
              {
                question: "Do you offer educational resources?",
                answer: "Yes, we provide comprehensive educational resources including webinars, tutorials, and a knowledge base to help you understand our prediction methodology and make the most of our platform."
              }
            ].map((faq, index) => (
              <div key={index} className="bg-gray-800 rounded-xl border border-gray-700 shadow-md p-6">
                <h3 className="text-xl font-bold text-white mb-3">{faq.question}</h3>
                <p className="text-gray-300">{faq.answer}</p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="container mx-auto px-6 py-12 relative z-10">
          <div className={`bg-gradient-to-r from-blue-600 to-indigo-700 rounded-2xl p-12 text-center ${animationClasses("delay-1000")}`}>
            <h2 className="text-3xl font-bold text-white mb-6">Ready to Experience the Future of Stock Prediction?</h2>
            <p className="text-blue-100 mb-8 max-w-2xl mx-auto">
              Join thousands of investors who are already using our AI-powered platform to make smarter investment decisions.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <button className="px-8 py-3 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-600 text-white rounded-xl font-semibold transition-all duration-300 transform hover:translate-y-px hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 group">
                <span className="flex items-center">
                  Sign Up Now
                  <ArrowRight size={18} className="ml-1 transition-transform duration-300 group-hover:translate-x-1" />
                </span>
              </button>
              <button className="px-8 py-3 border-2 border-gray-600 hover:border-blue-400 text-gray-200 hover:text-white rounded-xl font-semibold transition-all duration-300 hover:bg-gray-800 hover:bg-opacity-50 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-50">
                <span className="flex items-center">
                  Request a Demo
                  <ArrowUpRight size={18} className="ml-2" />
                </span>
              </button>
            </div>
          </div>
        </section>
      </div>
      <Footer />
      
      {/* Add CSS for animations (in a real application this would go in your stylesheet) */}
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
    </>
  );
}