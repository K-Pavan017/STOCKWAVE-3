import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import ContactUs from './pages/ContactUs';
import AboutUs from './pages/AboutUs';
import Dashboard from './pages/Dashboard';

import AppLoadingAnimation from './pages/AppLoadingAnimation';
import SignUp from './pages/SignUp';
import Login from './pages/Login';
import StockData from './pages/StockData';

import './App.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* <Route path="/" element={<Home />} /> */}
        <Route path="/" element={<AppLoadingAnimation />} />
        <Route path="/home" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/contact" element={<ContactUs />} />
        <Route path="/about" element={<AboutUs />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/loading" element={<AppLoadingAnimation />} />

        <Route path="/stock" element={<StockData/>} />

      </Routes>
    </BrowserRouter>
  );
}

export default App
