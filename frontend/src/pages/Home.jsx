// Home.jsx
import React, { useState } from 'react';
import NavBar from '../components/NavBar';
import Hero from '../components/Hero';
import HowItWorks from '../components/HowItWorks';
import Footer from '../components/Footer';
import LearnAndNews from '../components/LearnAndNews';

export default function Home() {
  const [darkMode, setDarkMode] = useState(false);

  return (
    <>
      <NavBar></NavBar>
        <Hero darkMode={darkMode} setDarkMode={setDarkMode} />
        <HowItWorks />
        <LearnAndNews />
        <Footer />
    </>
  );
}
