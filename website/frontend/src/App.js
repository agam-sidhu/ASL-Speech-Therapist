import React, { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Practice from "./pages/Practice";
import Results from "./pages/Results";

export default function App() {
  const [sessionData, setSessionData] = useState(null);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home setSessionData={setSessionData} />} />
        <Route path="/practice" element={<Practice sessionData={sessionData} setSessionData={setSessionData} />} />
        <Route path="/results" element={<Results sessionData={sessionData} />} />
      </Routes>
    </BrowserRouter>
  );
}