import { Routes, Route } from "react-router-dom";
import Home from "./Home";
import ResultPage from "./ResultPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/result" element={<ResultPage />} />
    </Routes>
  );
}