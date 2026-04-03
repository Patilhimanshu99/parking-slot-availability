import { useLocation, useNavigate } from "react-router-dom";

export default function ResultPage() {
  const { state } = useLocation();
  const navigate = useNavigate();

  if (!state) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-blue-100">
        <p className="text-black">No Data Found</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-blue-100 flex items-center justify-center">

      {/* MAIN CARD */}
      <div className="bg-white rounded-2xl shadow-xl p-10 w-[900px]">

        {/* TITLE */}
        <h1 className="text-3xl font-semibold mb-8 text-black">
          📊 Parking Results
        </h1>

        {/* STATS */}
        <div className="flex gap-12 mb-8 text-black">
          <div>
            <p className="text-3xl font-bold">{state.total}</p>
            <p>Total</p>
          </div>

          <div>
            <p className="text-3xl font-bold">{state.empty}</p>
            <p>Empty</p>
          </div>

          <div>
            <p className="text-3xl font-bold">{state.occupied}</p>
            <p>Occupied</p>
          </div>
        </div>

        {/* LEGEND */}
        <div className="flex gap-6 mb-6 text-sm text-black">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500 rounded"></div>
            <span>Empty</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Occupied</span>
          </div>
        </div>

        {/* SLOT GRID */}
        <div className="grid grid-cols-8 gap-4 mb-10">
          {state.results.map((r, i) => (
            <div
              key={i}
              className={`h-14 flex items-center justify-center rounded-lg font-bold text-black transition
              ${
                r === 0
                  ? "bg-blue-500 hover:bg-blue-600"   // EMPTY = BLUE
                  : "bg-red-500 hover:bg-red-600"     // OCCUPIED = RED
              }`}
            >
              {i + 1}
            </div>
          ))}
        </div>

        {/* BACK BUTTON */}
        <button
          onClick={() => navigate("/")}
          className="px-6 py-2 rounded-full bg-blue-500 text-white
          transition-all duration-300
          hover:shadow-lg hover:scale-105"
        >
          Back
        </button>
      </div>
    </div>
  );
}