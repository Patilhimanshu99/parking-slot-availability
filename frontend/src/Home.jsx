import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleSubmit = async () => {
    if (!image) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("image", image);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    setTimeout(() => {
      navigate("/result", { state: data });
    }, 1200);
  };

  return (
    <div className="min-h-screen flex items-center justify-center 
    bg-blue-100">

      {/* MAIN CARD */}
      <div className="bg-white rounded-2xl shadow-xl p-10 w-[700px]">

        <h1 className="text-4xl font-semibold text-center mb-8 text-gray-800">
          🚗 Parking Slot Detection
        </h1>

        <div className="grid grid-cols-2 gap-10 items-center">

          {/* LEFT */}
          <div>
            <p className="mb-3 font-medium text-gray-700">
              Upload Image
            </p>

            <input type="file" onChange={handleUpload} className="mb-6" />

            <button
              onClick={handleSubmit}
              className="px-6 py-2 rounded-full font-medium
              bg-rose-500 text-white
              transition-all duration-300
              hover:shadow-[0_0_20px_rgba(244,63,94,0.8)]
              hover:scale-105"
            >
              Detect Slots
            </button>

            {/* LOADING */}
            {loading && (
              <div className="mt-6 flex items-center gap-3 text-blue-600">
                <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                Processing image...
              </div>
            )}
          </div>

          {/* RIGHT PREVIEW */}
          <div className="h-[200px] w-[260px] mx-auto 
          bg-gray-100 rounded-xl 
          flex items-center justify-center overflow-hidden border">

            {preview ? (
              <img
                src={preview}
                className="max-h-full max-w-full object-contain"
              />
            ) : (
              <p className="text-gray-400">Preview</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}