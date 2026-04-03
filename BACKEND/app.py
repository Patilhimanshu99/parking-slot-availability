from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib

from skimage.feature import hog

app = Flask(__name__)
CORS(app)

MODEL_PATH = "outputs/model/parking_rf_model.pkl"
model = joblib.load(MODEL_PATH)


def extract_hog_features(img):
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


def predict_slot(img):
    features = extract_hog_features(img)
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]


def split_image(image, rows=2, cols=3):
    h, w, _ = image.shape
    slot_h = h // rows
    slot_w = w // cols

    slots = []

    for i in range(rows):
        for j in range(cols):
            y1 = i * slot_h
            y2 = (i + 1) * slot_h
            x1 = j * slot_w
            x2 = (j + 1) * slot_w

            slot = image[y1:y2, x1:x2]
            slots.append(slot)

    return slots


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    slots = split_image(image, rows=2, cols=3)

    results = []
    for slot in slots:
        pred = predict_slot(slot)
        results.append(int(pred))

    return jsonify({
        "results": results,
        "total": len(results),
        "empty": results.count(0),
        "occupied": results.count(1)
    })


if __name__ == "__main__":
    app.run(debug=True)