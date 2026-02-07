import cv2
import joblib
import numpy as np
from skimage.feature import hog


MODEL_PATH = "outputs/model/parking_rf_model.pkl"


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


def main():
    model = joblib.load(MODEL_PATH)

    # Put your test image here
    img_path = "dataset/test.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found:", img_path)
        print("Put an image as dataset/test.jpg")
        return

    features = extract_hog_features(img)
    features = np.array(features).reshape(1, -1)

    pred = model.predict(features)[0]

    if pred == 0:
        print("✅ Prediction: EMPTY")
    else:
        print("🚗 Prediction: OCCUPIED")


if __name__ == "__main__":
    main()
