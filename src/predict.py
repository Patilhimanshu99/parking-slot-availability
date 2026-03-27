import cv2
import joblib
import numpy as np
from skimage.feature import hog
import os

MODEL_PATH = "outputs/model/parking_rf_model.pkl"


def extract_hog_features(img):
    # Resize image
    img = cv2.resize(img, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print("Model file not found:", MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)

    # Test image path
    img_path = "dataset/test.jpg"

    # Check image exists
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        print("Please add an image at dataset/test.jpg")
        return

    # Read image
    img = cv2.imread(img_path)

    # Extract features
    features = extract_hog_features(img)
    features = np.array(features).reshape(1, -1)

    # Predict
    pred = model.predict(features)[0]

    # Output result
    if pred == 0:
        print("Prediction: EMPTY")
    else:
        print("Prediction: OCCUPIED")


if __name__ == "__main__":
    main()