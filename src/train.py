import os
import cv2
import joblib
import numpy as np

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATASET_DIR = "dataset"
EMPTY_DIR = os.path.join(DATASET_DIR, "empty")
OCCUPIED_DIR = os.path.join(DATASET_DIR, "occupied")

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


def load_data():
    X = []
    y = []

    # Empty = 0
    for file in os.listdir(EMPTY_DIR):
        path = os.path.join(EMPTY_DIR, file)
        img = cv2.imread(path)
        if img is None:
            continue
        X.append(extract_hog_features(img))
        y.append(0)

    # Occupied = 1
    for file in os.listdir(OCCUPIED_DIR):
        path = os.path.join(OCCUPIED_DIR, file)
        img = cv2.imread(path)
        if img is None:
            continue
        X.append(extract_hog_features(img))
        y.append(1)

    return np.array(X), np.array(y)


def main():
    print("Loading dataset...")
    X, y = load_data()

    print("Total samples:", len(y))
    print("Empty:", np.sum(y == 0))
    print("Occupied:", np.sum(y == 1))

    print("\nSplitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("\nTesting...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", round(acc * 100, 2), "%")

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs("outputs/model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
