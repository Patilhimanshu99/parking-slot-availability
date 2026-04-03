import cv2
import numpy as np
import joblib

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


def predict_slot(model, img):
    features = extract_hog_features(img)
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]
    return pred


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


def main():
    model = joblib.load(MODEL_PATH)

    img_path = "dataset/test.jpg"
    image = cv2.imread(img_path)

    slots = split_image(image, rows=2, cols=3)

    results = []

    for slot in slots:
        pred = predict_slot(model, slot)
        results.append(pred)

    # Display results
    print("\nSlot Results:")
    for i, r in enumerate(results):
        status = "EMPTY" if r == 0 else "OCCUPIED"
        print(f"Slot {i+1}: {status}")

    print("\nSummary:")
    print("Total:", len(results))
    print("Empty:", results.count(0))
    print("Occupied:", results.count(1))


if __name__ == "__main__":
    main()