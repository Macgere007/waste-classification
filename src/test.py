# detect_object_only_no_hand_filtered.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

def load_and_preprocess(img, target_size=(64, 64)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def contour_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0: 
        return 0
    return float(area) / hull_area

def contour_extent(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area == 0:
        return 0
    return float(area) / rect_area

def main():
    model_path = "models/waste_cnn.h5"
    class_labels = ['organic', 'inorganic']
    model = load_model(model_path)
    print("Model loaded successfully!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    stable_history = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = fgbg.apply(frame)

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (8000 < area < 60000):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h

            # Reject thin shapes (fingers)
            if aspect < 0.4 or aspect > 3.0:
                continue

            # Hand detection rejection: low solidity + low extent
            sol = contour_solidity(cnt)   # hands: 0.2–0.6, objects: 0.7–1.0
            ext = contour_extent(cnt)     # hands: low, objects: medium-high

            if sol < 0.65 or ext < 0.45:
                continue  # likely fingers or palm

            # Passed filters → consider as object
            detected = (x, y, w, h)
            break

        stable_history.append(detected is not None)

        if sum(stable_history) >= 3 and detected:
            x, y, w, h = detected
            pad = 20
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            crop = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            img = load_and_preprocess(crop)
            pred = model.predict(img)
            cls = class_labels[np.argmax(pred)]
            conf = np.max(pred)

            cv2.putText(frame, f"{cls} ({conf*100:.1f}%)",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No object detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Object Only Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
