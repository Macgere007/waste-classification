# detect_object_only_no_hand_filtered_v2.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

def load_and_preprocess(img, target_size=(64, 64)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Calculate how "thin" the contour is
def contour_thinness(w, h):
    return min(w, h)

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

        # ------------------------------
        # (1) SKIN COLOR REMOVAL
        # ------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV skin color range (adjustable)
        lower_skin = np.array([0, 28, 60], dtype=np.uint8)
        upper_skin = np.array([25, 180, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask = cv2.medianBlur(skin_mask, 7)

        # ------------------------------
        # (2) BACKGROUND SUBTRACTION
        # ------------------------------
        fgmask = fgbg.apply(frame)

        # Remove skin from fgmask
        fgmask = cv2.bitwise_and(fgmask, cv2.bitwise_not(skin_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (6000 < area < 70000):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # ------------------------------
            # (3) FINGER REMOVAL RULES
            # ------------------------------

            # (a) thinness → fingers = very thin
            if contour_thinness(w, h) < 35:
                continue

            # (b) extreme elongation → fingers/palm edge
            ratio = max(w, h) / min(w, h)
            if ratio > 2.5:
                continue

            # (c) reject perfect roundish shapes → palm
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            if circularity > 0.80:
                continue

            detected = (x, y, w, h)
            break

        # stability check
        stable_history.append(detected is not None)

        if sum(stable_history) >= 3 and detected:
            x, y, w, h = detected
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            crop = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img = load_and_preprocess(crop)
            pred = model.predict(img)
            cls = class_labels[np.argmax(pred)]
            conf = np.max(pred)

            cv2.putText(frame, f"{cls} ({conf*100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No object detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2)

        cv2.imshow("Object Only Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
