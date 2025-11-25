"""
Load trained model and classify a single image, then map predicted class to organic/inorganic.
Usage:
python src/infer.py --model models/waste_mobilenet.h5 --image sample.jpg
"""
import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import preprocess_image, load_class_mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--class_indices", default=None, help="Path to model_class_indices json (created by train)")
    parser.add_argument("--mapping", default="config/class_mapping.json", help="map class -> organic|inorganic")
    args = parser.parse_args()

    model = load_model(args.model)
    x = preprocess_image(args.image, target_size=(224,224))
    preds = model.predict(x)
    idx = int(np.argmax(preds[0]))
    if args.class_indices:
        with open(args.class_indices, "r") as f:
            class_map = json.load(f)   # { "0": "paper", ... }
        predicted_class = class_map[str(idx)]
    else:
        # if model saved with class indices file next to it, attempt to load
        import os
        base = os.path.splitext(args.model)[0]
        path = base + "_class_indices.json"
        try:
            with open(path, "r") as f:
                class_map = json.load(f)
            predicted_class = class_map[str(idx)]
        except Exception:
            predicted_class = str(idx)

    mapping = load_class_mapping(args.mapping)
    organicity = mapping.get(predicted_class, "unknown")

    print(f"Predicted class: {predicted_class} (index {idx}), organicity: {organicity}")
    print(f"Raw probs: {preds[0].tolist()}")

if __name__ == "__main__":
    main()