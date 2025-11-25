import json
import os
from tensorflow.keras.preprocessing import image
import numpy as np

def load_class_mapping(path="config/class_mapping.json"):
    with open(path, "r") as f:
        return json.load(f)

def save_class_mapping(mapping, path="config/class_mapping.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

def preprocess_image(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x