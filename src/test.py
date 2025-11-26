# test.py
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
import requests
from urllib.parse import urlparse
import tempfile

def download_image(url):
    """Download image from a URL and return a temporary file path."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image: {url}")

    suffix = os.path.splitext(urlparse(url).path)[1]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name

def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_url(path):
    return path.startswith("http://") or path.startswith("https://")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model (.h5)')
    parser.add_argument('--img', required=True, help='Image path, folder, or URL (Imgur supported)')
    parser.add_argument('--output', default='predictions2.csv', help='CSV file to save predictions')
    args = parser.parse_args()

    model = load_model(args.model)
    class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    results = []

    # Handle URL input
    if is_url(args.img):
        print(f"Downloading image from URL: {args.img}")
        downloaded_path = download_image(args.img)
        img_array = load_and_preprocess(downloaded_path)
        pred = model.predict(img_array)
        predicted_class = class_labels[np.argmax(pred)]
        results.append({'image': args.img, 'prediction': predicted_class})

    elif os.path.isdir(args.img):
        for root, dirs, files in os.walk(args.img):
            for img_file in files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, img_file)
                    img_array = load_and_preprocess(img_path)
                    pred = model.predict(img_array)
                    predicted_class = class_labels[np.argmax(pred)]
                    results.append({'image': os.path.relpath(img_path, args.img),
                                    'prediction': predicted_class})
    else:
        # Single local image
        img_array = load_and_preprocess(args.img)
        pred = model.predict(img_array)
        predicted_class = class_labels[np.argmax(pred)]
        results.append({'image': os.path.basename(args.img), 'prediction': predicted_class})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print(df.head(20))

if __name__ == '__main__':
    main()
