# test.py
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import pandas as pd

def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model (.h5)')
    parser.add_argument('--img', required=True, help='Path to image or folder of images')
    parser.add_argument('--output', default='predictions.csv', help='CSV file to save predictions')
    args = parser.parse_args()

    model = load_model(args.model)
    class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    results = []

    if os.path.isdir(args.img):
        for root, dirs, files in os.walk(args.img):
            for img_file in files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, img_file)
                    img_array = load_and_preprocess(img_path)
                    pred = model.predict(img_array)
                    predicted_class = class_labels[np.argmax(pred)]
                    results.append({'image': os.path.relpath(img_path, args.img), 'prediction': predicted_class})

    else:
        # single image
        img_array = load_and_preprocess(args.img)
        pred = model.predict(img_array)
        predicted_class = class_labels[np.argmax(pred)]
        results.append({'image': os.path.basename(args.img), 'prediction': predicted_class})

    # save all predictions to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    # print only first 10 rows
    print(df.head(20))

if __name__ == '__main__':
    main()
