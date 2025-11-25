```markdown
# Waste Classifier — Organic vs Inorganic (starter)

This repository is a starter project for classifying waste as organic vs inorganic using images and a public waste dataset (e.g., TrashNet). It uses TensorFlow / Keras with transfer learning and provides scripts to:

- Download or prepare a dataset arranged by class folders (one folder per trash class)
- Train a classifier (transfer learning with MobileNetV2)
- Run inference on single images and map predicted classes to organic/inorganic

Contents:
- src/train.py — training script
- src/infer.py — inference script (classify an image -> organic/inorganic)
- src/utils.py — helpers for dataset, mapping, model save/load
- config/class_mapping.json — map per-class labels -> "organic"/"inorganic"
- requirements.txt — Python dependencies

Dataset:
- Recommended: TrashNet (Kaggle) or any dataset with directory structure:
  dataset/
    class_1/
      img1.jpg
      ...
    class_2/
      ...
- If using Kaggle, install kaggle CLI and download TrashNet:
  1. Install and configure kaggle CLI (set KAGGLE_USERNAME and KAGGLE_KEY).
  2. `kaggle datasets download -d gspmoreira/articles-1` (or the specific dataset slug)
  3. Unzip and place into `data/` with classes as folders.

Usage (basic):

1. Create virtualenv
   python -m venv venv
   source venv/bin/activate

2. Install dependencies
   pip install -r requirements.txt

3. Prepare dataset
   Put dataset in `data/train` (subdirectories per class) and `data/val` (optional) or use `--validation_split` to split.

4. Train
   python src/train.py --data_dir data/train --epochs 10 --batch_size 32 --model_out models/waste_mobilenet.h5

5. Infer
   python src/infer.py --model models/waste_mobilenet.h5 --image sample.jpg

Mapping:
- Update `config/class_mapping.json` to designate which model classes are considered "organic" or "inorganic".

Notes & next steps:
- Add data augmentation, hyperparameter tuning, and a small web or mobile demo for camera inference.
- Optionally convert the trained model to TensorFlow Lite for mobile/edge deployment.

If you'd like, I can:
- Create this repo under your GitHub account (give repo name), or
- Push the starter files to a repo of my choice (tell me where), or
- Modify the pipeline for real-time webcam inference or object detection.
```