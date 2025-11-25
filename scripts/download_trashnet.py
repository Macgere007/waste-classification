#!/usr/bin/env python3
"""
Download the TrashNet dataset from Kaggle.

Usage:
  1) Make sure you have a Kaggle API token (kaggle.json) saved at:
       - Linux / macOS: ~/.kaggle/kaggle.json
       - Windows: C:/Users/<you>/.kaggle/kaggle.json
     (Create token on kaggle.com -> Account -> Create New API Token)

  2) Activate your virtualenv and install the kaggle package:
       pip install kaggle

  3) Run:
       python scripts/download_trashnet.py --output data/raw

This will download and unzip files into the output folder.
"""
import argparse
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_trashnet(output_dir="data/raw", dataset_slug="feyzazkefe/trashnet"):
    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {dataset_slug} to {output_dir} ...")
    api.dataset_download_files(dataset_slug, path=output_dir, unzip=True, quiet=False)
    print("Download complete. Files are in:", os.path.abspath(output_dir))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw", help="Output directory for downloaded dataset")
    parser.add_argument("--slug", default="feyzazkefe/trashnet", help="Kaggle dataset slug (owner/dataset)")
    args = parser.parse_args()
    download_trashnet(args.output, args.slug)

if __name__ == "__main__":
    main()