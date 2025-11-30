#!/usr/bin/env python3
"""
Download the Waste Classification dataset from Kaggle.

Usage:
  1) Make sure you have a Kaggle API token (kaggle.json) saved at:
       - Linux / macOS: ~/.kaggle/kaggle.json
       - Windows: C:/Users/<you>/.kaggle/kaggle.json
     (Create token on kaggle.com -> Account -> Create New API Token)

  2) Activate your virtualenv and install the kaggle package:
       pip install kaggle

  3) Run:
       # Downloads and unzips into data/raw/waste-classification-data
       python download_dataset.py --output data/raw

     OR, if you want the output folder to be named 'DATASET':
       python download_dataset.py --output DATASET --slug techsash/waste-classification-data

This will download and unzip files into the output folder.
"""
import argparse
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(output_dir="data/raw", dataset_slug="techsash/waste-classification-data"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        api = KaggleApi()
        # Authenticate using the kaggle.json file
        api.authenticate()
        
        print(f"Downloading {dataset_slug} to {output_dir} ...")
        # Download and unzip the files
        api.dataset_download_files(dataset_slug, path=output_dir, unzip=True, quiet=False)
        
        print("Download complete.")
        print("Files are now in:", os.path.abspath(output_dir))
        
    except Exception as e:
        print(f"\nERROR: Failed to download dataset.")
        print("Please check your internet connection and ensure your Kaggle API token is correctly set up.")
        print(f"Details: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download a specified dataset from Kaggle.")
    parser.add_argument("--output", default="data/raw", help="Output directory for downloaded dataset")
    # Updated default slug to the widely used Waste Classification data
    parser.add_argument("--slug", default="techsash/waste-classification-data", help="Kaggle dataset slug (owner/dataset)")
    args = parser.parse_args()
    download_dataset(args.output, args.slug)

if __name__ == "__main__":
    main()