import argparse
import os
import random
import shutil

def prepare_dataset(input_dir, output_dir, val_split=0.2):
    """
    Splits images from class subfolders in input_dir into separate 
    train and val directories in output_dir, maintaining class balance.
    
    Args:
        input_dir (str): Directory containing class subfolders (e.g., 'Organic', 'Recyclable').
        output_dir (str): Root directory where the 'train' and 'val' folders will be created.
        val_split (float): Ratio of data to be allocated to the validation set.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    # 

    # Clear and create output directories
    print(f"Preparing dataset into: {output_dir}")
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            print(f"Warning: Clearing existing directory {d}")
            shutil.rmtree(d)
        os.makedirs(d)

    # Assume input_dir contains subfolders per class (e.g., 'O', 'R')
    classes = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')]
    
    if not classes:
        print(f"Error: No class subfolders found in {input_dir}. Please check input structure.")
        return

    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        # Filter for common image files (case-insensitive)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"Warning: No images found in class folder {cls_path}. Skipping.")
            continue
            
        random.shuffle(images)

        val_count = int(len(images) * val_split)
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        # Print class split statistics
        print(f"  Class '{cls}': Total={len(images)}, Train={len(train_images)}, Val={len(val_images)}")

        # Copy validation images
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(val_cls_dir, exist_ok=True)
        for img in val_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(val_cls_dir, img))

        # Copy training images
        train_cls_dir = os.path.join(train_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        for img in train_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(train_cls_dir, img))

    print("\nDataset preparation complete.")
    print(f" - Training samples are ready in: {train_dir}")
    print(f" - Validation samples are ready in: {val_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by splitting into train and val folders.")
    parser.add_argument("--input", required=True, help="Input directory with raw dataset (containing class subfolders).")
    parser.add_argument("--output", required=True, help="Output directory for prepared dataset (will contain train/ and val/ subfolders).")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (e.g., 0.2 means 20% for validation).")
    args = parser.parse_args()

    # The dataset downloaded from Kaggle is named 'waste-classification-data'.
    # The actual images are usually in a subdirectory inside the downloaded folder.
    # We assume 'args.input' points to the directory containing the 'O' and 'R' folders.
    prepare_dataset(args.input, args.output, args.val_split)

if __name__ == "__main__":
    main()