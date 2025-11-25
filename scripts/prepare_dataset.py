import argparse
import os
import random
import shutil

def prepare_dataset(input_dir, output_dir, val_split=0.2):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    # Clear and create output directories
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Assume input_dir contains subfolders per class
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(images)

        val_count = int(len(images) * val_split)
        val_images = images[:val_count]
        train_images = images[val_count:]

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

    print(f"Dataset prepared:")
    print(f" - Training samples in: {train_dir}")
    print(f" - Validation samples in: {val_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by splitting into train and val")
    parser.add_argument("--input", required=True, help="Input directory with raw dataset")
    parser.add_argument("--output", required=True, help="Output directory for prepared dataset")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.val_split)

if __name__ == "__main__":
    main()
