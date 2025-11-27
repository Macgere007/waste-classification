import os
import matplotlib.pyplot as plt

DATA_DIR = "data"

def count_images(path):
    counts = {}
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = sum(
                1 for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            )
    return counts

train_counts = count_images(os.path.join(DATA_DIR, "train"))

# --- FULL-WIDTH PLOT ---
plt.figure(figsize=(12, 6))
plt.bar(train_counts.keys(), train_counts.values())
plt.title("Train Dataset Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
