import os
import shutil
import random

# Paths
input_path = "dataset/extracted_frames"
train_path = "dataset/train"
val_path = "dataset/val"

# Create train/val directories
os.makedirs(f"{train_path}/real", exist_ok=True)
os.makedirs(f"{train_path}/fake", exist_ok=True)
os.makedirs(f"{val_path}/real", exist_ok=True)
os.makedirs(f"{val_path}/fake", exist_ok=True)

def split_data(source_folder, train_dest, val_dest, split_ratio=0.8):
    """
    Splits dataset into training and validation sets.
    """
    files = os.listdir(source_folder)
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    for file in files[:split_index]:
        shutil.move(os.path.join(source_folder, file), train_dest)
    for file in files[split_index:]:
        shutil.move(os.path.join(source_folder, file), val_dest)

split_data(f"{input_path}/real", f"{train_path}/real", f"{val_path}/real")
split_data(f"{input_path}/fake", f"{train_path}/fake", f"{val_path}/fake")

print("✅ Data split into training and validation sets successfully!")
