import os
import random

# Paths
train_real = "dataset/train/real"
train_fake = "dataset/train/fake"
val_real = "dataset/val/real"
val_fake = "dataset/val/fake"

def keep_half_images(folder):
    """
    Keeps only 50% of images in the folder and deletes the rest.
    """
    files = os.listdir(folder)
    random.shuffle(files)  # Shuffle files randomly
    half_size = len(files) // 2  # Calculate 50%

    for file in files[half_size:]:  # Keep only first 50%
        os.remove(os.path.join(folder, file))  # Delete extra files

# Apply size reduction
keep_half_images(train_real)
keep_half_images(train_fake)
keep_half_images(val_real)
keep_half_images(val_fake)

print("✅ Dataset reduced to 50% of its original size!")
