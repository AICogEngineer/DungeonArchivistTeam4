import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import os
import random
import shutil

from io_ops import clear_directories, clear_or_create_folder

source_folder = "./data/raw/Dungeon Crawl Stone Soup Full"
target_folder = "./data/chaos"
num_images = 100
allowed_extensions = {'.jpg', '.jpeg', '.png'}

def generate_hex_name(used_names):
    while True:
        name = ''.join(random.choices('0123456789abcdef', k=6))
        if name not in used_names:
            used_names.add(name)
            return name

def main():
    # Delete existing target folder if it exists
    clear_or_create_folder(target_folder)

    # Recursively gather all image file paths
    all_images = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in allowed_extensions:
                all_images.append(os.path.join(root, file))

    # Count of images in original folder
    total_images = len(all_images)
    print(f"Total images found in the original folder: {total_images}")

    # Randomly select images
    if len(all_images) < num_images:
        raise ValueError(f"Not enough images to select {num_images}. Found {len(all_images)} images.")
    selected_images = random.sample(all_images, num_images)

    used_names = set()
    for img_path in selected_images:
        ext = os.path.splitext(img_path)[1].lower()
        hex_name = generate_hex_name(used_names) + ext
        dest_path = os.path.join(target_folder, hex_name)
        shutil.copy2(img_path, dest_path)

    print(f"Copied {num_images} images to '{target_folder}' with unique 6-character hexadecimal filenames.")


if __name__ == "__main__":
    main()