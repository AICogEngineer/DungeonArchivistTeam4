import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Process the image, converting to a numpy array
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    pass

# Go through and process every image
def process_images(root_dir='./data'):
    root = Path(root_dir)
    for label_dir in root.iterdir():
        # Ignore files - only folders
        if not label_dir.is_dir():
            continue
        label = label_dir.name

        for image_path in label_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() == '.png':
                # Process image
                pass

# Process images and train the model on them 
def train_on_images():
    process_images()

# Main python function
def main():
    # Load data/images

    # Build training dataset
    train_on_images()

if __name__ == "__main__":
    main()