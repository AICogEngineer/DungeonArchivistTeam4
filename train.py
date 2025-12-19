import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import chromadb

def get_image_paths(root_dir='./data'):
    root = Path(root_dir)
    image_paths = []
    for label_dir in root.iterdir():
        # Ignore files - only folders
        if not label_dir.is_dir():
            continue

        for image_path in label_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() == '.png':
                image_paths.append(str(image_path.relative_to(root)))
    return image_paths

# Process the image, converting to a numpy array
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    pass

# Go through and process every image
def process_images(image_paths):
    pass

# Process images and train the model on them 
def train_on_images():
    process_images()

def save_model(train_dataset):
    client = chromadb.Client()
    collection = client.create_collection("dungeon_tiles")
    pass

# Main python function
def main():
    # Load data/images
    image_paths = get_image_paths('./data')
    print(image_paths)
    # Build training dataset
    process_images(image_paths)
    # train_on_images()

    # Save to ChromaDB

if __name__ == "__main__":
    main()