import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import os

from src.data import load_dataset
from src.model import build_model


# Path to dataset (clean labeled data)
DATASET_PATH = "./data/raw/Dungeon Crawl Stone Soup Full"

# Training parameters (CPU-safe)
BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.2


# Main python function
def main():

    print("=== Dungeon Archivist (Phase 1 Training) ===")

    # Load data/images
    print("Loading dataset...")
    X, y, label_map = load_dataset(DATASET_PATH)

    print(f"Loaded {X.shape[0]} images")
    print(f"Image shape: {X.shape[1:]}")
    print(f"Number of classes: {len(label_map)}")
    print(f"Label map: {label_map}")

    # Build model
    print("Building model...")
    model = build_model(num_classes=len(label_map))
    model.summary()

    # Train model 
    print("Training model...")
    history = model.fit(
        X, y, 
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        shuffle=True
    )

    # Save model
    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/version_model.keras")
    
    print("Training Complete")

if __name__ == "__main__":
    main()