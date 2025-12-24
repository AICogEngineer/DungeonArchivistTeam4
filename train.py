import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import os
import chromadb
import math

from src.data import load_dataset
from src.model import build_model


# Path to dataset (clean labeled data)
DATASET_PATH = "./data/raw/Dungeon Crawl Stone Soup Full"

# Training parameters (CPU-safe)
BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 64

# Training callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

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


    # Train model 
    print("Training model...")
    history = model.fit(
        X, y, 
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        callbacks=[early_stop]
    )

    # Save model
    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/version_model.keras")

    # Create Chroma collection of embedded vectors
    client = chromadb.PersistentClient(
        path = "./models/vector_db"
    )

    collection_name = "dataset-A-embeddings"
    # Delete existing collection if it exists (for clean demo)
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.get_or_create_collection(
        name = collection_name,
        # metadata={"metric": "cosine"}
    )

    embedding_layer = model.get_layer("embedding")
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=embedding_layer.output
    )

    # Forward pass up to embedding layer
    # TODO batch?
    embeddings = embedding_model(X).numpy()

    inverse_label_map = {v: k for k, v in label_map.items()}

    ids = [f"img_{i}" for i in range(len(embeddings))]
    metadatas = [{"label": inverse_label_map[y[i]]} for i in range(len(y))]

    # Generate embeddings in batches
    EMBEDDING_BATCH_SIZE = 512 
    num_samples = len(embeddings)
    num_batches = math.ceil(num_samples / EMBEDDING_BATCH_SIZE)

    for i in range(num_batches):
        start = i * EMBEDDING_BATCH_SIZE
        end = min((i+1) * EMBEDDING_BATCH_SIZE, num_samples)

        batch_ids = ids[start:end]
        batch_embeddings = embeddings[start:end].tolist()
        batch_metadatas = metadatas[start:end]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

    print(f"Stored {len(embeddings)} embeddings in ChromaDB")
    
    print("Training Complete")

if __name__ == "__main__":
    main()