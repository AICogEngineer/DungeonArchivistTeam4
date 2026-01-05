import tensorflow as tf
import os
import chromadb
import math
import numpy as np
from keras import layers, models
from src.data import load_dataset
from src.model import build_model

from sklearn.model_selection import train_test_split

# Path to dataset (clean labeled data)
DATASET_PATH = "./data/raw/Dungeon Crawl Stone Soup Full"
MODEL_DIR = "models"
VECTOR_DB_PATH = os.path.join(MODEL_DIR, "vector_db")
MODEL_PATH = os.path.join(MODEL_DIR, "version_model.keras")

# Training parameters (CPU-safe)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 64
EMBEDDING_BATCH_SIZE = 512

# Training callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

log_dir = "logs/tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

def batch_iterable(iterable, batch_size):
    """Yield successive batches from an iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# Main python function
def main():

    print("=== Dungeon Archivist (Phase 1 Training) ===")

        # Load dataset
    X, y, label_map, paths = load_dataset(DATASET_PATH)
    print(f"Loaded {X.shape[0]} images of shape {X.shape[1:]} with {len(label_map)} classes.")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=42
    )

    # Build and train model
    model = build_model(num_classes=len(label_map))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=[early_stop, tensorboard_cb]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Prepare embedding model
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output
    )

    # Initialize ChromaDB collection
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection_name = "dataset-A-embeddings"
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.get_or_create_collection(name=collection_name)

    class_id_to_label = {v: k for k, v in label_map.items()}
    num_samples = X.shape[0]
    print(f"Generating embeddings for {num_samples} images in batches of {EMBEDDING_BATCH_SIZE}...")

    total_stored = 0
    for batch_idx, batch_slice in enumerate(batch_iterable(range(num_samples), EMBEDDING_BATCH_SIZE)):
        batch_X = X[batch_slice]
        batch_embeddings = embedding_model.predict(batch_X, batch_size=EMBEDDING_BATCH_SIZE, verbose=0)

        # Prepare IDs and metadata
        batch_ids = [f"img_{i}" for i in batch_slice]
        batch_metadatas = [
            {"label": class_id_to_label[y[i]]}
            for i in batch_slice
        ]

        # Add to collection
        collection.add(ids=batch_ids, embeddings=batch_embeddings.tolist(), metadatas=batch_metadatas)
        total_stored += len(batch_ids)
        print(f"  Batch {batch_idx+1}/{math.ceil(num_samples / EMBEDDING_BATCH_SIZE)} stored.")

    print(f"All embeddings stored in ChromaDB collection '{collection_name}'.")
    print(f"Stored {total_stored} embeddings in ChromaDB")    
    print("Training Complete")

if __name__ == "__main__":
    main()