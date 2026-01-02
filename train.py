import tensorflow as tf
import os
import chromadb
import math
import numpy as np
from src.data import load_dataset
from src.model import build_model

from sklearn.model_selection import train_test_split

# Path to dataset (clean labeled data)
DATASET_PATH = "./data/raw/Dungeon Crawl Stone Soup Full"

# Training parameters (CPU-safe)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 64

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

# Main python function
def main():

    print("=== Dungeon Archivist (Phase 1 Training) ===")

    # Load data/images
    print("Loading dataset...")
    X, y, label_map, paths = load_dataset(DATASET_PATH)

    # Split data set
    print("Splitting dataset into training and validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    print("Training labels range:", np.unique(y_train))
    print("Validation labels range:", np.unique(y_val))
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)

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
        X_train, y_train,   
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[early_stop, tensorboard_cb]
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
        metadata={"metric": "cosine"}   # Cosine is the default
    )

    embedding_layer = model.get_layer("embedding")
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=embedding_layer.output
    )

    # Forward pass up to embedding layer, generating embeddings
    # TODO batch?
    embeddings = embedding_model(X).numpy()

    # embeddings = embedding_model.predict(X, batch_size=EMBEDDING_BATCH_SIZE)

    class_id_to_label = {v: k for k, v in label_map.items()}

    # Get list of ids and metadatas
    ids = [f"img_{i}" for i in range(len(embeddings))]
    metadatas = [
        {
            "label": class_id_to_label[y[i]],
            "rel_path": paths[i]
        }
        for i in range(len(y))
    ]

    # Save embeddings in batches
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