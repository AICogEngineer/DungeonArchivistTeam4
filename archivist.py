from pathlib import Path
import os
import shutil
import math
import numpy as np
from tensorflow import keras
from keras import models
import chromadb
from src.data import preprocess_for_interface
from src.io_ops import clear_or_create_folder

# Paths
MODEL_PATH = Path("models/version_model.keras")
CHAOS_DIR = Path("data/chaos_data")
SORTED_DIR = Path("data/restored_archive")
REVIEW_DIR = Path("data/review_pile")

# Parameters
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.7
DEBUG_OUTPUT = False  # Set False for normal operation
BATCH_SIZE = 32      # For querying multiple images at once


def weighted_vote_distance(neighbors, similarities):
    votes = {}
    for neighbor, sim in zip(neighbors, similarities):
        label = neighbor['label']
        votes[label] = votes.get(label, 0) + sim  # weight by similarity
    best_label = max(votes, key=votes.get)
    confidence = votes[best_label] / sum(votes.values())
    return best_label, confidence


def batch_iterable(iterable, batch_size):
    """Yield successive batches from an iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def run_archivist():
    # Clear folders for sorted and reviewable images
    clear_or_create_folder(str(SORTED_DIR))
    clear_or_create_folder(str(REVIEW_DIR))

    print("Loading trained Dungeon Archivist model...")
    trained_model = keras.models.load_model(MODEL_PATH)

    embedding_model = models.Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("embedding").output
    )

    # Load vector DB
    client = chromadb.PersistentClient(path="./models/vector_db")
    collection = client.get_collection("dataset-A-embeddings")
    print(f"Vector DB count: {collection.count()}")

    # Gather all images in chaos folder
    all_images = [
        os.path.join(root, f)
        for root, _, files in os.walk(CHAOS_DIR)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    num_images = len(all_images)
    print(f"Found {num_images} images to classify in {CHAOS_DIR}")

    for batch_idx, batch_files in enumerate(batch_iterable(all_images, BATCH_SIZE)):
        # Preprocess batch
        batch_X = np.vstack([preprocess_for_interface(f) for f in batch_files])

        # Compute embeddings
        batch_embeddings = embedding_model.predict(batch_X, batch_size=BATCH_SIZE, verbose=0)

        for i, (file_path, vector) in enumerate(zip(batch_files, batch_embeddings)):
            # Query vector DB
            results = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=TOP_K
            )

            neighbors = results['metadatas'][0]
            distances = results['distances'][0]
            similarities = [1 - d for d in distances]  # Convert distance -> similarity

            label, confidence = weighted_vote_distance(neighbors, similarities)

            # Determine destination folder
            dest_dir = os.path.join(SORTED_DIR, label) if confidence >= CONFIDENCE_THRESHOLD else REVIEW_DIR
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))

            # Debug
            if DEBUG_OUTPUT: # and batch_idx * BATCH_SIZE + i < 10:
                print(f"{os.path.basename(file_path)} -> {dest_dir} (confidence: {confidence:.2f})")
                for n, s in zip(neighbors, similarities):
                    print(f"    {n['label']} (sim={s:.2f})")


def main():
    run_archivist()


if __name__ == "__main__":
    main()
