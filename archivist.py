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
NEW_SORTED_DIR = Path("data/new_restored_archive")
NEW_REVIEW_DIR = Path("data/new_review_pile")
DATASET_C = Path("data/dataset_c")

# Parameters
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.6  # Somewhat loose threshold
DEBUG_OUTPUT = True  # Set False for normal operation
BATCH_SIZE = 32      # For querying multiple images at once


def weighted_vote_distance(neighbors, similarities, temperature=0.05):
    # Apply softmax to similarities
    sims = np.array(similarities)
    sims_exp = np.exp(sims / temperature)   # emphasizes the top ones

    votes = {}
    for neighbor, sim in zip(neighbors, sims_exp):
        label = neighbor['label']
        votes[label] = votes.get(label, 0) + sim

    best_label = max(votes, key=votes.get)
    confidence = votes[best_label] / sum(votes.values())
    return best_label, confidence


def batch_iterable(iterable, batch_size):
    """Yield successive batches from an iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def run_archivist(source_dir, sorted_dir, review_dir, collection_name):
    # Clear folders for sorted and reviewable images
    clear_or_create_folder(sorted_dir)
    clear_or_create_folder(review_dir)

    print("Loading trained Dungeon Archivist model...")
    trained_model = keras.models.load_model(MODEL_PATH)

    embedding_model = models.Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("embedding").output
    )

    # Load vector DB
    client = chromadb.PersistentClient(path="./models/vector_db")
    collection = client.get_collection(collection_name)
    print(f"Vector DB count: {collection.count()}")

    # Gather all images in chaos folder
    all_images = [
        os.path.join(root, f)
        for root, _, files in os.walk(source_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    num_images = len(all_images)
    print(f"Found {num_images} images to classify in {source_dir}")

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
            dest_dir = os.path.join(sorted_dir, label) if confidence >= CONFIDENCE_THRESHOLD else review_dir
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))

            # Debug
            if DEBUG_OUTPUT: # and batch_idx * BATCH_SIZE + i < 10:
                print(f"{os.path.basename(file_path)} -> {dest_dir} (confidence: {confidence:.2f})")
                for n, s in zip(neighbors, similarities):
                    print(f"    {n['label']} (sim={s:.2f})")


def run_archivist_on_b():
    collection_name = "dataset-A-embeddings"
    run_archivist(CHAOS_DIR, str(SORTED_DIR),str(REVIEW_DIR),collection_name)

def run_archivist_on_c():
    collection_name = "dataset-AB-embeddings"
    run_archivist(DATASET_C, str(NEW_SORTED_DIR),str(NEW_REVIEW_DIR),collection_name)

def main():
    run_archivist_on_c()

if __name__ == "__main__":
    main()
