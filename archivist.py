from pathlib import Path
import os
import shutil
import math
import numpy as np
from tensorflow import keras
from keras import models
import chromadb
from datetime import datetime
import numpy as np
from tensorflow import keras 
from keras import models
import chromadb
import shutil
import csv
import os

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
    
    # Analysis output seteup
    os.makedirs("analysis", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"analysis/run_results_{run_id}.csv"

    results_file = open("analysis/run_results.csv", "w", newline="")
    writer = csv.writer(results_file)
    writer.writerow([
        "filename",
        "predicted_label",
        "confidence",
        "destination",
        "top_k_labels",
        "top_k_similarities"
    ])
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

    if collection.count() == 0:
        results_file.close()
        raise RuntimeError(
            "Vector DB is empty. Run train.py before running archivist.py."
        )

    i = 0   # Debug first 10 images
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(root, filename)
            img = preprocess_for_interface(image_path)
            vector = embedding_model.predict(img, verbose=0)[0]
            results = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=TOP_K
            )
            if i <= 10:
                print(results)
            neighbors = results['metadatas'][0]
            similarities = results['distances'][0]  # Chroma usually gives distances
            # Chroma returns cosine distance:
            #   0.0 = identical vectors
            #   1.0 = maximally different
            # Convert to similarity so higher = more similar
            similarities = [1.0 - d for d in similarities]  # cosine similarity: 1 - distance
            label, confidence = weighted_vote_distance(neighbors, similarities)

            destination = ( 
                sorted_dir
                if confidence >= CONFIDENCE_THRESHOLD
                else review_dir
            )

            writer.writerow([
                filename, 
                label,
                round(confidence, 4),
                destination,
                [n["label"] for n in neighbors],
                [round(s, 4) for s in similarities]

            ])

            if confidence >= CONFIDENCE_THRESHOLD:    
                # best_match_path = neighbors[0]["rel_path"]    # use rel_path if you think you can get more specific
                # best_match_path = label
                # dest_dir = os.path.join(SORTED_DIR, os.path.dirname(best_match_path))
                dest_dir = os.path.join(sorted_dir, label)
            else:
                dest_dir = review_dir
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy2(image_path, os.path.join(dest_dir, filename))
            
            # Debug output
            if DEBUG_OUTPUT and (i < 10):
                print(f"{filename} -> {dest_dir} (confidence: {confidence:.2f})")
                print("  Top neighbors:")
                for n, s in zip(neighbors, similarities):
                    print(f"    {n['label']} (sim={s:.2f})")
            i += 1

    results_file.close()
    print("Analysis results saved to analysis/run_results.csv")

def run_archivist_on_b():
    collection_name = "dataset-A-embeddings"
    run_archivist(CHAOS_DIR, str(SORTED_DIR),str(REVIEW_DIR),collection_name)

def run_archivist_on_c():
    collection_name = "dataset-AB-embeddings"
    run_archivist(DATASET_C, str(NEW_SORTED_DIR),str(NEW_REVIEW_DIR),collection_name)

def main():
    run_archivist_on_c()
    # run_archivist_on_b()

if __name__ == "__main__":
    main()
