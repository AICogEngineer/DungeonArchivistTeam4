from pathlib import Path
import numpy as np
from tensorflow import keras 
from keras import models
import chromadb
import os
import shutil


from src.data import preprocess_for_interface
from src.io_ops import clear_or_create_folder

# Paths 
MODEL_PATH = Path("models/version_model.keras")
CHAOS_DIR = Path("data/chaos_data")
SORTED_DIR = Path("data/restored_archive")
REVIEW_DIR = Path("data/review_pile")

TOP_K = 5
CONFIDENCE_THRESHOLD = 0.7
DEBUG_OUTPUT = True  # Set False for normal operation

def weighted_vote(neighbors):
    labels = [m['label'] for m in neighbors]
    counts = {l: labels.count(l) for l in set(labels)} 
    best_label = max(counts, key=counts.get)
    confidence = counts[best_label] / len(labels)
    return best_label, confidence

def weighted_vote_distance(neighbors, similarities):
    votes = {}
    for neighbor, sim in zip(neighbors, similarities):
        label = neighbor['label']
        votes[label] = votes.get(label, 0) + sim  # weight by similarity
    
    # Pick label with highest weighted vote
    best_label = max(votes, key=votes.get)
    confidence = votes[best_label] / sum(votes.values())
    return best_label, confidence

def run_archivist():
    # Clear folders for sorted and reviewable images (function accepts str, not Path)
    print("Clearing directories...")
    clear_or_create_folder(str(SORTED_DIR))    
    clear_or_create_folder(str(REVIEW_DIR))
    print("Loading trained Dungeon Archivist...")
    # Load trained model
    trained_model = keras.models.load_model(MODEL_PATH)

    # Create embedding model
    embedding_layer = trained_model.get_layer("embedding")
    embedding_model = models.Model(
        inputs=trained_model.input,
        outputs=embedding_layer.output
    )

    # Load vector db
    client = chromadb.PersistentClient(path="./models/vector_db")
    
    collections = client.list_collections()
    print("Avaible collections:", collections)

    
    collection = client.get_collection("dataset-A-embeddings")
    print("Vector DB count:", collection.count())

    i = 0   # Debug first 10 images
    for root, dirs, files in os.walk(CHAOS_DIR):
        for filename in files:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(root, filename)
            img = preprocess_for_interface(image_path)
            vector = embedding_model.predict(img)[0]
            results = collection.query(
                    query_embeddings=[vector.tolist()],
                    n_results=TOP_K
                )
            neighbors = results['metadatas'][0]
            similarities = results['distances'][0]  # Chroma usually gives distances
            # If distances, convert to similarity:
            similarities = [1 - d for d in similarities]  # cosine similarity: 1 - distance
            label, confidence = weighted_vote_distance(neighbors, similarities)

            if confidence >= CONFIDENCE_THRESHOLD:    
                # best_match_path = neighbors[0]["rel_path"]    # use rel_path if you think you can get more specific
                # best_match_path = label
                # dest_dir = os.path.join(SORTED_DIR, os.path.dirname(best_match_path))
                dest_dir = os.path.join(SORTED_DIR, label)
            else:
                dest_dir = REVIEW_DIR
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


def main():
    run_archivist()

if __name__ == "__main__":
    main()