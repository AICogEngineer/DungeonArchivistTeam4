from pathlib import Path
import numpy as np
from tensorflow import keras 

from src.data import preprocess_for_interface
from src.labeling import LABEL_MAP
from src.io_ops import move_to_sorted

# Paths 
MODEL_PATH = Path("models/version_model.keras")
CHAOS_DIR = Path("data/chaos")
SORTED_DIR = Path("data/sorted")

def run_archivist():
    print("Loading trained Dungeon Archivist...")
    model = keras.models.load_model(MODEL_PATH)

    for image_path in CHAOS_DIR.iterdir():
        if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        img = preprocess_for_interface(image_path)
        predictions = model.predict(img, verbose=0)

        predicted_index = np.argmax(predictions)
        category = LABEL_MAP [predicted_index]

        move_to_sorted(image_path, category, SORTED_DIR)

def main():
    run_archivist()

if __name__ == "__main__":
    main()