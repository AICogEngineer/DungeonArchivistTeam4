import os
import numpy as np
from PIL import Image

from src.io_ops import clear_or_create_folder

IMG_SIZE = (32, 32)
HIGH_LEVEL_FOLDERS_ONLY = True # If True, only top-level folders are used as labels
DEBUG_SAVE_NORMALIZED = True  # <-- toggle debug save
DEBUG_SAVE_DIR = "data/debug_normalized"  # folder to save normalized images

def normalize_background(image, mode="random", constant_color=(0,0,0)):
    image = image.astype(np.float32)

    rgb = image[..., :3]
    alpha = image[..., 3:4] / 255.0

    if mode == "random":
        bg_color = np.random.randint(0, 256, size=(1, 1, 3)).astype(np.float32)
    else:
        bg_color = np.array(constant_color, dtype=np.float32).reshape(1, 1, 3)

    bg = np.ones_like(rgb) * bg_color
    out = rgb * alpha + bg * (1 - alpha)

    return out

def load_and_preprocess_image(image_path, bg_mode="random"):
    # Load image with alpha
    image = Image.open(image_path).convert("RGBA")

    # Resize (NEAREST is usually best for pixel art)
    image = image.resize(IMG_SIZE, Image.NEAREST)

    # Convert to numpy
    image = np.array(image)

    # Normalize background (removes transparency)
    image = normalize_background(image, mode=bg_mode)

    # Normalize to [0, 1]
    image = image / 255.0

    return image.astype(np.float32)

def save_debug_image(image_array, save_path):
    # Convert back to 0-255 RGB for saving
    img_to_save = (image_array * 255).astype(np.uint8)
    Image.fromarray(img_to_save).save(save_path)

def load_dataset(root_dir):
    X, y, paths = [], [], []
    label_map = {}

    if DEBUG_SAVE_NORMALIZED:
        clear_or_create_folder(DEBUG_SAVE_DIR)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(".png"):
                continue

            image_path = os.path.join(root, file)

            # Determine label based on HIGH_LEVEL_FOLDERS_ONLY
            rel_dir = os.path.relpath(root, root_dir)

            if HIGH_LEVEL_FOLDERS_ONLY:
                label = rel_dir.split(os.sep)[0]  # Only top-level folder
            else:
                label = rel_dir  # Full relative path including subfolders

            if label not in label_map:
                label_map[label] = len(label_map)

            try:
                image = load_and_preprocess_image(image_path)
                # Save debug image if enabled
                if DEBUG_SAVE_NORMALIZED:
                    # Build save path preserving folder structure
                    rel_save_path = os.path.relpath(image_path, root_dir)
                    save_path = os.path.join(DEBUG_SAVE_DIR, rel_save_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_debug_image(image, save_path)
                X.append(image)
                y.append(label_map[label])
                paths.append(os.path.relpath(image_path, root_dir))
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

    return np.array(X), np.array(y), label_map, paths

def preprocess_for_interface(image_path):
    # Loads a single image and adds batch dimentions for prediction

    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0) # Shape (1, 32, 32, 3)
    return image