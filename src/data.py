import os
import numpy as np
import tensorflow as tf

IMG_SIZE = (32, 32)

def load_and_preprocess_image(image_path):

    # Load a PNG image and convert it to a normalized tensor.
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_dataset(root_dir):

    # finds dataset directory and returns X, y and label_map
    X = []
    y = []
    paths = []  # Store paths so subfolder structure is preserved in metadata

    # Top-level folders = labels
    label_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    print(label_names)

    label_map = {name: idx for idx, name in enumerate(label_names)}

    # Iterate over the initial top-level folders directly in the data set
    for label_name, label_idx in label_map.items():
        label_dir = os.path.join(root_dir, label_name)
        
        # Go over every file found in the various subfolders
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    image_path = os.path.join(root, file)
                    try:
                        image = load_and_preprocess_image(image_path)
                        X.append(image)
                        y.append(label_idx)
                        rel_path = os.path.relpath(image_path, root_dir)
                        paths.append(rel_path)
                    except Exception as e:
                        print(f"Skipping {image_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)

    return X, y, label_map, paths

def preprocess_for_interface(image_path):
    # Loads a single image and adds batch dimentions for prediction

    image = load_and_preprocess_image(image_path)
    image = tf.expand_dims(image, axis=0) # Shape (1, 32, 32, 3)
    return image