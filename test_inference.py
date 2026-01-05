import os 
import numpy as np
import tensorflow as tf
from src.data import load_and_preprocess_image
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity


TEST_DIR = "./data/test"
MODEL_PATH = "./models/version_model.keras"
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")

def load_test_dataset(test_dir):
    images = []
    paths = []

    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                path = os.path.join(root, file)
                img = load_and_preprocess_image(path)
                images.append(img)
                paths.append(path)

    return np.array(images), paths

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

print("Loading test data...")
X_test, test_paths = load_test_dataset(TEST_DIR)
print(f"Loaded {len(X_test)} test images")

print("Running inference...")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)

print("Generating embeddings...")
embeddings = embedding_model.predict(X_test)

print("Embedding shape:", embeddings.shape)

similarity = cosine_similarity(embeddings)

query_idx = 0 
top_matches = np.argsort(similarity[query_idx])[-6:-1]

print("Query image:", test_paths[query_idx])
print("Most similar images:")
for idx in top_matches:
    print(test_paths[idx])