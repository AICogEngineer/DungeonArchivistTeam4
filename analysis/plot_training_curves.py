import json
import matplotlib.pyplot as plt
import os

HISTORY_PATH = "analysis/training_history_dataset-A-embeddings.json"
# HISTORY_PATH_2 = "analysis/training_history_dataset-AB-embeddings.json"

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(
        "training_history.json not found. "
        "Make sure train.py saves training history first."
    )

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = range(1, len(history["loss"]) + 1)

# Loss Curve
plt.figure()
plt.plot(epochs, history["loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("analysis/loss_curve.png")
plt.close()

# Accuracy Curve
plt.figure()
plt.plot(epochs, history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("analysis/accuracy_curve.png")
plt.close()

print("Saved training curves to analysis/")