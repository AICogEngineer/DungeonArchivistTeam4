# Dungeon Archivist
AI-Powered Asset Restoration Using CNN Embeddings and Vector Search

---

## Project Overview

The Dungeon Archivist is an AI system designed to restore a chaotic archive of unlabeled game assets.  
Given thousands of mixed, randomly named sprite images, the system analyzes visual content and automatically sorts each asset into the correct category (for example: dungeon, monster, item).

The project is completed in two phases:

- Phase 1: Train a Convolutional Neural Network (CNN) on a labeled dataset and store learned image embeddings in a vector database.
- Phase 2: Use vector similarity search to classify and restore previously unseen chaos images, with confidence-based decision making and human review support.

---

## System Architecture

LABELED DATA (Dataset A)
        |
        v
CNN (Feature + Embedding Learning)
        |
        v
Vector Embeddings
        |
        v
ChromaDB (Persistent Vector Store)
        |
        v
----------------------------------
        |
CHAOS DATA (Unlabeled Images)
        |
        v
CNN Embedding Extraction
        |
        v
Vector Similarity Search (KNN)
        |
        v
Confidence Threshold
   |            |
   v            v
Auto-Sorted   Review Pile

---

## Dataset Description

### Dataset A — Labeled Training Data

Source: Dungeon Crawl Stone Soup sprite set

Categories:
- dungeon
- effect
- emissaries
- gui
- item
- misc
- monster
- player

Usage: Used to train the CNN and generate vector embeddings

---

### Dataset B — Chaos Dataset

- Provided by the instructor
- Approximately 4,000 unlabeled 32×32 sprite images
- Random filenames with no folder structure
- Used only during inference (Phase 2)

---

## Training Phase (Phase 1)

### Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of:
- Convolution and max-pooling layers
- A dense embedding layer named "embedding" with 64 dimensions
- A softmax output layer used only during training

---

### Training Outputs

Trained model saved to:
- models/version_model.keras

Vector embeddings stored in:
- models/vector_db/

---

### Key Results

- Approximately 6,000 labeled images processed
- Image embeddings persisted using ChromaDB
- The model learns visual similarity rather than memorizing filenames

---

## Inference and Restoration Phase (Phase 2)

During inference, the system processes the chaos dataset using the following steps:

1. Load the trained CNN
2. Extract embeddings from the embedding layer
3. Query ChromaDB using vector similarity search (KNN)
4. Perform weighted voting across nearest neighbors
5. Apply a confidence threshold:
   - High confidence results are automatically sorted
   - Low confidence results are sent to a review pile

---

### Confidence Logic

- Top-K nearest neighbors are evaluated
- Majority vote determines the predicted label
- Confidence is calculated as the vote ratio
- The confidence threshold prevents overconfident misclassification

---

## Output Structure

After running archivist.py, the output directory structure is:

data/
- restored_archive/
  - dungeon/
  - effect/
  - item/
  - monster/
  - ...
- review_pile/

The restored_archive directory contains confidently classified assets.  
The review_pile directory contains ambiguous assets for human inspection.

---

## How to Run the Project

Train the model by running:
- python train.py

Restore the chaos dataset by running:
- python archivist.py

---

## Results Summary

- Successfully restored a chaotic archive into meaningful categories
- Demonstrated effective use of CNN-based embeddings
- Used a vector database for similarity search
- Applied confidence-based decision making
- Implemented a human-in-the-loop review mechanism

---

## Limitations

- Class imbalance in the training data can affect confidence levels
- Flat category structure with no hierarchical subcategories
- Some visually similar assets are intentionally routed to the review pile
- Command-line interface only, with no graphical user interface

---

## Future Improvements

- Retrain the model using reviewed assets to improve accuracy
- Introduce hierarchical or multi-level categories
- Add a web-based review interface
- Experiment with deeper CNN architectures
- Support multi-label classification

---

## Conclusion

The Dungeon Archivist demonstrates how deep learning embeddings and vector similarity search can be combined to solve real-world data organization problems.

By incorporating confidence-aware automation and human oversight, the system goes beyond simple classification and successfully fulfills all technical and design requirements for automated archive restoration.
