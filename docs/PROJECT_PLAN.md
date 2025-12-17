# The Dungeon Archivist — Project Plan

## 1. Problem Statement
We have thousands of 32x32 game assets with corrupted filenames and no folder structure.
The goal is to automatically restore organization using visual similarity and AI.

## 2. High-Level Solution
1. Train a CNN on a clean labeled dataset (Dataset A)
2. Extract embeddings from the CNN
3. Store embeddings in a vector database (ChromaDB)
4. Use similarity search to auto-label unknown assets (Dataset B)
5. Reject low-confidence items for manual review

## 3. Datasets
### Dataset A (Labeled)
- Source: Dungeon Crawl 32x32 Tiles
- Classes: Weapon, Environment, Enemy, Consumable, Misc

### Dataset B (Unlabeled)
- Provided by instructor
- Random filenames
- Used only for inference initially

## 4. Technical Stack
- Python 3.x
- TensorFlow / Keras
- ChromaDB
- NumPy
- Pillow / OpenCV

## 5. Project Phases
| Phase | Description |
|------|------------|
| Phase 1 | Train CNN on Dataset A |
| Phase 2 | Generate embeddings + store in ChromaDB |
| Phase 3 | Auto-archive Dataset B |
| Phase 4 | Retrain with expanded dataset |
| Phase 5 | Evaluation & demo |

## 6. Risks & Mitigations
- Risk: Misclassification
  - Mitigation: Similarity voting + review pile
- Risk: Overfitting
  - Mitigation: Small model + validation split
