import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "analysis/run_results.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("run_results.csv not found. Run archivist.py first.")

# Load results
df = pd.read_csv(CSV_PATH)

if len(df) == 0:
    print("No chaos run data found. Skipping operational analysis.")
    exit(0)

print("=== ANALYSIS SUMMARY ===")

# 1. Total images
total_images = len(df)
print(f"Total images processed: {total_images}")

# 2. Sorted vs review
sorted_count = len(df[df["destination"] == "restored_archive"])
review_count = len(df[df["destination"] == "review_pile"])

print(f"Auto-sorted images: {sorted_count} ({sorted_count/total_images:.1%})")
print(f"Sent to review: {review_count} ({review_count/total_images:.1%})")

# 3. Confidence stats
print("\nConfidence Statistics:")
print(df["confidence"].describe())

# 4. Predictions per label
print("\nPredicted label distribution:")
print(df["predicted_label"].value_counts())

# 5. Save plots
plt.figure()
df["confidence"].hist(bins=10)
plt.title("Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Image Count")
plt.savefig("analysis/confidence_distribution.png")
plt.close()

print("\nSaved confidence distribution plot.")
