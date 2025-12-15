# Random-Forest-Classification-for-Your-Car-Rental-Company

"""
Model Evaluation Using Precision, Recall, and F1-Score
------------------------------------------------------
This script compares the performance of two binary classifiers
using NumPy-based calculations of true positives, false positives,
false negatives, precision, recall, and F1-score.
"""

import numpy as np

# --------------------------------------------------
# Step 1: Define ground-truth labels and predictions
# --------------------------------------------------

# True labels for the dataset
real_labels = np.array([True, True, False, True, True])

# Predictions from Model 1
model_1_preds = np.array([True, False, False, False, False])

# Predictions from Model 2
model_2_preds = np.array([True, True, True, True, True])

# --------------------------------------------------
# Step 2: Define a function to compute evaluation metrics
# --------------------------------------------------

def evaluate_model(true_labels, predictions):
    """
    Computes TP, FP, FN, precision, recall, and F1-score.
    """

    tp = ((true_labels == True) & (predictions == True)).sum()
    fp = ((true_labels == False) & (predictions == True)).sum()
    fn = ((true_labels == True) & (predictions == False)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return tp, fp, fn, precision, recall, f1

# --------------------------------------------------
# Step 3: Evaluate both models
# --------------------------------------------------

model_1_results = evaluate_model(real_labels, model_1_preds)
model_2_results = evaluate_model(real_labels, model_2_preds)

# --------------------------------------------------
# Step 4: Display results
# --------------------------------------------------

print("Model 1 Evaluation")
print(f"True Positives: {model_1_results[0]}")
print(f"False Positives: {model_1_results[1]}")
print(f"False Negatives: {model_1_results[2]}")
print(f"Precision: {model_1_results[3]:.2f}")
print(f"Recall: {model_1_results[4]:.2f}")
print(f"F1 Score: {model_1_results[5]:.2f}\n")

print("Model 2 Evaluation")
print(f"True Positives: {model_2_results[0]}")
print(f"False Positives: {model_2_results[1]}")
print(f"False Negatives: {model_2_results[2]}")
print(f"Precision: {model_2_results[3]:.2f}")
print(f"Recall: {model_2_results[4]:.2f}")
print(f"F1 Score: {model_2_results[5]:.2f}")



"""
Model Performance Comparison
----------------------------
This script visualizes and compares the Precision, Recall, and F1-score
of two classification models using a bar chart.
"""

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Step 1: Define evaluation metrics and model scores
# --------------------------------------------------

metrics = ["Precision", "Recall", "F1-score"]

model_1_scores = [model_1_precision, model_1_recall, model_1_f1]
model_2_scores = [model_2_precision, model_2_recall, model_2_f1]

# --------------------------------------------------
# Step 2: Set bar positions
# --------------------------------------------------

x = np.arange(len(metrics))
bar_width = 0.35

# --------------------------------------------------
# Step 3: Create the bar chart
# --------------------------------------------------

plt.figure()
plt.bar(x - bar_width / 2, model_1_scores, width=bar_width, label="Model 1")
plt.bar(x + bar_width / 2, model_2_scores, width=bar_width, label="Model 2")

# --------------------------------------------------
# Step 4: Customize the plot
# --------------------------------------------------

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Comparison of Model Performance")
plt.legend()

# --------------------------------------------------
# Step 5: Display the plot
# --------------------------------------------------

plt.show()
