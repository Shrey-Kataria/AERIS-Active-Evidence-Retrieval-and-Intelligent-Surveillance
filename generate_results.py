"""
AERIS Research Results Generator
=================================
Reads experimental results from ``results.csv`` and produces
publication-quality (600 DPI) figures and summary metrics for
evaluating AERIS against baseline surveillance strategies.

Generated outputs (saved to ``aeris_results/``):
    - Confusion matrices for each method
    - Precision / Recall / F1 bar chart
    - Accuracy comparison bar chart
    - F1-score vs. average cameras queried scatter plot
    - AERIS decision latency histogram

Author : Shrey Kataria
License: MIT
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# AERIS RESEARCH RESULT GENERATOR
# ============================================================

INPUT_FILE = "results.csv"
OUTPUT_FOLDER = "aeris_results"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 1. READ RESULTS
# ============================================================

if not os.path.exists(INPUT_FILE):
    print("ERROR: results.csv was not found.")
    print()
    print("Put generate_results.py in the same folder as results.csv")
    input("Press Enter to close...")
    exit()


data = pd.read_csv(INPUT_FILE)

print()
print("==============================================")
print("        AERIS RESEARCH RESULT GENERATOR")
print("==============================================")
print()

print("Results loaded successfully.")
print("Number of events:", len(data))
print()


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ground_truth",
    "single_camera",
    "all_cameras",
    "random",
    "aeris"
]

for column in required_columns:

    if column not in data.columns:

        print("ERROR: Missing column:", column)
        print()
        print("Required columns are:")
        
        for c in required_columns:
            print("-", c)

        input("Press Enter to close...")
        exit()


# ============================================================
# 3. METHODS
# ============================================================

methods = {
    "Single Camera": "single_camera",
    "All Cameras": "all_cameras",
    "Random Selection": "random",
    "AERIS": "aeris"
}


y_true = data["ground_truth"].astype(int)


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

metrics = []

for method_name, column_name in methods.items():

    y_pred = data[column_name].astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics.append({
        "Method": method_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


metrics_df = pd.DataFrame(metrics)


# ============================================================
# 5. SAVE METRICS
# ============================================================

metrics_file = os.path.join(
    OUTPUT_FOLDER,
    "aeris_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print("Metrics calculated.")
print()

print(metrics_df.to_string(index=False))

print()
print("Metrics saved to:")
print(metrics_file)


# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

print()
print("Generating confusion matrices...")


for method_name, column_name in methods.items():

    y_pred = data[column_name].astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    figure, axis = plt.subplots(
        figsize=(5.5, 5)
    )

    image = axis.imshow(matrix)

    # Write numbers inside cells
    for row in range(2):

        for col in range(2):

            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold"
            )

    axis.set_xlabel(
        "Predicted Class",
        fontsize=11
    )

    axis.set_ylabel(
        "Actual Class",
        fontsize=11
    )

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    axis.set_xticklabels([
        "No Intrusion",
        "Intrusion"
    ])

    axis.set_yticklabels([
        "No Intrusion",
        "Intrusion"
    ])

    axis.set_title(
        method_name + " - Confusion Matrix",
        fontsize=13,
        fontweight="bold"
    )

    figure.colorbar(image)

    figure.tight_layout()

    filename = (
        method_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    filepath = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    figure.savefig(
        filepath,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(figure)

    print("Created:", filename)


# ============================================================
# 7. PERFORMANCE COMPARISON
# ============================================================

print()
print("Generating performance comparison...")


x = np.arange(len(methods))

width = 0.25

figure, axis = plt.subplots(
    figsize=(9, 5.5)
)

precision_values = metrics_df["Precision"]
recall_values = metrics_df["Recall"]
f1_values = metrics_df["F1"]


axis.bar(
    x - width,
    precision_values,
    width,
    label="Precision"
)

axis.bar(
    x,
    recall_values,
    width,
    label="Recall"
)

axis.bar(
    x + width,
    f1_values,
    width,
    label="F1-score"
)


axis.set_ylabel(
    "Score",
    fontsize=11
)

axis.set_xlabel(
    "Method",
    fontsize=11
)

axis.set_title(
    "Performance Comparison",
    fontsize=14,
    fontweight="bold"
)

axis.set_xticks(x)

axis.set_xticklabels(
    list(methods.keys()),
    rotation=15
)

axis.set_ylim(
    0,
    1.05
)

axis.legend()

axis.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

figure.tight_layout()

performance_file = os.path.join(
    OUTPUT_FOLDER,
    "performance_comparison.png"
)

figure.savefig(
    performance_file,
    dpi=600,
    bbox_inches="tight"
)

plt.close(figure)

print("Created: performance_comparison.png")


# ============================================================
# 8. ACCURACY COMPARISON
# ============================================================

figure, axis = plt.subplots(
    figsize=(8, 5)
)

accuracy_values = metrics_df["Accuracy"]

bars = axis.bar(
    list(methods.keys()),
    accuracy_values
)

axis.set_ylabel(
    "Accuracy",
    fontsize=11
)

axis.set_xlabel(
    "Method",
    fontsize=11
)

axis.set_title(
    "Accuracy Comparison",
    fontsize=14,
    fontweight="bold"
)

axis.set_ylim(
    0,
    1.05
)

axis.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


# Add percentage values
for bar, value in zip(
    bars,
    accuracy_values
):

    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        str(round(value * 100, 2)) + "%",
        ha="center",
        fontsize=10
    )


figure.tight_layout()

accuracy_file = os.path.join(
    OUTPUT_FOLDER,
    "accuracy_comparison.png"
)

figure.savefig(
    accuracy_file,
    dpi=600,
    bbox_inches="tight"
)

plt.close(figure)

print("Created: accuracy_comparison.png")


# ============================================================
# 9. CAMERA QUERY ANALYSIS
# ============================================================

if "cameras_queried" in data.columns:

    print()
    print("Generating camera-query analysis...")

    average_cameras = []

    for method_name in methods:

        if method_name == "Single Camera":

            average = 1

        elif method_name == "All Cameras":

            # Number of cameras available in this experiment
            average = 2

        elif method_name == "Random Selection":

            average = data["cameras_queried"].mean()

        else:

            average = data["cameras_queried"].mean()

        average_cameras.append(average)


    f1_values = metrics_df["F1"].values


    figure, axis = plt.subplots(
        figsize=(8, 5.5)
    )


    axis.scatter(
        average_cameras,
        f1_values,
        s=100
    )


    for i, method_name in enumerate(
        methods.keys()
    ):

        axis.annotate(
            method_name,
            (
                average_cameras[i],
                f1_values[i]
            ),
            xytext=(7, 7),
            textcoords="offset points"
        )


    axis.set_xlabel(
        "Average Cameras Queried",
        fontsize=11
    )

    axis.set_ylabel(
        "F1-score",
        fontsize=11
    )

    axis.set_title(
        "Decision Performance vs. Camera Usage",
        fontsize=14,
        fontweight="bold"
    )

    axis.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    axis.set_ylim(
        0,
        1.05
    )

    figure.tight_layout()

    budget_file = os.path.join(
        OUTPUT_FOLDER,
        "f1_vs_camera_usage.png"
    )

    figure.savefig(
        budget_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(figure)

    print("Created: f1_vs_camera_usage.png")


# ============================================================
# 10. AERIS CAMERA-2 QUERY RATE
# ============================================================

if "camera2_requested" in data.columns:

    print()
    print("Generating AERIS camera-query rate...")


    query_values = data[
        "camera2_requested"
    ].astype(int)


    requested = query_values.sum()

    total = len(query_values)

    not_requested = total - requested


    labels = [
        "Camera 2 Not Requested",
        "Camera 2 Requested"
    ]

    values = [
        not_requested,
        requested
    ]


    figure, axis = plt.subplots(
        figsize=(7, 5)
    )


    bars = axis.bar(
        labels,
        values
    )


    axis.set_ylabel(
        "Number of Events",
        fontsize=11
    )

    axis.set_title(
        "AERIS Additional Evidence Requests",
        fontsize=14,
        fontweight="bold"
    )


    for bar, value in zip(
        bars,
        values
    ):

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(total * 0.01, 1),
            str(value),
            ha="center",
            fontsize=10
        )


    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )


    figure.tight_layout()


    query_file = os.path.join(
        OUTPUT_FOLDER,
        "aeris_camera2_requests.png"
    )


    figure.savefig(
        query_file,
        dpi=600,
        bbox_inches="tight"
    )


    plt.close(figure)

    print("Created: aeris_camera2_requests.png")


# ============================================================
# 11. LATENCY COMPARISON
# ============================================================

if "latency_ms" in data.columns:

    print()
    print("Generating latency analysis...")


    average_latency = data[
        "latency_ms"
    ].mean()


    print(
        "Average recorded latency:",
        round(average_latency, 2),
        "ms"
    )


    figure, axis = plt.subplots(
        figsize=(7, 5)
    )


    axis.hist(
        data["latency_ms"],
        bins=20
    )


    axis.set_xlabel(
        "Decision Latency (ms)",
        fontsize=11
    )

    axis.set_ylabel(
        "Number of Events",
        fontsize=11
    )

    axis.set_title(
        "AERIS Decision Latency Distribution",
        fontsize=14,
        fontweight="bold"
    )


    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )


    figure.tight_layout()


    latency_file = os.path.join(
        OUTPUT_FOLDER,
        "aeris_latency_distribution.png"
    )


    figure.savefig(
        latency_file,
        dpi=600,
        bbox_inches="tight"
    )


    plt.close(figure)

    print("Created: aeris_latency_distribution.png")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print()
print("==============================================")
print("           GENERATION COMPLETE")
print("==============================================")
print()

print("All generated files are inside:")

print(
    os.path.abspath(OUTPUT_FOLDER)
)

print()

print("Generated:")
print("1. aeris_metrics.csv")
print("2. Single-camera confusion matrix")
print("3. All-camera confusion matrix")
print("4. Random-selection confusion matrix")
print("5. AERIS confusion matrix")
print("6. Performance comparison")
print("7. Accuracy comparison")

if "cameras_queried" in data.columns:
    print("8. F1 vs camera usage")

if "camera2_requested" in data.columns:
    print("9. AERIS camera-query analysis")

if "latency_ms" in data.columns:
    print("10. AERIS latency distribution")

print()
print("These figures are saved at 600 DPI.")
print("They can be used as the basis for Springer paper figures.")

print()

input("Press Enter to close...")