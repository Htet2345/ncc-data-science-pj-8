"""Diagnostic charts for the synthetic exercise."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def chart(title, subtitle, left=0.15):
    figure = Figure(figsize=(10, 6), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    figure.subplots_adjust(left=left, right=0.94, bottom=0.23, top=0.78)
    figure.text(0.1, 0.94, title, fontsize=19, weight="bold", color="#15283F")
    figure.text(0.1, 0.88, subtitle, fontsize=10, color="#526275")
    figure.text(0.1, 0.045, "DATA SCIENCE PJ8 | Synthetic records and labels - classroom demonstration", fontsize=9, color="#526275")
    return figure, axes


def save_plots(matrix, importances, output):
    matrix = np.asarray(matrix)
    figure, axes = chart("Synthetic loan-label confusion matrix", "Held-out samples only; rows are actual labels, columns are predictions")
    axes.imshow(matrix, cmap="Blues", vmin=0)
    axes.set_xticks([0, 1], ["0 - Not approved", "1 - Approved"])
    axes.set_yticks([0, 1], ["0 - Not approved", "1 - Approved"])
    axes.set(xlabel="Predicted synthetic label", ylabel="Actual synthetic label")
    for i in range(2):
        for j in range(2):
            axes.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=22, color="white" if matrix[i, j] > matrix.max() / 2 else "#15283F")
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    figure.clear()

    figure, axes = chart("Decision-tree feature importances", "Training-fitted impurity importances; these do not establish causation", left=0.36)
    rows = importances.sort_values("importance", ascending=True)
    labels = rows.feature.str.replace("numeric__", "", regex=False).str.replace("category__", "", regex=False)
    bars = axes.barh(labels, rows.importance, color="#2457A7")
    axes.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes.set(xlabel="Impurity-based importance", xlim=(0, max(0.1, rows.importance.max()) * 1.25))
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="x", color="#E5EAF0")
    axes.set_axisbelow(True)
    figure.savefig(output / "feature_importances.png", dpi=160)
    figure.clear()
