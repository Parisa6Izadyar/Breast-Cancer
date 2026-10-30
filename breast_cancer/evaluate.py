from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require aligned binary labels and positive-class probabilities")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(y), "dead_prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "dead_precision": float(precision_score(y, predicted, zero_division=0)),
        "dead_recall": float(recall_score(y, predicted, zero_division=0)),
        "dead_f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix": {"true_alive": int(tn), "false_dead": int(fp), "false_alive": int(fn), "true_dead": int(tp)},
    }


def plot_results(results, selected, output: Path):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "breast-status-benchmark"})
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = [name.replace("_", " ") for name in names]
    colors = ["#2166ac" if n == selected else "#a6bddb" for n in names]
    for axis, key, title in zip(axes, ["average_precision", "balanced_accuracy"], ["Test average precision (Dead = 1)", "Test balanced accuracy"]):
        values = [results[n][key] for n in names]
        axis.barh(labels, values, color=colors)
        axis.set(xlim=(0, 1), title=title)
        axis.bar_label(axis.containers[0], fmt="%.3f", padding=3)
    fig.suptitle("Held-out vital-status classification; model selected by training CV")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"model_comparison.{suffix}", metadata={"Date": None} if suffix == "svg" else None, dpi=150)
    plt.close(fig)
    c = results[selected]["confusion_matrix"]
    values = np.array([[c["true_alive"], c["false_dead"]], [c["false_alive"], c["true_dead"]]])
    fig, axis = plt.subplots(figsize=(5, 4), layout="constrained")
    axis.imshow(values, cmap="Blues")
    for (i, j), value in np.ndenumerate(values):
        axis.text(j, i, str(value), ha="center", va="center", color="white" if value > values.max()/2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"], xlabel="Predicted status", ylabel="Recorded status", title=f"{selected.replace('_', ' ')}; threshold 0.5")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"selected_confusion.{suffix}", metadata={"Date": None} if suffix == "svg" else None, dpi=150)
    plt.close(fig)
