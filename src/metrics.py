"""score every model the same way so the results can be compared"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.data import LABEL_ORDER

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
INK, MUTED = "#0b0b0b", "#898781"

def compute_metrics(y_true, y_pred):
    """works out accuracy plus the macro and weighted scores for one set of guesses"""
    labels = list(range(len(LABEL_ORDER)))
    # labels is needed so that a slice missing a class is not averaged over fewer classes
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "per_class_report": classification_report(
            y_true, y_pred, labels=labels, target_names=LABEL_ORDER,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }

def format_metrics_report(metrics, model_name):
    """print one model's scores as plain text"""
    lines = [f"Model: {model_name}",
            f"  Macro F1        : {metrics['macro_f1']:.4f}",
            f"  Macro precision : {metrics['macro_precision']:.4f}",
            f"  Macro recall    : {metrics['macro_recall']:.4f}",
            f"  Weighted F1     : {metrics['weighted_f1']:.4f}",
            f"  Accuracy        : {metrics['accuracy']:.4f}", "", "Per class:"]
    for label in LABEL_ORDER:
        r = metrics["per_class_report"][label]
        lines.append(f"  {label:10s} precision={r['precision']:.3f}  "
                     f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  "
                     f"support={int(r['support'])}")
    return "\n".join(lines)

def plot_confusion_matrix(cm, model_name, normalise=False, save=True):
    """draws the confusion matrix normalise it when the classes differ in size"""
    from matplotlib.colors import LinearSegmentedColormap

    data = cm.astype(float)
    if normalise:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data, totals, out=np.zeros_like(data), where=totals > 0)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    im = ax.imshow(data, cmap=LinearSegmentedColormap.from_list("ramp", BLUE_RAMP))
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER, fontsize=9, color=MUTED)
    ax.set_yticklabels(LABEL_ORDER, fontsize=9, color=MUTED)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("True", color=INK)
    ax.set_title(model_name, fontsize=11, loc="left", color=INK)

    hi = data.max() or 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{data[i, j]:.2f}" if normalise else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="#ffffff" if data[i, j] > hi * 0.6 else INK)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        safe = model_name.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe}.png", dpi=150,
                    bbox_inches="tight")
    return fig

def save_metrics(metrics, model_name, split):
    """save one model's scores as json one file per model and split"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(" ", "_").replace("/", "_")
    payload = {"model_name": model_name, "split": split,
               **{k: metrics[k] for k in ("accuracy", "macro_f1", "macro_precision",
                                          "macro_recall", "weighted_f1")},
               "per_class_report": metrics["per_class_report"],
               "confusion_matrix": metrics["confusion_matrix"].tolist(),
               "label_order": LABEL_ORDER}
    out = RESULTS_DIR / f"{safe}_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out

def comparison_table():
    """builds the comparison table straight from the saved test files"""
    import pandas as pd

    rows = []
    for path in sorted(RESULTS_DIR.glob("*_test.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows.append({"model": d["model_name"],
                     **{k: round(d[k], 4) for k in
                        ("macro_f1", "weighted_f1", "macro_precision",
                         "macro_recall", "accuracy")}})
    df = pd.DataFrame(rows)
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True) if len(df) else df
