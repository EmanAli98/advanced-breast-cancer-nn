from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import REPORT_DIR


def find_best_threshold(y_true, y_prob):
    """Choose the threshold with the best F1 score."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


def evaluate_and_plot(y_true, y_prob, threshold=0.5, prefix="test"):
    """Evaluate predictions and save confusion matrix, ROC, and PR curves."""
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, target_names=["malignant", "benign"], output_dict=True)
    auc_score = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({prefix})")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{prefix}_roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix})")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{prefix}_precision_recall_curve.png", dpi=160)
    plt.close()

    return {
        "threshold": float(threshold),
        "roc_auc": float(auc_score),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
