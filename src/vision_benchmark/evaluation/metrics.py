"""Image-level metrics shared by every model: one defect score per image, label 1 = defective."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_threshold(labels, scores):
    """Score threshold with the highest F1. Pick it on val only, then reuse it on test."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # last point has no threshold


def image_metrics(labels, scores, threshold):
    """AUROC and AP (no threshold needed) plus accuracy, precision, recall, F1 and counts at the threshold."""
    labels = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"auroc": roc_auc_score(labels, scores), "ap": average_precision_score(labels, scores),
            "threshold": threshold, "accuracy": (tp + tn) / len(labels), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
