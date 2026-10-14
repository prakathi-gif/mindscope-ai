"""
Comprehensive Academic Evaluation Engine for Multi-Class Text Classification.
Computes Accuracy, Macro/Weighted F1, Per-Class Metrics, Log-Loss, ECE, Calibration, and ROC/PR curves.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    log_loss, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from config import LABEL_NAMES, NUM_CLASSES

def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """
    Computes Expected Calibration Error (ECE) and calibration reliability curve data.
    ECE measures how closely predicted confidence matches empirical accuracy.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_centers.append((bin_lower + bin_upper) / 2.0)
            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_counts.append(bin_count)

    curve_data = {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": float(ece)
    }
    return float(ece), curve_data

def evaluate_classification_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = LABEL_NAMES,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes complete academic evaluation metrics for multi-class classification.
    """
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted metrics
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp_cls[i])
        }

    # Multiclass Log-Loss (clipped for stability)
    prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    loss = float(log_loss(y_true, prob_clipped, labels=list(range(len(class_names)))))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-15)

    # Calibration ECE
    ece, cal_curve = compute_expected_calibration_error(y_true, y_prob)

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "log_loss": loss,
        "ece": ece,
        "calibration_curve": cal_curve,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "num_samples": len(y_true)
    }

