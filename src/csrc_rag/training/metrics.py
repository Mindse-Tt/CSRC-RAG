from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    subset_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    return {
        "Micro-F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "HammingLoss": float(hamming_loss(y_true, y_pred)),
        "SubsetAccuracy": subset_accuracy,
    }

