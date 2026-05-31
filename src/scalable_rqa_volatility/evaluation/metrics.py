"""
Compute standard classification metrics.

This module provides a small helper for evaluating binary classification
predictions using accuracy, F1 score, and ROC-AUC. The returned metrics are
converted to plain Python floats so they can be logged or serialized easily.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }