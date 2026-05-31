"""
Save, load, and plot model prediction outputs.

This module defines a prediction bundle format and helper functions for storing
classification predictions as compressed NumPy files. It also provides plotting
utilities for ROC curves, confusion matrices, and combined ROC comparisons
across multiple models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


@dataclass(frozen=True)
class PredictionBundle:
    name: str
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_predictions_npz(path: Path, name: str, y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        name=np.asarray([name], dtype=object),
        y_true=np.asarray(y_true, dtype=int),
        y_score=np.asarray(y_score, dtype=float),
        y_pred=np.asarray(y_pred, dtype=int),
    )


def load_predictions_npz(path: Path) -> PredictionBundle:
    z = np.load(Path(path), allow_pickle=True)
    name = str(z["name"][0]) if "name" in z else Path(path).stem
    return PredictionBundle(
        name=name,
        y_true=np.asarray(z["y_true"], dtype=int),
        y_score=np.asarray(z["y_score"], dtype=float),
        y_pred=np.asarray(z["y_pred"], dtype=int),
    )


def plot_roc(bundle: PredictionBundle, out_path: Path, title: str | None = None) -> None:
    fpr, tpr, _ = roc_curve(bundle.y_true, bundle.y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC: {bundle.name} (AUC={roc_auc:.3f})")
    fig.tight_layout()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(bundle: PredictionBundle, out_path: Path, title: str | None = None) -> None:
    cm = confusion_matrix(bundle.y_true, bundle.y_pred, labels=[0, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or f"Confusion Matrix: {bundle.name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_combined_roc(bundles: Iterable[PredictionBundle], out_path: Path, title: str = "ROC Curves") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for b in bundles:
        fpr, tpr, _ = roc_curve(b.y_true, b.y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{b.name} (AUC={roc_auc:.3f})")

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)