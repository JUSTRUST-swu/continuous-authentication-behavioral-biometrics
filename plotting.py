"""Shared helpers for authentication ROC / score plots."""

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from auth_metrics import far_frr_at_threshold, roc_auc_score


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc
    return plt


def save_figure(fig, path: str, dpi: int) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt = ensure_matplotlib()
    plt.close(fig)
    return path


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_pooled_scores(scores_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(scores_csv)
    if "label" not in df.columns or "score_ll" not in df.columns:
        raise ValueError(f"Expected label/score_ll columns in {scores_csv}")
    g = pd.to_numeric(df.loc[df["label"] == "genuine", "score_ll"], errors="coerce")
    i = pd.to_numeric(df.loc[df["label"] == "impostor", "score_ll"], errors="coerce")
    g = g.to_numpy(dtype=float)
    i = i.to_numpy(dtype=float)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    if len(g) == 0 or len(i) == 0:
        raise ValueError(f"No finite genuine/impostor scores in {scores_csv}")
    return g, i


def sweep_far_frr(
    genuine: Sequence[float],
    impostor: Sequence[float],
    n_thresholds: int = 501,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(list(genuine), dtype=float)
    i = np.asarray(list(impostor), dtype=float)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    lo = float(min(np.min(g), np.min(i)))
    hi = float(max(np.max(g), np.max(i)))
    if lo == hi:
        thresholds = np.asarray([lo], dtype=float)
    else:
        thresholds = np.linspace(lo, hi, int(n_thresholds))
    fars = np.empty(len(thresholds), dtype=float)
    frrs = np.empty(len(thresholds), dtype=float)
    for idx, t in enumerate(thresholds):
        far, frr = far_frr_at_threshold(g, i, float(t))
        fars[idx] = far
        frrs[idx] = frr
    return thresholds, fars, frrs


def roc_curve_from_scores(
    genuine: Sequence[float],
    impostor: Sequence[float],
    n_thresholds: int = 501,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return FPR(=FAR), TPR(=1-FRR) along increasing threshold, and ROC-AUC.

    Does not inject synthetic (0,0)/(1,1) endpoints — only observed operating
    points from the threshold sweep. Legend AUC is Mann–Whitney on pooled scores.
    """
    thresholds, fars, frrs = sweep_far_frr(genuine, impostor, n_thresholds=n_thresholds)
    order = np.argsort(thresholds)
    fpr = np.asarray(fars, dtype=float)[order]
    tpr = 1.0 - np.asarray(frrs, dtype=float)[order]
    auc = roc_auc_score(genuine, impostor)
    return fpr, tpr, auc
