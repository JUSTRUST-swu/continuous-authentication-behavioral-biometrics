"""Authentication metrics: FAR, FRR, EER, ROC-AUC."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def far_frr_at_threshold(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    threshold: float,
) -> Tuple[float, float]:
    """
    Accept if score >= threshold (higher LL = more genuine-like).

    FAR = accepted impostors / all impostors
    FRR = rejected genuines / all genuines
    """
    g = np.asarray(list(genuine_scores), dtype=float)
    i = np.asarray(list(impostor_scores), dtype=float)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    if len(g) == 0:
        frr = float("nan")
    else:
        frr = float(np.mean(g < threshold))
    if len(i) == 0:
        far = float("nan")
    else:
        far = float(np.mean(i >= threshold))
    return far, frr


def calibrate_genuine_quantile_threshold(
    validation_genuine_scores: Sequence[float],
    quantile: float = 0.05,
) -> float:
    scores = np.asarray(list(validation_genuine_scores), dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("No finite validation genuine scores for threshold calibration")
    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"quantile must be in [0,1], got {q}")
    return float(np.quantile(scores, q))


def roc_auc_score(genuine_scores: Sequence[float], impostor_scores: Sequence[float]) -> float:
    """
    Mann–Whitney U / Wilcoxon form of ROC-AUC.
    Label: genuine=1, impostor=0; higher score preferred for genuine.
    """
    g = np.asarray(list(genuine_scores), dtype=float)
    i = np.asarray(list(impostor_scores), dtype=float)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    if len(g) == 0 or len(i) == 0:
        return float("nan")

    # All pairwise comparisons
    # AUC = P(score_g > score_i) + 0.5 P(equal)
    diff = g[:, None] - i[None, :]
    return float(np.mean((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)))


def compute_eer(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    n_thresholds: int = 501,
) -> Tuple[float, float]:
    """
    Approximate EER by sweeping thresholds between min and max scores.

    Returns (eer, eer_threshold) where FAR≈FRR.
    This EER is a **reporting metric** only — do not reuse as decision threshold.
    """
    g = np.asarray(list(genuine_scores), dtype=float)
    i = np.asarray(list(impostor_scores), dtype=float)
    g = g[np.isfinite(g)]
    i = i[np.isfinite(i)]
    if len(g) == 0 or len(i) == 0:
        return float("nan"), float("nan")

    lo = float(min(np.min(g), np.min(i)))
    hi = float(max(np.max(g), np.max(i)))
    if lo == hi:
        far, frr = far_frr_at_threshold(g, i, lo)
        return float(0.5 * (far + frr)), lo

    thresholds = np.linspace(lo, hi, int(n_thresholds))
    best = None
    for t in thresholds:
        far, frr = far_frr_at_threshold(g, i, float(t))
        if not np.isfinite(far) or not np.isfinite(frr):
            continue
        gap = abs(far - frr)
        eer = 0.5 * (far + frr)
        cand = (gap, eer, float(t))
        if best is None or cand[0] < best[0] or (np.isclose(cand[0], best[0]) and cand[1] < best[1]):
            best = cand
    if best is None:
        return float("nan"), float("nan")
    return float(best[1]), float(best[2])


def compute_binary_metrics(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold)
    auc = roc_auc_score(genuine_scores, impostor_scores)
    eer, eer_t = compute_eer(genuine_scores, impostor_scores)
    return {
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_t,
        "n_genuine": float(np.sum(np.isfinite(np.asarray(list(genuine_scores), dtype=float)))),
        "n_impostor": float(np.sum(np.isfinite(np.asarray(list(impostor_scores), dtype=float)))),
        "decision_threshold": float(threshold),
    }
