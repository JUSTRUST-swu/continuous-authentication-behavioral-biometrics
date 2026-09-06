"""
Modality comparison figures for authentication evaluation.

Outputs (under --output-dir):
  - roc_curves.png              ROC: FPR(FAR) vs TPR(1-FRR), AUC in legend
  - far_frr_vs_threshold.png    FAR & FRR vs threshold with EER marker (3 panels)
  - far_frr_grouped_bars.png    Grouped FAR/FRR error-rate (%) bars
  - auc_bars.png                ROC-AUC bars (higher is better)
  - eer_bars.png                EER bars (lower is better)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from auth_metrics import compute_eer, far_frr_at_threshold, roc_auc_score

MODALITY_ORDER = ("keyboard", "mouse", "all")
MODALITY_LABELS = {
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "all": "Keyboard + Mouse",
}
MODALITY_COLORS = {
    "keyboard": "#4C78A8",
    "mouse": "#F58518",
    "all": "#54A24B",
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot modality authentication metric figures.")
    p.add_argument(
        "--root",
        default="results/evaluation_modality",
        help="Root with keyboard/mouse/all subfolders and comparison_summary.csv.",
    )
    p.add_argument(
        "--output-dir",
        default="results/evaluation_modality/figures",
        help="Directory for PNG outputs.",
    )
    p.add_argument(
        "--aggregation",
        choices=["macro", "pooled"],
        default="pooled",
        help=(
            "Aggregation for bar charts (default: pooled). "
            "ROC / FAR-FRR-vs-threshold curves always use pooled scores; "
            "default pooled keeps bar AUC/EER aligned with those curves."
        ),
    )
    p.add_argument("--n-thresholds", type=int, default=501)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc
    return plt


def _save(fig, path: str, dpi: int) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt = _ensure_matplotlib()
    plt.close(fig)
    return path


def _style_ax(ax) -> None:
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


def load_modality_scores(root: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for modality in MODALITY_ORDER:
        path = os.path.join(root, modality, "authentication_scores.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Run: python run_modality_ablation.py"
            )
        out[modality] = load_pooled_scores(path)
    return out


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

    Does **not** inject synthetic (0,0)/(1,1) endpoints — only observed
    operating points from the threshold sweep (same path as FAR/FRR plots).
    Legend AUC is Mann–Whitney on the pooled scores (threshold-free).
    """
    thresholds, fars, frrs = sweep_far_frr(genuine, impostor, n_thresholds=n_thresholds)
    order = np.argsort(thresholds)
    fpr = np.asarray(fars, dtype=float)[order]
    tpr = 1.0 - np.asarray(frrs, dtype=float)[order]
    auc = roc_auc_score(genuine, impostor)
    return fpr, tpr, auc


def plot_roc_curves(
    modality_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    n_thresholds: int = 501,
    dpi: int = 150,
) -> str:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for modality in MODALITY_ORDER:
        g, i = modality_scores[modality]
        fpr, tpr, auc = roc_curve_from_scores(g, i, n_thresholds=n_thresholds)
        label = f"{MODALITY_LABELS[modality]} (AUC = {auc:.3f})"
        ax.plot(
            fpr,
            tpr,
            color=MODALITY_COLORS[modality],
            linewidth=2.0,
            label=label,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR (FAR)")
    ax.set_ylabel("TPR (1 − FRR)")
    ax.set_title("ROC Curve by modality (pooled scores)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    _style_ax(ax)
    fig.tight_layout()
    return _save(fig, output_path, dpi)


def plot_far_frr_vs_threshold(
    modality_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    n_thresholds: int = 501,
    dpi: int = 150,
) -> str:
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, modality in zip(axes, MODALITY_ORDER):
        g, i = modality_scores[modality]
        thresholds, fars, frrs = sweep_far_frr(g, i, n_thresholds=n_thresholds)
        eer, eer_t = compute_eer(g, i, n_thresholds=n_thresholds)
        ax.plot(thresholds, fars, color="#E45756", linewidth=1.8, label="FAR")
        ax.plot(thresholds, frrs, color="#4C78A8", linewidth=1.8, label="FRR")
        if np.isfinite(eer) and np.isfinite(eer_t):
            ax.scatter(
                [eer_t],
                [eer],
                color="black",
                s=36,
                zorder=5,
                label=f"EER = {eer:.3f}",
            )
            ax.axvline(eer_t, color="gray", linestyle="--", linewidth=0.9, alpha=0.8)
            ax.axhline(eer, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.set_title(MODALITY_LABELS[modality])
        ax.set_xlabel("Threshold (log-likelihood)")
        ax.set_ylim(0.0, 1.0)
        _style_ax(ax)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[0].set_ylabel("Error rate")
    fig.suptitle(
        "FAR / FRR vs threshold (EER at intersection; pooled scores)", y=1.02
    )
    fig.tight_layout()
    return _save(fig, output_path, dpi)


def plot_far_frr_grouped_bars(
    summary_df: pd.DataFrame,
    output_path: str,
    aggregation: str = "macro",
    dpi: int = 150,
) -> str:
    plt = _ensure_matplotlib()
    far_col = f"far_{aggregation}"
    frr_col = f"frr_{aggregation}"
    for col in (far_col, frr_col, "modality"):
        if col not in summary_df.columns:
            raise ValueError(f"Missing column {col} in comparison summary")

    order = [m for m in MODALITY_ORDER if m in set(summary_df["modality"].astype(str))]
    df = summary_df.set_index("modality").loc[order]
    labels = [MODALITY_LABELS[m] for m in order]
    far_pct = pd.to_numeric(df[far_col], errors="coerce").to_numpy(dtype=float) * 100.0
    frr_pct = pd.to_numeric(df[frr_col], errors="coerce").to_numpy(dtype=float) * 100.0

    x = np.arange(len(order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar(
        x - w / 2,
        far_pct,
        width=w,
        label="FAR",
        color="#E45756",
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + w / 2,
        frr_pct,
        width=w,
        label="FRR",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error rate (%)")
    ax.set_xlabel("Modality")
    ax.set_title(f"FAR / FRR by modality ({aggregation})")
    ax.set_ylim(0.0, max(100.0, float(np.nanmax([far_pct.max(), frr_pct.max()])) * 1.15))
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    return _save(fig, output_path, dpi)


def plot_single_metric_bars(
    summary_df: pd.DataFrame,
    value_col: str,
    output_path: str,
    *,
    title: str,
    ylabel: str,
    color: str,
    ylim: Tuple[float, float],
    higher_is_better_note: str,
    dpi: int = 150,
) -> str:
    plt = _ensure_matplotlib()
    if "modality" not in summary_df.columns or value_col not in summary_df.columns:
        raise ValueError(f"Need modality and {value_col} in summary CSV")

    order = [m for m in MODALITY_ORDER if m in set(summary_df["modality"].astype(str))]
    df = summary_df.set_index("modality").loc[order]
    labels = [MODALITY_LABELS[m] for m in order]
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(x, vals, width=0.55, color=color, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Modality")
    ax.set_title(f"{title}\n({higher_is_better_note})")
    _style_ax(ax)
    fig.tight_layout()
    return _save(fig, output_path, dpi)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.root, "comparison_summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"Missing {summary_path}. Run: python run_modality_ablation.py"
        )
    summary_df = pd.read_csv(summary_path)
    modality_scores = load_modality_scores(args.root)
    if args.aggregation != "pooled":
        print(
            f"Note: bar charts use aggregation={args.aggregation!r}; "
            "ROC and FAR/FRR-vs-threshold curves always use pooled scores."
        )

    paths: List[str] = []
    paths.append(
        plot_roc_curves(
            modality_scores,
            os.path.join(args.output_dir, "roc_curves.png"),
            n_thresholds=args.n_thresholds,
            dpi=args.dpi,
        )
    )
    paths.append(
        plot_far_frr_vs_threshold(
            modality_scores,
            os.path.join(args.output_dir, "far_frr_vs_threshold.png"),
            n_thresholds=args.n_thresholds,
            dpi=args.dpi,
        )
    )
    paths.append(
        plot_far_frr_grouped_bars(
            summary_df,
            os.path.join(args.output_dir, "far_frr_grouped_bars.png"),
            aggregation=args.aggregation,
            dpi=args.dpi,
        )
    )
    paths.append(
        plot_single_metric_bars(
            summary_df,
            f"roc_auc_{args.aggregation}",
            os.path.join(args.output_dir, "auc_bars.png"),
            title="ROC-AUC by modality",
            ylabel=f"ROC-AUC ({args.aggregation})",
            color="#4C78A8",
            ylim=(0.0, 1.0),
            higher_is_better_note="higher is better",
            dpi=args.dpi,
        )
    )
    paths.append(
        plot_single_metric_bars(
            summary_df,
            f"eer_{args.aggregation}",
            os.path.join(args.output_dir, "eer_bars.png"),
            title="EER by modality",
            ylabel=f"EER ({args.aggregation})",
            color="#E45756",
            ylim=(0.0, 1.0),
            higher_is_better_note="lower is better",
            dpi=args.dpi,
        )
    )

    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
