"""ROC / ROC-AUC figures for local_aic vs global_weighted_aic."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from plot_modality_figures import load_pooled_scores, roc_curve_from_scores


POLICY_ORDER = ("local_aic", "global_weighted_aic")
POLICY_LABELS = {
    "local_aic": "Local AIC",
    "global_weighted_aic": "Global weighted AIC",
}
POLICY_COLORS = {
    "local_aic": "#4C78A8",
    "global_weighted_aic": "#F58518",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot ROC curves / ROC-AUC bars for AIC selection comparison."
    )
    p.add_argument(
        "--root",
        default="results/evaluation_aic_selection_gmm",
        help="Root with local_aic/ and global_weighted_aic/ subfolders.",
    )
    p.add_argument(
        "--input-csv",
        default=None,
        help="Wide comparison CSV (default: <root>/comparison_summary.csv).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Bar chart output (default: <root>/roc_auc_by_selection.png).",
    )
    p.add_argument(
        "--roc-output",
        default=None,
        help="ROC curve output (default: <root>/roc_curves.png).",
    )
    p.add_argument(
        "--metric",
        choices=["macro", "pooled", "both"],
        default="both",
        help="Which ROC-AUC column(s) to plot in the bar chart.",
    )
    p.add_argument(
        "--title",
        default="Authentication ROC-AUC by AIC selection (+GMM)",
        help="Bar chart title.",
    )
    p.add_argument(
        "--roc-title",
        default="ROC Curve by AIC selection (+GMM, pooled scores)",
        help="ROC curve title.",
    )
    p.add_argument(
        "--bars-only",
        action="store_true",
        help="Only write the ROC-AUC bar chart.",
    )
    p.add_argument(
        "--roc-only",
        action="store_true",
        help="Only write the ROC curve plot.",
    )
    p.add_argument("--n-thresholds", type=int, default=501)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_policy_scores(root: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for policy in POLICY_ORDER:
        path = os.path.join(root, policy, "authentication_scores.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Run: python run_aic_selection_ablation.py --include-gmm"
            )
        out[policy] = load_pooled_scores(path)
    return out


def plot_roc_curves(
    policy_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "ROC Curve by AIC selection (+GMM, pooled scores)",
    n_thresholds: int = 501,
    dpi: int = 150,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for ROC plotting.") from exc

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for policy in POLICY_ORDER:
        if policy not in policy_scores:
            continue
        g, i = policy_scores[policy]
        fpr, tpr, auc = roc_curve_from_scores(g, i, n_thresholds=n_thresholds)
        label = f"{POLICY_LABELS.get(policy, policy)} (AUC = {auc:.3f})"
        ax.plot(
            fpr,
            tpr,
            color=POLICY_COLORS.get(policy, "#4C78A8"),
            linewidth=2.0,
            label=label,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR (FAR)")
    ax.set_ylabel("TPR (1 − FRR)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_roc_auc_bars(
    summary_df: pd.DataFrame,
    output_path: str,
    metric: str = "both",
    title: str = "Authentication ROC-AUC by AIC selection",
    dpi: int = 150,
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required for ROC-AUC plotting.") from exc

    required = {"distribution_selection", "roc_auc_macro", "roc_auc_pooled"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary CSV: {sorted(missing)}")

    present = set(summary_df["distribution_selection"].astype(str))
    order = [p for p in POLICY_ORDER if p in present]
    if not order:
        raise ValueError("No local_aic/global_weighted_aic rows found in summary CSV")

    df = summary_df.set_index("distribution_selection").loc[order]
    if "label" in df.columns:
        labels = [
            str(df.loc[p, "label"]) if pd.notna(df.loc[p, "label"]) else POLICY_LABELS[p]
            for p in order
        ]
    else:
        labels = [POLICY_LABELS.get(p, p) for p in order]
    x = np.arange(len(order))
    colors = [POLICY_COLORS.get(p, "#4C78A8") for p in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if metric == "macro":
        vals = pd.to_numeric(df["roc_auc_macro"], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylabel("ROC-AUC (macro)")
    elif metric == "pooled":
        vals = pd.to_numeric(df["roc_auc_pooled"], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylabel("ROC-AUC (pooled)")
    else:
        macro = pd.to_numeric(df["roc_auc_macro"], errors="coerce").to_numpy(dtype=float)
        pooled = pd.to_numeric(df["roc_auc_pooled"], errors="coerce").to_numpy(dtype=float)
        w = 0.35
        b1 = ax.bar(
            x - w / 2,
            macro,
            width=w,
            label="macro",
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.6,
        )
        b2 = ax.bar(
            x + w / 2,
            pooled,
            width=w,
            label="pooled",
            color="#F58518",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
        ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
        ax.set_ylabel("ROC-AUC")
        ax.legend(frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Distribution selection")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    root = args.root
    input_csv = args.input_csv or os.path.join(root, "comparison_summary.csv")
    bar_out = args.output or os.path.join(root, "roc_auc_by_selection.png")
    roc_out = args.roc_output or os.path.join(root, "roc_curves.png")

    if not args.roc_only:
        df = pd.read_csv(input_csv)
        out = plot_roc_auc_bars(
            df,
            bar_out,
            metric=args.metric,
            title=args.title,
            dpi=args.dpi,
        )
        print(f"Saved: {out}")

    if not args.bars_only:
        scores = load_policy_scores(root)
        out = plot_roc_curves(
            scores,
            roc_out,
            title=args.roc_title,
            n_thresholds=args.n_thresholds,
            dpi=args.dpi,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
