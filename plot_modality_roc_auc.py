"""Bar chart of ROC-AUC for keyboard / mouse / all modalities."""

from __future__ import annotations

import argparse
import os

import pandas as pd

MODALITY_ORDER = ("keyboard", "mouse", "all")
MODALITY_LABELS = {
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "all": "Keyboard + Mouse",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot ROC-AUC bar chart from modality comparison_summary.csv."
    )
    p.add_argument(
        "--input-csv",
        default="results/evaluation_modality/comparison_summary.csv",
        help="Wide comparison CSV from run_modality_ablation.py.",
    )
    p.add_argument(
        "--output",
        default="results/evaluation_modality/roc_auc_by_modality.png",
        help="Output image path.",
    )
    p.add_argument(
        "--metric",
        choices=["macro", "pooled", "both"],
        default="both",
        help="Which ROC-AUC column(s) to plot (default: both as grouped bars).",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def plot_roc_auc_bars(
    summary_df: pd.DataFrame,
    output_path: str,
    metric: str = "both",
    dpi: int = 150,
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required for ROC-AUC plotting.") from exc

    required = {"modality", "roc_auc_macro", "roc_auc_pooled"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary CSV: {sorted(missing)}")

    order = [m for m in MODALITY_ORDER if m in set(summary_df["modality"].astype(str))]
    if not order:
        raise ValueError("No keyboard/mouse/all rows found in summary CSV")

    df = summary_df.set_index("modality").loc[order]
    labels = [MODALITY_LABELS.get(m, m) for m in order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if metric == "macro":
        vals = pd.to_numeric(df["roc_auc_macro"], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, vals, width=0.55, color="#4C78A8", edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylabel("ROC-AUC (macro)")
    elif metric == "pooled":
        vals = pd.to_numeric(df["roc_auc_pooled"], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, vals, width=0.55, color="#F58518", edgecolor="black", linewidth=0.6)
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
    ax.set_xlabel("Modality")
    ax.set_title("Authentication ROC-AUC by modality")
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
    df = pd.read_csv(args.input_csv)
    out = plot_roc_auc_bars(df, args.output, metric=args.metric, dpi=args.dpi)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
