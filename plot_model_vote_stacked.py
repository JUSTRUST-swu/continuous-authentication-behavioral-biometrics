"""100% stacked bar chart of majority-vote model shares per feature."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

FEATURE_ORDER = (
    "dwell_mean",
    "dwell_std",
    "flight_mean",
    "flight_std",
    "velocity_mean",
    "velocity_std",
)

FEATURE_LABELS = {
    "dwell_mean": "Dwell mean",
    "dwell_std": "Dwell std",
    "flight_mean": "Flight mean",
    "flight_std": "Flight std",
    "velocity_mean": "Velocity mean",
    "velocity_std": "Velocity std",
}

# Stable order; GMM first when present so the dominant opt-in family is readable.
MODEL_ORDER = (
    "GMM",
    "Gaussian",
    "Log-normal",
    "Gamma",
    "Weibull",
    "Log-logistic",
    "Student-t",
)

# Distinct, print-friendly palette (not purple-gradient default).
MODEL_COLORS = {
    "GMM": "#1b9e77",
    "Gaussian": "#d95f02",
    "Log-normal": "#7570b3",
    "Gamma": "#e7298a",
    "Weibull": "#66a61e",
    "Log-logistic": "#e6ab02",
    "Student-t": "#a6761d",
}

# Dark fills → white text; light fills → dark text.
_LIGHT_FILL = {"Weibull", "Log-logistic", "Student-t", "Log-normal"}

# Hide in-bar labels below this share (user request: <15%).
_MIN_LABEL_SHARE = 0.15


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot 100% stacked bars from model_fit_aggregated_vote_counts.csv."
    )
    p.add_argument(
        "--input-csv",
        default="results/main_kmt_gmm/tables/model_fit_aggregated_vote_counts.csv",
        help="Vote-count CSV from main.py aggregation.",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output PNG path (default: <input-dir>/../plots/model_vote_stacked_<criterion>.png).",
    )
    p.add_argument(
        "--criterion",
        choices=["majority_vote_aic", "majority_vote_bic", "both"],
        default="majority_vote_aic",
        help="Which vote criterion to plot (default: majority_vote_aic).",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--title",
        default="",
        help="Optional figure title override.",
    )
    p.add_argument(
        "--legend-only",
        action="store_true",
        help="Save only the color legend (no stacked bars).",
    )
    p.add_argument(
        "--legend-ncol",
        type=int,
        default=4,
        help="Number of legend columns when --legend-only (default: 4).",
    )
    return p.parse_args()


def _default_output_path(input_csv: str, criterion: str) -> str:
    tables_dir = os.path.dirname(os.path.abspath(input_csv))
    root = os.path.dirname(tables_dir)
    plots_dir = os.path.join(root, "plots")
    tag = criterion.replace("majority_vote_", "")
    return os.path.join(plots_dir, f"model_vote_stacked_{tag}.png")


def _default_legend_path(input_csv: str, criterion: str) -> str:
    tables_dir = os.path.dirname(os.path.abspath(input_csv))
    root = os.path.dirname(tables_dir)
    plots_dir = os.path.join(root, "plots")
    tag = criterion.replace("majority_vote_", "")
    return os.path.join(plots_dir, f"model_vote_legend_{tag}.png")


def _models_present(vote_df: pd.DataFrame, criterion: str) -> List[str]:
    sub = vote_df[vote_df["criterion"] == criterion]
    if sub.empty:
        raise ValueError(f"No rows for criterion={criterion!r}")
    present = set(sub.loc[pd.to_numeric(sub["n_votes"], errors="coerce").fillna(0) > 0, "model"].astype(str))
    return _ordered_models(present)


def plot_legend_only(
    models: Sequence[str],
    output_path: str,
    ncol: int = 4,
    dpi: int = 150,
) -> str:
    """Save a standalone legend image for the given model names."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _configure_korean_font(plt)

    labels = [str(m) for m in models if str(m)]
    if not labels:
        raise ValueError("No models to draw in legend")

    handles = [
        Patch(
            facecolor=MODEL_COLORS.get(m, "#7f7f7f"),
            edgecolor="#333333",
            linewidth=0.5,
            label=m,
        )
        for m in labels
    ]
    n = len(handles)
    cols = max(1, min(int(ncol), n))

    fig, ax = plt.subplots(figsize=(max(5.0, 1.7 * cols), 1.6))
    ax.set_axis_off()
    ax.legend(
        handles=handles,
        loc="center",
        ncol=cols,
        frameon=False,
        fontsize=11,
        columnspacing=1.3,
        handlelength=1.5,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)
    return output_path


def _ordered_features(features: Iterable[str]) -> List[str]:
    known = [f for f in FEATURE_ORDER if f in features]
    extra = sorted(f for f in features if f not in FEATURE_ORDER)
    return known + extra


def _feature_display(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return str(name).replace("_", " ").strip().capitalize()


def _ordered_models(models: Iterable[str]) -> List[str]:
    known = [m for m in MODEL_ORDER if m in models]
    extra = sorted(m for m in models if m not in MODEL_ORDER)
    return known + extra


def pivot_vote_shares(vote_df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """Return features × models matrix of vote shares in [0, 1]."""
    sub = vote_df[vote_df["criterion"] == criterion].copy()
    if sub.empty:
        raise ValueError(f"No rows for criterion={criterion!r}")

    sub["n_votes"] = pd.to_numeric(sub["n_votes"], errors="coerce").fillna(0.0)
    pivot = (
        sub.pivot_table(
            index="feature",
            columns="model",
            values="n_votes",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(_ordered_features(sub["feature"].astype(str).unique()))
    )
    models = _ordered_models(pivot.columns.astype(str))
    pivot = pivot.reindex(columns=models, fill_value=0.0)
    totals = pivot.sum(axis=1).replace(0.0, np.nan)
    shares = pivot.div(totals, axis=0).fillna(0.0)
    return shares


def _segment_label(model: str, share: float) -> Optional[str]:
    """In-bar text: full model name + pct; omit shares below 15%."""
    if share < _MIN_LABEL_SHARE:
        return None
    return f"{model}\n{100.0 * share:.0f}%"


def _configure_korean_font(plt) -> None:
    """Prefer a Korean-capable system font so axis labels render correctly."""
    from matplotlib import font_manager

    # Name lookup first, then common Windows/macOS file paths.
    named = (
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in named:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return

    path_candidates = (
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    )
    for path in path_candidates:
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot_vote_stacked(
    vote_df: pd.DataFrame,
    output_path: str,
    criterion: str = "majority_vote_aic",
    title: Optional[str] = None,
    panel_label: Optional[str] = None,
    dpi: int = 150,
) -> str:
    import matplotlib.pyplot as plt

    _configure_korean_font(plt)

    shares = pivot_vote_shares(vote_df, criterion)
    features = list(shares.index.astype(str))
    feature_labels = [_feature_display(f) for f in features]
    models = list(shares.columns.astype(str))
    x = np.arange(len(features))
    bottoms = np.zeros(len(features), dtype=float)

    if panel_label is None:
        panel_label = "(a)" if str(criterion).endswith("aic") else "(b)"

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    handles = []
    for model in models:
        vals = shares[model].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, "#7f7f7f")
        bars = ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.78,
            label=model,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        if float(np.sum(vals)) > 0:
            handles.append(bars)
        text_color = "#222222" if model in _LIGHT_FILL else "white"
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            label = _segment_label(model, float(v))
            if not label:
                continue
            ax.text(
                i,
                b + v / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                linespacing=1.05,
                fontweight="semibold" if model == "GMM" else "normal",
            )
        bottoms = bottoms + vals

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{int(100 * t)}%" for t in np.linspace(0.0, 1.0, 6)])
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=20, ha="right")
    crit_label = "AIC" if criterion.endswith("aic") else "BIC"
    ax.set_ylabel(f"사용자 비중 (다수결 투표, {crit_label})")
    ax.set_xlabel("")
    # ax.set_title(title or f"특징별 모델 선택 (100% 누적, 다수결 {crit_label})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)

    if panel_label:
        ax.text(
            -0.08,
            1.02,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            clip_on=False,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(criteria: Optional[Sequence[str]] = None):
    args = parse_args()
    vote_df = pd.read_csv(args.input_csv)
    required = {"feature", "criterion", "model", "n_votes"}
    missing = required - set(vote_df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {args.input_csv}: {sorted(missing)}")

    if criteria is None:
        if args.criterion == "both":
            criteria = ["majority_vote_aic", "majority_vote_bic"]
        else:
            criteria = [args.criterion]

    saved = []
    for crit in criteria:
        if args.legend_only:
            if args.output and len(criteria) == 1:
                out = args.output
            elif args.output and len(criteria) > 1:
                root, ext = os.path.splitext(args.output)
                tag = crit.replace("majority_vote_", "")
                out = f"{root}_{tag}{ext or '.png'}"
            else:
                out = _default_legend_path(args.input_csv, crit)
            models = _models_present(vote_df, crit)
            path = plot_legend_only(
                models,
                out,
                ncol=int(args.legend_ncol),
                dpi=int(args.dpi),
            )
        else:
            if args.output and len(criteria) == 1:
                out = args.output
            elif args.output and len(criteria) > 1:
                root, ext = os.path.splitext(args.output)
                tag = crit.replace("majority_vote_", "")
                out = f"{root}_{tag}{ext or '.png'}"
            else:
                out = _default_output_path(args.input_csv, crit)
            path = plot_vote_stacked(
                vote_df,
                out,
                criterion=crit,
                title=args.title or None,
                dpi=int(args.dpi),
            )
        saved.append(path)
        print(f"Saved: {path}")
    return saved


if __name__ == "__main__":
    main()
