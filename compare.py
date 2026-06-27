import os

import numpy as np
import pandas as pd

from main import build_feature_df_for_user
from visualize import FEATURE_COLUMNS


def load_user_feature_frames(
    dataset_dir="./raw_kmt_dataset",
    user_ids=None,
    data_group="true_data",
    preprocessed_dir="results/preprocessed_kmt",
):
    """
    Return {user_label: feature_df} for requested user ids.
    Uses the same feature path as main.py (preprocessed JSON when present).
    """
    if user_ids is None:
        user_ids = [1, 2, 3, 4, 5]

    pp = preprocessed_dir if (preprocessed_dir and str(preprocessed_dir).strip()) else ""

    frames = {}
    for user_id in user_ids:
        user_label = f"user_{user_id:04d}"
        raw_path = os.path.join(dataset_dir, f"raw_kmt_user_{user_id:04d}.json")
        if not os.path.exists(raw_path):
            continue
        frames[user_label] = build_feature_df_for_user(
            raw_path,
            preprocessed_dir=pp,
            logger=None,
        )
    return frames


def plot_user_feature_histograms(
    user_frames,
    output_dir="results/compare_histograms",
    bins=40,
    density=True,
):
    """
    Plot overlaid histograms (different color per user) for each feature.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for compare histogram plotting.") from exc

    os.makedirs(output_dir, exist_ok=True)
    if not user_frames:
        return []

    users = list(user_frames.keys())
    suffix = "_".join(u.replace("user_", "") for u in users)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

    saved_paths = []
    for feature in FEATURE_COLUMNS:
        global_min = float("inf")
        global_max = float("-inf")
        per_user_vals = {}
        for user in users:
            df = user_frames[user]
            if feature not in df.columns:
                continue
            vals = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(dtype=float)
            per_user_vals[user] = vals
            if vals.size > 0:
                global_min = min(global_min, float(np.min(vals)))
                global_max = max(global_max, float(np.max(vals)))

        if global_min == float("inf"):
            continue

        if global_max - global_min < 1e-12:
            bin_edges = np.linspace(global_min - 1.0, global_max + 1.0, bins + 1)
        else:
            bin_edges = np.linspace(global_min, global_max, bins + 1)

        fig = plt.figure(figsize=(8, 5))
        for idx, user in enumerate(users):
            vals = per_user_vals.get(user, np.array([]))
            if vals.size == 0:
                continue
            plt.hist(
                vals,
                bins=bin_edges,
                alpha=0.35,
                density=density,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.4,
                label=user,
            )

        plt.title(f"Feature Histogram Comparison: {feature}")
        plt.xlabel(feature)
        plt.ylabel("density" if density else "count")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{feature}__users_{suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main():
    user_frames = load_user_feature_frames(
        dataset_dir="./raw_kmt_dataset",
        user_ids=[1, 2, 3, 4, 5],
        data_group="true_data",
    )
    paths = plot_user_feature_histograms(
        user_frames, output_dir="results/compare_histograms", bins=40, density=True
    )
    print(f"Saved {len(paths)} comparison histograms to results/compare_histograms/")


if __name__ == "__main__":
    main()
