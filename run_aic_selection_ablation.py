"""
Compare local_aic vs global_weighted_aic distribution selection.

Same split seed / ratios / feature_set for both runs so only the model-family
selection policy differs.

  local_aic:            per enrolled user, pick AIC-minimizing family on train
  global_weighted_aic:  n_used-weighted mean AIC over all users' train partitions
                        (after each user's train-only clip+log1p), then fit params
                        on the enrolled user's transformed train.
                        Shares the distribution *family* across the evaluation
                        cohort (population info on train only; not test leakage).
                        Enrollee-specific parameters are still fit locally.

Writes:
  results/evaluation_aic_selection/{local_aic,global_weighted_aic}/
  results/evaluation_aic_selection/comparison_summary.csv
  results/evaluation_aic_selection/comparison_summary_long.csv
  results/evaluation_aic_selection/model_selection_agreement.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd

from authentication_eval import run_authentication_eval


SELECTIONS = (
    ("local_aic", "local_aic"),
    ("global_weighted_aic", "global_weighted_aic"),
)

SELECTION_LABELS = {
    "local_aic": "Local AIC",
    "global_weighted_aic": "Global weighted AIC",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run local_aic vs global_weighted_aic authentication_eval and compare."
    )
    p.add_argument(
        "--output-root",
        default="results/evaluation_aic_selection",
        help="Root folder; each selection policy gets a subfolder.",
    )
    p.add_argument("--dataset-dir", default="./raw_kmt_dataset")
    p.add_argument("--preprocessed-dir", default="results/preprocessed_kmt")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--threshold-mode", default="validation_eer")
    p.add_argument("--genuine-quantile", type=float, default=0.05)
    p.add_argument(
        "--feature-set",
        default="all",
        choices=["all", "dwell", "flight", "velocity", "mouse", "keyboard"],
        help="Feature group for both runs (default: all).",
    )
    p.add_argument("--window-size", type=float, default=5.0)
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--user-range", type=int, nargs=2, metavar=("START", "END"), default=None)
    p.add_argument("--users", type=int, nargs="+", default=None)
    return p.parse_args()


def _resolve_user_ids(args) -> Optional[List[int]]:
    if args.user_range is not None:
        lo, hi = args.user_range
        return list(range(int(lo), int(hi) + 1))
    if args.users is not None:
        return list(args.users)
    return None


def _wide_from_long(compare_df: pd.DataFrame, key_col: str, key_values) -> pd.DataFrame:
    wide_rows = []
    for key in key_values:
        sub = compare_df[compare_df[key_col] == key]
        wide = {key_col: key}
        for _, r in sub.iterrows():
            wide[f"{r['metric']}_macro"] = r["macro"]
            wide[f"{r['metric']}_pooled"] = r["pooled"]
        wide_rows.append(wide)
    return pd.DataFrame(wide_rows)


def compare_model_selections(local_dir: str, global_dir: str) -> pd.DataFrame:
    """
    Per (enrolled_user, feature): which family each policy chose, and whether they agree.

    Rows missing a distribution on either side are marked agree=False and should be
    excluded from agreement_rate via summarize_model_agreement (comparable pairs only).
    """
    local_path = os.path.join(local_dir, "fitted_models.csv")
    global_path = os.path.join(global_dir, "fitted_models.csv")
    if not os.path.isfile(local_path) or not os.path.isfile(global_path):
        return pd.DataFrame()

    local_df = pd.read_csv(local_path)
    global_df = pd.read_csv(global_path)
    need = {"enrolled_user", "feature", "distribution"}
    if need - set(local_df.columns) or need - set(global_df.columns):
        return pd.DataFrame()

    merged = local_df[["enrolled_user", "feature", "distribution"]].merge(
        global_df[["enrolled_user", "feature", "distribution"]],
        on=["enrolled_user", "feature"],
        how="outer",
        suffixes=("_local_aic", "_global_weighted_aic"),
    )
    local_dist = merged["distribution_local_aic"]
    global_dist = merged["distribution_global_weighted_aic"]
    both = local_dist.notna() & global_dist.notna()
    merged["agree"] = False
    merged.loc[both, "agree"] = local_dist[both].astype(str).values == global_dist[
        both
    ].astype(str).values
    merged["comparable"] = both
    return merged.sort_values(["enrolled_user", "feature"]).reset_index(drop=True)


def summarize_model_agreement(agree_df: pd.DataFrame) -> pd.DataFrame:
    if agree_df is None or agree_df.empty:
        return pd.DataFrame(
            [{"scope": "overall", "n": 0, "n_agree": 0, "agreement_rate": float("nan")}]
        )

    def _rows_for(sub: pd.DataFrame, scope: str) -> dict:
        if "comparable" in sub.columns:
            usable = sub[sub["comparable"].fillna(False)]
        else:
            usable = sub
        if usable.empty:
            return {
                "scope": scope,
                "n": 0,
                "n_agree": 0,
                "agreement_rate": float("nan"),
            }
        ag = usable["agree"].fillna(False)
        return {
            "scope": scope,
            "n": int(len(usable)),
            "n_agree": int(ag.sum()),
            "agreement_rate": float(ag.mean()),
        }

    rows = [_rows_for(agree_df, "overall")]
    for feature, sub in agree_df.groupby("feature"):
        rows.append(_rows_for(sub, f"feature:{feature}"))
    return pd.DataFrame(rows)


def _assert_splits_identical(path_a: str, path_b: str) -> None:
    if not os.path.isfile(path_a) or not os.path.isfile(path_b):
        raise SystemExit(
            f"Missing split_assignments for AIC comparison ({path_a} / {path_b})."
        )
    same = pd.read_csv(path_a).equals(pd.read_csv(path_b))
    if not same:
        raise SystemExit(
            f"split_assignments mismatch between {path_a} and {path_b}. "
            "AIC selection comparison requires identical splits."
        )
    print(f"\nsplit_assignments identical across policies: {same}")


def main():
    args = parse_args()
    user_ids = _resolve_user_ids(args)
    os.makedirs(args.output_root, exist_ok=True)

    compare_rows = []
    out_dirs = {}

    for folder_name, selection in SELECTIONS:
        out_dir = os.path.join(args.output_root, folder_name)
        out_dirs[folder_name] = out_dir
        print(
            f"\n=== distribution_selection={selection} "
            f"feature_set={args.feature_set} → {out_dir} ==="
        )
        result = run_authentication_eval(
            dataset_dir=args.dataset_dir,
            preprocessed_dir=args.preprocessed_dir if args.preprocessed_dir.strip() else "",
            output_dir=out_dir,
            user_ids=user_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            threshold_mode=args.threshold_mode,
            genuine_quantile=args.genuine_quantile,
            feature_set=args.feature_set,
            distribution_selection=selection,
            window_size=args.window_size,
            stride=args.stride,
        )
        summary = result["summary"]
        print(summary.to_string(index=False))
        for _, row in summary.iterrows():
            compare_rows.append(
                {
                    "distribution_selection": folder_name,
                    "label": SELECTION_LABELS.get(folder_name, folder_name),
                    "feature_set": args.feature_set,
                    "metric": row["metric"],
                    "macro": row["macro"],
                    "pooled": row["pooled"],
                }
            )

    compare_df = pd.DataFrame(compare_rows)
    wide_df = _wide_from_long(
        compare_df,
        key_col="distribution_selection",
        key_values=[name for name, _ in SELECTIONS],
    )
    wide_df.insert(
        1,
        "label",
        wide_df["distribution_selection"].map(SELECTION_LABELS),
    )

    long_path = os.path.join(args.output_root, "comparison_summary_long.csv")
    wide_path = os.path.join(args.output_root, "comparison_summary.csv")
    compare_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    agree_df = compare_model_selections(
        out_dirs["local_aic"], out_dirs["global_weighted_aic"]
    )
    agree_path = os.path.join(args.output_root, "model_selection_agreement.csv")
    agree_summary_path = os.path.join(
        args.output_root, "model_selection_agreement_summary.csv"
    )
    agree_df.to_csv(agree_path, index=False)
    agree_summary = summarize_model_agreement(agree_df)
    agree_summary.to_csv(agree_summary_path, index=False)

    # Sanity: split assignments must match across policies
    _assert_splits_identical(
        os.path.join(out_dirs["local_aic"], "split_assignments.csv"),
        os.path.join(out_dirs["global_weighted_aic"], "split_assignments.csv"),
    )

    print("\n=== comparison (macro) ===")
    print(wide_df.to_string(index=False))
    print("\n=== model-family agreement (local vs global) ===")
    print(agree_summary.to_string(index=False))
    print(
        "\nNote: global_weighted_aic shares distribution *family* across users "
        "(train partitions only; no test leakage). Parameters remain per-enrollee."
    )
    print(f"\nSaved: {wide_path}")
    print(f"Saved: {long_path}")
    print(f"Saved: {agree_path}")
    print(f"Saved: {agree_summary_path}")
    print(
        f"Per-policy outputs under: {args.output_root}/"
        "{local_aic,global_weighted_aic}/"
    )


if __name__ == "__main__":
    main()
