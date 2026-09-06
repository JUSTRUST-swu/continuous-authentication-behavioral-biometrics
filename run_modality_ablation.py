"""
Compare keyboard-only vs mouse-only vs keyboard+mouse (all) authentication performance.

Uses the same split seed / ratios for every modality so splits match.
Writes each run under results/evaluation_modality/<name>/ and a comparison CSV.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd

from authentication_eval import run_authentication_eval


MODALITIES = (
    ("keyboard", "keyboard"),
    ("mouse", "mouse"),
    ("all", "all"),
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run keyboard / mouse / all authentication_eval and compare metrics."
    )
    p.add_argument(
        "--output-root",
        default="results/evaluation_modality",
        help="Root folder; each modality gets a subfolder (default: results/evaluation_modality).",
    )
    p.add_argument("--dataset-dir", default="./raw_kmt_dataset")
    p.add_argument("--preprocessed-dir", default="results/preprocessed_kmt")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--threshold-mode", default="validation_eer")
    p.add_argument("--genuine-quantile", type=float, default=0.05)
    p.add_argument("--distribution-selection", default="local_aic")
    p.add_argument("--window-size", type=float, default=5.0)
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--user-range", type=int, nargs=2, metavar=("START", "END"), default=None)
    p.add_argument("--users", type=int, nargs="+", default=None)
    return p.parse_args()


def _assert_splits_identical(paths: List[str]) -> None:
    if len(paths) < 2:
        return
    base = pd.read_csv(paths[0])
    for path in paths[1:]:
        other = pd.read_csv(path)
        if not base.equals(other):
            raise SystemExit(
                f"split_assignments mismatch between {paths[0]} and {path}. "
                "Modality comparison requires identical splits."
            )


def main():
    args = parse_args()
    if args.user_range is not None:
        lo, hi = args.user_range
        user_ids: Optional[List[int]] = list(range(int(lo), int(hi) + 1))
    elif args.users is not None:
        user_ids = list(args.users)
    else:
        user_ids = None

    os.makedirs(args.output_root, exist_ok=True)
    compare_rows = []
    split_paths: List[str] = []

    for folder_name, feature_set in MODALITIES:
        out_dir = os.path.join(args.output_root, folder_name)
        print(f"\n=== modality={folder_name} feature_set={feature_set} → {out_dir} ===")
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
            feature_set=feature_set,
            distribution_selection=args.distribution_selection,
            window_size=args.window_size,
            stride=args.stride,
        )
        summary = result["summary"]
        print(summary.to_string(index=False))
        split_path = os.path.join(out_dir, "split_assignments.csv")
        if os.path.isfile(split_path):
            split_paths.append(split_path)
        for _, row in summary.iterrows():
            compare_rows.append(
                {
                    "modality": folder_name,
                    "feature_set": feature_set,
                    "metric": row["metric"],
                    "macro": row["macro"],
                    "pooled": row["pooled"],
                }
            )

    _assert_splits_identical(split_paths)

    compare_df = pd.DataFrame(compare_rows)
    # Wide table for easy reading: one row per modality
    wide_rows = []
    for modality, _ in MODALITIES:
        sub = compare_df[compare_df["modality"] == modality]
        wide = {"modality": modality}
        for _, r in sub.iterrows():
            wide[f"{r['metric']}_macro"] = r["macro"]
            wide[f"{r['metric']}_pooled"] = r["pooled"]
        wide_rows.append(wide)
    wide_df = pd.DataFrame(wide_rows)

    long_path = os.path.join(args.output_root, "comparison_summary_long.csv")
    wide_path = os.path.join(args.output_root, "comparison_summary.csv")
    compare_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("\n=== comparison (macro) ===")
    print(wide_df.to_string(index=False))
    print(f"\nSaved: {wide_path}")
    print(f"Saved: {long_path}")
    print(f"Per-modality outputs under: {args.output_root}/{{keyboard,mouse,all}}/")


if __name__ == "__main__":
    main()
